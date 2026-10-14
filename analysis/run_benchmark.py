"""多方案基准评测主流程。

三段式设计（先广筛、再精调、最后交叉验证稳健性）：
  Stage A 全模型广筛：~37 个模型 × 重复分层 CV，固定超参，统一条件排队次
  Stage B 嵌套调参：Stage A 前 K 名进入内层网格调参的嵌套 CV，并做三重显著性比较
  Stage C 稳健性交叉确认：LOOCV / 蒙特卡洛留出 / Bootstrap 乐观度校正 / 多种子

全部建模步骤（预处理、单变量筛选、调参）都包在管道里，只在外层训练折内 fit，
因此不存在数据泄漏；多重比较同时报告 Holm、Nemenyi 与 DeLong 三种口径。
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from analysis.config import OUTPUT_DIR, RANDOM_SEED
from analysis.data_loader import data_quality_notes, load_data, split_xy
from analysis.evaluation import holm_adjust
from analysis.explain import explain_best
from analysis.inference import (delong_ci, delong_pairwise, delong_test,
                               pooled_oof, pooled_performance)
from analysis.metrics_ext import (cluster_bootstrap_diff_ci,
                                  compute_metrics, paired_patient_bootstrap_diff)
from analysis.models_ext import FAMILY, get_ext_grids, get_ext_models
from analysis.resampling import (bootstrap_optimism, evaluate_splits, loo_auc,
                                 montecarlo_splits, multi_seed_scores)
from analysis.stats_ext import merge_tests

BASELINE = "Logistic_LASSO_baseline"
BENCH_DIR = OUTPUT_DIR / "benchmark"


# --------------------------------------------------------------------------
# 单次评估逻辑（可被 joblib 并行调用）
# --------------------------------------------------------------------------

def evaluate_model_cv(X, y, name, estimator, splits, n_folds, seed,
                      grid=None, inner_folds=3, grid_n_jobs=1):
    """在一个CV划分列表上评估单个模型。

    返回 (rows: list[dict], records: list[(idx, proba)])
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    rows, records = [], []
    for i, (tr, te) in enumerate(splits):
        rep, fold = (i // n_folds, i % n_folds) if n_folds else (0, i)
        try:
            if grid:
                gs = GridSearchCV(
                    clone(estimator), grid,
                    cv=StratifiedKFold(inner_folds, shuffle=True, random_state=seed),
                    scoring="roc_auc", n_jobs=grid_n_jobs, refit=True)
                gs.fit(X[tr], y[tr])
                fitted, best_params = gs.best_estimator_, gs.best_params_
            else:
                fitted = clone(estimator).fit(X[tr], y[tr])
                best_params = {}
            proba = np.asarray(fitted.predict_proba(X[te]), dtype=np.float64)[:, 1]
            m = compute_metrics(y[te], proba)
        except Exception as exc:                       # 单个折失败不中断整体
            warnings.warn(f"[{name}] fold {i} failed: {exc}")
            m = {k: np.nan for k in
                 ["auc", "ap", "brier", "brier_skill", "log_loss", "accuracy",
                  "balanced_accuracy", "sensitivity", "specificity",
                  "sens_at_spec80", "ppv", "npv", "mcc",
                  "calibration_intercept", "calibration_slope", "ici",
                  "mean_net_benefit"]}
            best_params = {}

        rows.append({"model": name, "repeat": int(rep), "fold": int(fold),
                     "n_test": int(len(te)), **m,
                     **{f"param_{k}": v for k, v in best_params.items()}})
        records.append((te.copy(), proba.copy()))
    return name, rows, records


def _task(entry, X, y, splits, n_folds, seed, grid, inner_folds, grid_n_jobs):
    name, est = entry
    return evaluate_model_cv(X, y, name, est, splits, n_folds, seed,
                             grid=grid, inner_folds=inner_folds,
                             grid_n_jobs=grid_n_jobs)


def run_parallel(models, X, y, splits, n_folds, seed, grids=None,
                 inner_folds=3, grid_n_jobs=1, jobs=8, verbose=True):
    """跨模型并行跑 CV。返回 (fold_df, oof_records)。"""
    grids = grids or {}
    results = Parallel(n_jobs=jobs, backend="loky", verbose=5 if verbose else 0)(
        delayed(_task)((name, est), X, y, splits, n_folds, seed,
                      grids.get(name), inner_folds, grid_n_jobs)
        for name, est in models.items()
    )
    rows, oof = [], {}
    for name, r, recs in results:
        rows.extend(r)
        oof[name] = recs
    return pd.DataFrame(rows), oof


# --------------------------------------------------------------------------
# 汇总与统计
# --------------------------------------------------------------------------

CORE_METRICS = ["auc", "ap", "brier", "balanced_accuracy", "sensitivity",
                "specificity", "ppv", "npv", "mcc", "calibration_slope",
                "calibration_intercept", "ici", "mean_net_benefit"]


def summarize_folds(fold_df: pd.DataFrame, metrics=CORE_METRICS) -> pd.DataFrame:
    """按模型汇总每个指标的均值与（AUC 的）标准差，并按 AUC 降序。"""
    rows = []
    for name, g in fold_df.groupby("model"):
        rec = {"model": name, "family": FAMILY.get(name, "Other"),
               "n_folds": int(len(g))}
        for m in metrics:
            rec[f"{m}_mean"] = float(g[m].mean())
        rec["auc_sd"] = float(g["auc"].std())
        rec["auc_min"] = float(g["auc"].min())
        rec["auc_max"] = float(g["auc"].max())
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values("auc_mean", ascending=False)
    return out.reset_index(drop=True)


def family_summary(fold_df: pd.DataFrame) -> pd.DataFrame:
    """按方法族汇总：族内所有模型所有折的均值，以及该族"最佳模型"的平均 AUC。"""
    df = fold_df.copy()
    df["family"] = df["model"].map(FAMILY)
    per_model = df.groupby(["family", "model"])["auc"].mean().reset_index()
    idx = per_model.groupby("family")["auc"].idxmax()
    best = per_model.loc[idx].set_index("family")
    g = df.groupby("family")["auc"]
    out = pd.DataFrame({
        "family": g.mean().index,
        "n_models": df.groupby("family")["model"].nunique().to_numpy(),
        "auc_mean": g.mean().to_numpy(),
        "auc_sd_across_folds": g.std().to_numpy(),
        "best_model_auc": best.loc[g.mean().index, "auc"].to_numpy(),
        "best_model": best.loc[g.mean().index, "model"].to_numpy(),
    }).sort_values("auc_mean", ascending=False)
    return out.reset_index(drop=True)


def filter_covered_oof(y, oof):
    """剔除折外预测未覆盖全样本或含 NaN 的模型（否则池化与绘图会报错）。"""
    from analysis.inference import pooled_oof as _pool
    good, dropped = {}, []
    for name, recs in oof.items():
        try:
            p = _pool(y, recs)
            if np.all(np.isfinite(p)):
                good[name] = recs
                continue
        except Exception:
            pass
        dropped.append(name)
    if dropped:
        warnings.warn(f"dropped models with incomplete OOF coverage: {dropped}")
    return good, dropped


def comparisons_vs_baseline(fold_df, oof, y, baseline=BASELINE, seed=RANDOM_SEED):
    """三重显著性口径 vs 基线：Wilcoxon+Holm（折级）、DeLong（患者级）、
    两种自助 ΔAUC 置信区间。"""
    from scipy.stats import wilcoxon
    from analysis.inference import pooled_oof as _pool

    pivot = fold_df.pivot_table(index=["repeat", "fold"], columns="model",
                                values="auc")
    pooled = {k: _pool(y, v) for k, v in oof.items()}
    rows = []
    for model in pivot.columns:
        if model == baseline:
            continue
        diff = pivot[model].to_numpy() - pivot[baseline].to_numpy()
        diff = diff[np.isfinite(diff)]
        if diff.size and not np.allclose(diff, 0):
            stat, p = wilcoxon(diff)
        else:
            p = 1.0
        auc_a, auc_b, z, p_delong = delong_test(y, pooled[model], pooled[baseline])
        lo_pb, hi_pb = paired_patient_bootstrap_diff(
            y, pooled[model], pooled[baseline], seed=seed)
        lo_cb, hi_cb = cluster_bootstrap_diff_ci(diff, seed=seed)
        rows.append({
            "model": model, "auc_fold_mean": float(pivot[model].mean()),
            "auc_diff_fold": float(np.nanmean(diff)),
            "auc_diff_ci_block_lo": lo_cb, "auc_diff_ci_block_hi": hi_cb,
            "auc_diff_ci_patient_lo": lo_pb, "auc_diff_ci_patient_hi": hi_pb,
            "p_wilcoxon": float(p),
            "auc_pooled": float(auc_a),
            "auc_diff_pooled": float(auc_a - auc_b),
            "z_delong": z, "p_delong": float(p_delong),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["p_wilcoxon_holm"] = holm_adjust(out["p_wilcoxon"].to_numpy())
        out["p_delong_holm"] = holm_adjust(out["p_delong"].to_numpy())
    return out.sort_values("auc_diff_fold", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Stage A
# --------------------------------------------------------------------------

def stage_a(X, y, models, n_folds, n_repeats, seed, jobs, table_dir):
    from sklearn.model_selection import RepeatedStratifiedKFold
    print(f"\n=== Stage A：{len(models)} 模型广筛 "
          f"(外层 {n_folds} 折 × {n_repeats} 次重复分层 CV，固定超参) ===")
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats,
                                   random_state=seed)
    splits = list(rskf.split(X, y))
    t0 = time.time()
    fold_df, oof = run_parallel(models, X, y, splits, n_folds, seed, jobs=jobs)
    oof, dropped = filter_covered_oof(y, oof)
    print(f"Stage A 完成，用时 {time.time() - t0:.0f}s"
          + (f"；剔除 {dropped}" if dropped else ""))

    summary = summarize_folds(fold_df)
    fam = family_summary(fold_df)
    fold_df.to_csv(table_dir / "stage_a_fold_metrics.csv", index=False)
    summary.to_csv(table_dir / "stage_a_summary.csv", index=False)
    fam.to_csv(table_dir / "stage_a_family_summary.csv", index=False)

    # 整体检验：Friedman + Nemenyi（全部模型、每个 repeat×fold 为一个区组）
    pivot = fold_df.pivot_table(index=["repeat", "fold"], columns="model",
                                values="auc").dropna(axis=1, how="any")
    post, fried = merge_tests(pivot, BASELINE)
    post.to_csv(table_dir / "stage_a_friedman_nemenyi.csv", index=False)
    print(f"Friedman χ²={fried['statistic']:.2f}, p={fried['p_value']:.3g} "
          f"({fried['n_models']} 模型 × {fried['n_blocks']} 区组)")

    # 每个模型的 pooled OOF 性能（含自助/DeLong 双置信区间）
    pooled_tbl = pooled_performance(y, oof)
    pooled_tbl.to_csv(table_dir / "stage_a_pooled_performance.csv", index=False)
    return {"fold_df": fold_df, "oof": oof, "summary": summary, "family": fam,
            "friedman": fried, "nemenyi": post, "pooled": pooled_tbl,
            "cd": post.attrs.get("critical_difference")}


# --------------------------------------------------------------------------
# Stage B
# --------------------------------------------------------------------------

def stage_b(X, y, models, top_k, n_folds, n_repeats, inner_folds, seed, jobs,
            table_dir, stage_a_summary):
    print(f"\n=== Stage B：前 {top_k} 名 + 基线进入嵌套调参 ===")
    ranked = stage_a_summary["model"].tolist()
    finalists = [m for m in ranked[:top_k] if m in models]
    if BASELINE in models and BASELINE not in finalists:
        finalists.append(BASELINE)
    subm = {m: models[m] for m in finalists}
    grids = {m: get_ext_grids()[m] for m in finalists if m in get_ext_grids()}
    print(f"入选：{list(subm)}；带网格：{list(grids)}")

    from sklearn.model_selection import RepeatedStratifiedKFold
    splits = list(RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats,
                                          random_state=seed).split(X, y))
    t0 = time.time()
    fold_df, oof = run_parallel(subm, X, y, splits, n_folds, seed, grids=grids,
                                inner_folds=inner_folds, grid_n_jobs=1, jobs=jobs)
    oof, dropped = filter_covered_oof(y, oof)
    print(f"Stage B 完成，用时 {time.time() - t0:.0f}s"
          + (f"；剔除 {dropped}" if dropped else ""))

    summary = summarize_folds(fold_df)
    comp = comparisons_vs_baseline(fold_df, oof, y, seed=seed)
    pooled_tbl = pooled_performance(y, oof)
    fold_df.to_csv(table_dir / "stage_b_fold_metrics.csv", index=False)
    summary.to_csv(table_dir / "stage_b_summary.csv", index=False)
    comp.to_csv(table_dir / "stage_b_comparison_vs_baseline.csv", index=False)
    pooled_tbl.to_csv(table_dir / "stage_b_pooled_performance.csv", index=False)
    return {"fold_df": fold_df, "oof": oof, "summary": summary, "comp": comp,
            "pooled": pooled_tbl, "finalists": finalists}


# --------------------------------------------------------------------------
# Stage C
# --------------------------------------------------------------------------

def _stage_c_one(name, est, X, y, seed, n_boot, n_mc, mc_seeds):
    """单个模型的四种重采样评估（供 joblib 并行调用）。"""
    loo = loo_auc(X, y, clone(est))
    mc = float(np.nanmean(evaluate_splits(
        X, y, clone(est), montecarlo_splits(X, y, n_splits=n_mc, seed=seed))))
    boot = bootstrap_optimism(X, y, clone(est), n_boot=n_boot, seed=seed)
    seeds = tuple(range(1, mc_seeds + 1))
    per_seed = multi_seed_scores(X, y, clone(est), seeds=seeds,
                                 n_splits=5, n_repeats=2)
    vals = list(per_seed.values())
    return ({"model": name, "family": FAMILY.get(name, "Other"),
             "auc_loo": loo, "auc_montecarlo": mc,
             "auc_apparent": boot["apparent"], "optimism": boot["optimism"],
             "auc_corrected": boot["corrected"],
             "auc_corrected_ci_lo": boot["corrected_ci"][0],
             "auc_corrected_ci_hi": boot["corrected_ci"][1],
             "auc_632": boot["b632"], "n_boot": boot["n_boot_used"]},
            {"model": name, **{f"seed_{s}": per_seed[s] for s in seeds},
             "auc_seed_mean": float(np.mean(vals)),
             "auc_seed_sd": float(np.std(vals))})


def stage_c(X, y, models, finalists, max_models, seed, jobs, table_dir,
            n_boot=200, n_mc=100, mc_seeds=5):
    print(f"\n=== Stage C：稳健性交叉确认（最多 {max_models} 个模型）===")
    ranked_models = [m for m in finalists if m in models][:max_models]
    print(f"  模型：{ranked_models}")
    out = Parallel(n_jobs=min(jobs, max(1, len(ranked_models))), backend="loky")(
        delayed(_stage_c_one)(name, models[name], X, y, seed, n_boot, n_mc,
                              mc_seeds)
        for name in ranked_models
    )
    rows = [o[0] for o in out]
    seed_rows = [o[1] for o in out]
    for r in rows:
        print(f"  {r['model']}: LOO={r['auc_loo']:.3f} "
              f"MC={r['auc_montecarlo']:.3f} corrected={r['auc_corrected']:.3f}")

    rob = pd.DataFrame(rows)
    seed_df = pd.DataFrame(seed_rows)
    rob.to_csv(table_dir / "stage_c_robustness.csv", index=False)
    seed_df.to_csv(table_dir / "stage_c_seed_stability.csv", index=False)

    # 多种子下的秩稳定性
    from analysis.stats_ext import rank_stability
    rank_df = seed_df.set_index("model")[
        [c for c in seed_df.columns if c.startswith("seed_")]].T
    rank_df = rank_df.rank(axis=1, ascending=False)   # 每个种子内按 AUC 排名
    stab = rank_stability(rank_df)
    stab.to_csv(table_dir / "stage_c_rank_stability.csv", index=False)
    return {"robustness": rob, "seed": seed_df, "stability": stab}


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="冒烟：缩小 CV 与模型规模")
    ap.add_argument("--top-k", type=int, default=8, help="Stage B 入选模型数")
    ap.add_argument("--jobs", type=int, default=8, help="并行进程数")
    ap.add_argument("--stages", default="ABC", help="要运行的阶段，如 AB / C")
    ap.add_argument("--max-c-models", type=int, default=4, help="Stage C 模型数上限")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    n_folds, n_repeats, inner_folds = 5, 5, 3
    light = args.quick
    if args.quick:
        n_folds, n_repeats, inner_folds = 3, 1, 2

    table_dir = BENCH_DIR / "tables"
    fig_dir = BENCH_DIR / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()
    notes = data_quality_notes(df)
    X, y, feature_names = split_xy(df)
    models = get_ext_models(seed=args.seed, light=light)
    print(f"共 {len(models)} 个候选模型，队列 {len(y)} 例（事件 {int(y.sum())} 例）")

    results = {"n_models": len(models), "n_folds": n_folds,
               "n_repeats": n_repeats, "inner_folds": inner_folds,
               "n_events": int(y.sum()), "n_total": len(y), "seed": args.seed}

    if "A" in args.stages:
        a = stage_a(X, y, models, n_folds, n_repeats, args.seed, args.jobs, table_dir)
        results["stage_a"] = a
        finalists_for_c = a["summary"]["model"].tolist()
    else:
        a_summary = pd.read_csv(table_dir / "stage_a_summary.csv")
        finalists_for_c = a_summary["model"].tolist()
        results["stage_a"] = {"summary": a_summary}

    if "B" in args.stages:
        b = stage_b(X, y, models, args.top_k, n_folds, n_repeats, inner_folds,
                    args.seed, args.jobs, table_dir, results["stage_a"]["summary"])
        results["stage_b"] = b
        with open(BENCH_DIR / "stage_meta.json", "w", encoding="utf-8") as f:
            json.dump({"finalists": b["finalists"], "quick": args.quick,
                       "top_k": args.top_k, "seed": args.seed}, f, indent=2)
    else:
        meta = json.loads((BENCH_DIR / "stage_meta.json").read_text(encoding="utf-8"))
        results["stage_b"] = {"finalists": meta["finalists"]}
        for fn, key in [("stage_b_summary.csv", "summary"),
                        ("stage_b_comparison_vs_baseline.csv", "comp"),
                        ("stage_b_pooled_performance.csv", "pooled")]:
            p = table_dir / fn
            if p.exists():
                results["stage_b"][key] = pd.read_csv(p)

    if "C" not in args.stages:
        # 复用此前已完成的 Stage C 结果（内容只依赖 Stage B 的入选模型与种子）
        c_tables = {}
        for fn, key in [("stage_c_robustness.csv", "robustness"),
                        ("stage_c_seed_stability.csv", "seed"),
                        ("stage_c_rank_stability.csv", "stability")]:
            p = table_dir / fn
            if p.exists():
                c_tables[key] = pd.read_csv(p)
        if len(c_tables) == 3:
            results["stage_c"] = c_tables

    if "C" in args.stages:
        c = stage_c(X, y, models, results["stage_b"]["finalists"],
                    args.max_c_models, args.seed, args.jobs, table_dir,
                    n_boot=20 if args.quick else 200,
                    n_mc=20 if args.quick else 100)
        results["stage_c"] = c

    # ---- 图形 ----
    from analysis.bench_plots import (plot_family_boxplot, plot_rank_cd,
                                      plot_robustness_bar)
    from analysis.plots import plot_calibration, plot_dca, plot_roc

    def _safe(fn, *a, **kw):
        """单个图形失败不打断报告生成。"""
        try:
            return fn(*a, **kw)
        except Exception as exc:                        # pragma: no cover
            warnings.warn(f"plot {getattr(fn, '__name__', fn)} failed: {exc}")
            return None

    oof_a = results["stage_a"]["oof"]
    all_sorted = results["stage_a"]["summary"]["model"].tolist()
    top12 = [BASELINE] + [m for m in all_sorted if m != BASELINE][:11]
    _safe(plot_roc, y, {k: oof_a[k] for k in top12},
          fig_dir / "roc_stage_a_top12.png")
    _safe(plot_family_boxplot, results["stage_a"]["fold_df"],
          fig_dir / "auc_by_family_boxplot.png")
    if "B" in args.stages:
        _safe(plot_roc, y, results["stage_b"]["oof"], fig_dir / "roc_stage_b.png")
        _safe(plot_calibration, y, results["stage_b"]["oof"],
              fig_dir / "calibration_stage_b.png")
        _safe(plot_dca, y, results["stage_b"]["oof"], fig_dir / "dca_stage_b.png")
    if "C" in args.stages:
        _safe(plot_robustness_bar, results["stage_c"]["robustness"],
              fig_dir / "stage_c_robustness.png")
    post = results["stage_a"].get("nemenyi")
    if post is not None:
        _safe(plot_rank_cd, results["stage_a"]["friedman"]["mean_ranks"],
              results["stage_a"]["cd"], BASELINE, fig_dir / "nemenyi_ranks.png")

    # ---- SHAP：总榜第一 ----
    best = all_sorted[0]
    print(f"\n最优模型（Stage A 平均 AUC）：{best}，拟合全数据做 SHAP ...")
    est = models[best].fit(X, y)
    shap_res = explain_best(est, X, feature_names, fig_dir, seed=args.seed)
    results["best_model"] = best
    results["shap_method"] = shap_res["method"]

    # ---- 报告 ----
    from analysis.bench_report import write_benchmark_report
    path = write_benchmark_report(results, BENCH_DIR / "report.md",
                                  quality_notes=notes,
                                  shap_top=shap_res["importance"].head(10))
    print(f"\n报告写入 {path}")

    top10 = results["stage_a"]["summary"].head(10)
    print("\nStage A 前 10：")
    print(top10[["model", "family", "auc_mean", "auc_sd"]].to_string(index=False))


if __name__ == "__main__":
    main()

"""把三段式评测的全部结果写成一份可直接用于投稿的 Markdown 报告。"""
import pandas as pd


def _md(df: pd.DataFrame, fmt=".3f") -> str:
    if df is None or len(df) == 0:
        return "_（无数据）_"
    return df.to_markdown(index=False, floatfmt=fmt)


def _fmt_p(df, cols) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].map(lambda v: f"{v:.2e}" if isinstance(v, float) and v < 1e-3
                                else (f"{v:.4f}" if isinstance(v, float) else v))
    return out


def write_benchmark_report(results, out_path, quality_notes=None,
                           shap_top=None) -> str:
    quality_notes = quality_notes or []
    sa = results.get("stage_a", {})
    sb = results.get("stage_b", {})
    sc = results.get("stage_c", {})
    n_models = results.get("n_models")
    n_folds = results.get("n_folds")
    n_repeats = results.get("n_repeats")
    inner = results.get("inner_folds")
    best = results.get("best_model")

    L = []
    L.append("# CRT 应答预测：多方案机器学习基准评测报告")
    L.append("")
    L.append("**任务**：在 `data.xlsx`（163 例 CRT 患者）上，以 `response` 为因变量、"
             "其余 28 列为候选自变量，系统比较**传统统计模型与 AI 模型**的预测能力。")
    L.append("")
    L.append("## 1. 设计总览")
    L.append("")
    L.append(f"- **候选模型数**：{n_models} 个，覆盖 6 个方法族"
             "（线性/正则、核与贝叶斯、树与集成、深度学习、集成之集成、筛选消融）")
    L.append(f"- **Stage A（广筛）**：外层 {n_folds} 折 × {n_repeats} 次重复分层 CV，"
             "固定超参，全部模型在同一套划分上公平排队")
    L.append(f"- **Stage B（精调）**：A 阶段前若干名进入嵌套 CV，"
             f"内层 {inner} 折网格搜索优化 AUC")
    L.append("- **Stage C（稳健性）**：留一法（LOOCV）、蒙特卡洛重复留出（100 次 80/20）、"
             "Bootstrap 乐观度校正（200 次）、多种子重复 CV")
    L.append(f"- **队列**：{results.get('n_total')} 例，"
             f"应答 {results.get('n_events')} 例"
             f"（应答率 {results.get('n_events') / results.get('n_total') * 100:.1f}%）")
    L.append(f"- **随机种子**：{results.get('seed')}（所有模型、CV 划分与调参统一）")
    L.append("- **防泄漏**：预处理、单变量筛选、超参选择全部封装在管道内，"
             "只在外层训练折内 fit")
    L.append("")

    # ---------------- Stage A ----------------
    if "summary" in sa:
        L.append("## 2. Stage A：全模型广筛结果")
        L.append("")
        L.append("### 2.1 总榜（按外层折平均 AUC 降序，前 15）")
        L.append("")
        top = sa["summary"].head(15)
        L.append(_md(top[["model", "family", "auc_mean", "auc_sd", "auc_min",
                          "auc_max", "ap_mean", "brier_mean", "mean_net_benefit_mean",
                          "calibration_slope_mean"]]))
        L.append("")
        L.append("完整表格见 `tables/stage_a_summary.csv` "
                 f"（共 {len(sa['summary'])} 个模型）。")
        L.append("")
        L.append("### 2.2 按方法族汇总")
        L.append("")
        L.append(_md(sa.get("family"), fmt=".3f"))
        L.append("")
        L.append("### 2.3 池化折外预测性能（含 DeLong 与自助双置信区间，前 10）")
        L.append("")
        pooled = sa.get("pooled")
        if pooled is not None:
            L.append(_md(pooled.head(10)[["model", "auc", "auc_ci_delong_lo",
                                          "auc_ci_delong_hi", "auc_ci_boot_lo",
                                          "auc_ci_boot_hi", "calib_slope",
                                          "calib_intercept", "brier"]]))
        L.append("")

    fried = sa.get("friedman")
    post = sa.get("nemenyi")
    if fried:
        L.append("### 2.4 多模型整体检验：Friedman + Nemenyi")
        L.append("")
        L.append(f"- Friedman 总括检验：χ² = **{fried['statistic']:.3f}**，"
                 f"p = **{fried['p_value']:.3g}**（{fried['n_models']} 个模型 × "
                 f"{fried['n_blocks']} 个重采样区组）")
        cd = post.attrs.get("critical_difference") if post is not None else None
        if cd is not None:
            L.append(f"- Nemenyi 临界差 CD = **{cd:.2f}**（α=0.05）："
                     "平均秩落在基线 ±CD 区间内的模型，与基线无统计学显著差异")
        L.append("")
        if post is not None:
            L.append("与已发表 LASSO 基线的 Nemenyi 事后检验（Holm 校正）：")
            L.append("")
            L.append(_md(_fmt_p(post, ["p_nemenyi", "p_holm"]), fmt=".3f"))
        L.append("")
        L.append("![Nemenyi 秩图](../figures/nemenyi_ranks.png)")
        L.append("")

    # ---------------- Stage B ----------------
    if "summary" in sb:
        L.append("## 3. Stage B：嵌套调参后的前排名")
        L.append("")
        L.append(_md(sb["summary"][["model", "family", "auc_mean", "auc_sd",
                                    "auc_min", "auc_max", "mean_net_benefit_mean",
                                    "calibration_slope_mean"]], fmt=".3f"))
        L.append("")
        L.append("### 3.1 与 LASSO 基线的三重显著性比较")
        L.append("")
        comp = sb.get("comp")
        if comp is not None:
            L.append("**口径说明**")
            L.append("")
            L.append("- `p_wilcoxon_holm`：以每个外层折 AUC 为配对单位的 Wilcoxon "
                     "符号秩检验 + Holm 校正（配对说法正确的是这一种）")
            L.append("- `p_delong_holm`：在池化折外预测上做 DeLong 检验 + Holm 校正"
                     "（患者层面、阈值无关）")
            L.append("- `auc_diff_ci_*`：ΔAUC 的两种自助置信区间，"
                     "block=折块自助，patient=患者层面配对自助")
            L.append("")
            L.append(_md(_fmt_p(comp, ["p_wilcoxon", "p_wilcoxon_holm",
                                       "p_delong", "p_delong_holm"]), fmt=".3f"))
        L.append("")
        L.append("![ROC](../figures/roc_stage_b.png)")
        L.append("")
        L.append("![Calibration](../figures/calibration_stage_b.png)")
        L.append("")
        L.append("![DCA](../figures/dca_stage_b.png)")
        L.append("")

    # ---------------- Stage C ----------------
    if "robustness" in sc:
        L.append("## 4. Stage C：结论在多种重采样下是否站得住")
        L.append("")
        L.append(_md(sc["robustness"], fmt=".3f"))
        L.append("")
        L.append("`auc_apparent`→`optimism`→`auc_corrected` 是 Bootstrap 乐观度校正链条："
                 "模型在全数据上的表观性能减去自抽样估计的乐观度，"
                 "得到对外部数据的近似期望性能。")
        L.append("")
        L.append("![Robustness](../figures/stage_c_robustness.png)")
        L.append("")
        if "stability" in sc:
            L.append("### 4.1 多种子下的名次稳定性")
            L.append("")
            L.append(_md(sc["stability"], fmt=".2f"))
            L.append("")
            L.append("`times_best`/`times_top3` 表示在多个随机种子下该模型获得第一 / "
                     "进入前三的次数——用于区分真实方法优势与随机种子运气。")
            L.append("")

    # ---------------- SHAP ----------------
    if shap_top is not None and len(shap_top):
        L.append(f"## 5. 最优模型（{best}）的可解释性")
        L.append("")
        L.append(f"SHAP 方法：**{results.get('shap_method')}**"
                 "（在全队列上拟合，仅用于解释，不影响上述任何评估结果）")
        L.append("")
        L.append(_md(shap_top.to_frame("mean_abs_shap"), fmt=".4f"))
        L.append("")
        L.append("![SHAP bar](../figures/shap_bar.png)")
        L.append("")

    if quality_notes:
        L.append("## 6. 数据质量备注")
        L.append("")
        L += [f"- {n}" for n in quality_notes]
        L.append("")

    # ---------------- 方法与局限 ----------------
    L.append("## 7. 可复用的 Methods 文本")
    L.append("")
    L.append(
        f"Candidate predictors comprised all {int(results.get('n_total'))} patients' "
        "28 baseline variables. We benchmarked "
        f"{n_models} modelling approaches spanning penalised regression, "
        "linear/quadratic discriminant analysis, naive Bayes, kernel machines, "
        "Gaussian processes, k-nearest neighbours, tree ensembles, gradient boosting, "
        "feed-forward networks, TabNet, an FT-Transformer, soft voting, stacked "
        "generalisation and a non-negative super learner. Every learner was embedded "
        "in a leakage-free pipeline in which imputation, scaling, optional univariate "
        "screening and hyper-parameter tuning were fitted inside the outer training "
        "fold only. Stage A used "
        f"stratified {n_folds}-fold cross-validation repeated {n_repeats} times with "
        "fixed default hyper-parameters; Stage B re-evaluated the leading candidates "
        f"with nested cross-validation (inner {inner}-fold grid search optimising AUC). "
        "Discrimination was quantified by AUC and average precision; calibration by "
        "Brier score, calibration slope and intercept; clinical utility by decision- "
        "curve net benefit. Differences against the published LASSO reference model "
        "were tested with paired Wilcoxon signed-rank tests over folds (Holm "
        "correction), DeLong tests on pooled out-of-fold predictions, and bootstrap "
        "confidence intervals. A Friedman omnibus test with Nemenyi post-hoc analysis "
        "(Demšar 2006) was used for the overall multi-model comparison. Leading models "
        "were further re-validated with leave-one-out cross-validation, 100 replicate "
        "Monte-Carlo holdouts, bootstrap optimism correction (Harrell 1996) and "
        "multi-seed repetition.")
    L.append("")
    L.append("## 8. 局限")
    L.append("")
    L += [
        "- 单中心、样本量有限（163 例、28 个候选变量，EPV 约 3），"
        "所有结论本质上是内部验证，尚需外部队列确认。",
        "- 地真手写数据与超参网格有限：广筛阶段使用固定超参，"
        "因此某方法族的代表模型可能未达到其最优配置（Stage B 已对前排名补做调参）。",
        "- 深度学习方案的样本量远低于其常规需求，其劣势应理解为"
        "「该数据结构下不适配」，而非否定方法本身。",
        "- 净收益区间依赖阈概率范围假设，临床解读时应结合具体决策场景。",
    ]
    L.append("")
    L.append("## 9. 复现")
    L.append("")
    L.append("```bash")
    L.append("python -m analysis.run_benchmark            # 完整三段式评测")
    L.append("python -m analysis.run_benchmark --quick    # 冒烟（缩小 CV 与模型规模）")
    L.append("```")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L), encoding="utf-8")
    return str(out_path)

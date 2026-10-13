import argparse

import torch

from analysis.config import (FIGURE_DIR, N_FOLDS, N_INNER_FOLDS, N_REPEATS,
                             RANDOM_SEED, TABLE_DIR)
from analysis.data_loader import data_quality_notes, load_data, split_xy
from analysis.evaluation import compare_to_baseline, run_comparison, summarize
from analysis.explain import explain_best
from analysis.models import get_grids, get_models
from analysis.plots import plot_calibration, plot_dca, plot_roc
from analysis.report import write_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="1 repeat, no grids - smoke test of the full pipeline")
    args = ap.parse_args()

    n_folds, n_repeats, inner_folds = N_FOLDS, N_REPEATS, N_INNER_FOLDS
    light = False
    if args.quick:
        n_repeats, inner_folds, light = 1, 2, True

    df = load_data()
    notes = data_quality_notes(df)
    X, y, feature_names = split_xy(df)
    models = get_models(seed=RANDOM_SEED, light=light)
    grids = {} if light else get_grids(seed=RANDOM_SEED)

    print(f"Running {len(models)} models: {list(models.keys())}")
    print(f"CV: outer {n_folds}-fold x {n_repeats} repeats, inner {inner_folds}-fold"
          + ("" if light else " grid search"))
    fold_df, oof = run_comparison(X, y, models, grids=grids,
                                  n_folds=n_folds, n_repeats=n_repeats,
                                  inner_folds=inner_folds, seed=RANDOM_SEED)
    summary = summarize(fold_df)
    comp = compare_to_baseline(fold_df)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(TABLE_DIR / "fold_results.csv", index=False)
    summary.to_csv(TABLE_DIR / "model_summary.csv", index=False)
    comp.to_csv(TABLE_DIR / "model_comparison_vs_baseline.csv", index=False)

    plot_roc(y, oof, FIGURE_DIR / "roc_curves.png")
    plot_calibration(y, oof, FIGURE_DIR / "calibration_curves.png")
    plot_dca(y, oof, FIGURE_DIR / "dca_curves.png")

    best_model = summary.iloc[0]["model"]
    print(f"\nBest model: {best_model}; fitting on full data for SHAP...")
    best_est = models[best_model].fit(X, y)  # 全数据拟合，仅用于 SHAP 展示
    shap_res = explain_best(best_est, X, feature_names, FIGURE_DIR, seed=RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    report_path = write_report(
        summary, comp, best_model, shap_res["method"],
        TABLE_DIR.parent / "report.md",
        extra={"n_folds": n_folds, "n_repeats": n_repeats,
               "inner_folds": inner_folds, "n_events": int(y.sum()),
               "n_total": len(y), "seed": RANDOM_SEED, "device": device},
        quality_notes=notes,
        shap_top=shap_res["importance"].head(10))

    print("\n" + summary.to_string(index=False))
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()

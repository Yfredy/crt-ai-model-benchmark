"""基准评测专用图形：族间分布、Nemenyi 秩与临界差、稳健性对比。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.models_ext import FAMILY

PALETTE = {
    "Linear & Regularized": "#1B4F9C",
    "Kernel & Bayes": "#2E86AB",
    "Trees & Ensembles": "#2A9D5C",
    "Deep Learning": "#C0392B",
    "Ensemble of Ensembles": "#8E44AD",
    "Ablation (screening)": "#7F8C8D",
    "Other": "#BDC3C7",
}


def _color(model):
    return PALETTE.get(FAMILY.get(model, "Other"), "#BDC3C7")


def plot_family_boxplot(fold_df, out_path, metric="auc"):
    """按模型给出折 AUC 分布，按家族着色，按中位数排序。"""
    df = fold_df.copy()
    df["family"] = df["model"].map(FAMILY)
    order = (df.groupby("model")[metric].median()
               .sort_values(ascending=False).index.tolist())
    data = [df.loc[df["model"] == m, metric].dropna().to_numpy() for m in order]
    colors = [_color(m) for m in order]

    fig_h = max(5.0, 0.26 * len(order))
    fig, ax = plt.subplots(figsize=(9, fig_h))
    # matplotlib ≥3.9 用 tick_labels 取代了 labels
    try:
        bplot = ax.boxplot(data, vert=False, tick_labels=order, patch_artist=True,
                           widths=0.65, showmeans=True,
                           meanprops={"marker": "D", "markerfacecolor": "white",
                                      "markeredgecolor": "black", "markersize": 3.5})
    except TypeError:  # pragma: no cover - 旧版 matplotlib
        bplot = ax.boxplot(data, vert=False, labels=order, patch_artist=True,
                           widths=0.65, showmeans=True)
    for patch, c in zip(bplot["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    for whisk in bplot["whiskers"]:
        whisk.set_linewidth(0.7)
    ax.axvline(0.5, color="crimson", ls="--", lw=0.9)
    for lbl, c in zip(ax.get_yticklabels(), colors):
        lbl.set_color(c)
    ax.set_xlabel(f"Out-of-fold {metric.upper()} (per fold)")
    ax.set_title(f"Model performance distribution by method family "
                 f"({df['repeat'].nunique()} repeated {fold_df['model'].nunique()} models)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[f], alpha=0.8)
               for f in [k for k in PALETTE if k in set(df["family"])]]
    ax.legend(handles, [k for k in PALETTE if k in set(df["family"])],
              loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=190)
    plt.close(fig)
    return str(out_path)


def plot_rank_cd(mean_ranks, cd, control, out_path):
    """平均秩条形图 + Nemenyi 临界差带：带内的模型与对照无显著差异。"""
    ranks = pd.Series(mean_ranks).sort_values()
    ctrl = float(ranks.get(control, np.nan))
    fig, ax = plt.subplots(figsize=(9, max(5.0, 0.26 * len(ranks))))
    colors = [_color(m) for m in ranks.index]
    ax.barh(range(len(ranks)), ranks.to_numpy(), color=colors, alpha=0.85)
    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels(ranks.index)
    ax.invert_yaxis()
    if np.isfinite(ctrl):
        ax.axvspan(ctrl - cd, ctrl + cd, color="orange", alpha=0.18,
                   label=f"Not significantly different from {control} (CD={cd:.2f})")
        ax.axvline(ctrl, color="black", ls="--", lw=1.0)
    ax.set_xlabel("Mean rank (1 = best)")
    ax.set_title("Nemenyi critical-difference diagram")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=190)
    plt.close(fig)
    return str(out_path)


def plot_robustness_bar(robustness_df, out_path):
    """LOOCV / 蒙特卡洛 / Bootstrap 乐观度校正 三种重采样下的 AUC 对比。"""
    df = robustness_df.sort_values("auc_corrected", ascending=False)
    idx = np.arange(len(df))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 0.75 * len(df) + 2.2))
    ax.bar(idx - w, df["auc_loo"], w, label="LOOCV", color="#1B4F9C", alpha=0.85)
    ax.bar(idx, df["auc_montecarlo"], w, label="Monte-Carlo holdout (100x)",
           color="#2A9D5C", alpha=0.85)
    ax.bar(idx + w, df["auc_corrected"], w,
           label="Bootstrap optimism-corrected", color="#C0392B", alpha=0.85)
    lo = np.clip(df["auc_corrected"] - df["auc_corrected_ci_lo"], 0, None)
    hi = np.clip(df["auc_corrected_ci_hi"] - df["auc_corrected"], 0, None)
    ax.errorbar(idx + w, df["auc_corrected"], yerr=[lo, hi], fmt="none",
                ecolor="black", elinewidth=0.8, capsize=2)
    ax.axhline(0.5, color="crimson", ls="--", lw=0.9)
    ax.set_xticks(idx)
    ax.set_xticklabels(df["model"], rotation=30, ha="right")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("Robustness of leading models under three resampling schemes")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=190)
    plt.close(fig)
    return str(out_path)

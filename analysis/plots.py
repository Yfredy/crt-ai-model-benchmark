import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve

from analysis.dca import net_benefit, net_benefit_treat_all, net_benefit_treat_none
from analysis.inference import pooled_oof


def _pooled(y, oof_records):
    """`analysis.inference.pooled_oof` 的 plot 层别名。

    共享实现（多重复取均值）只有一份，位于 `analysis.inference`；这里仅把
    覆盖不全时的 `ValueError` 翻译成 plot 层历史上的 `AssertionError`，以保持
    既有调用方/测试的行为不变。新代码请直接使用 `pooled_oof`（抛 `ValueError`）。
    """
    try:
        return pooled_oof(y, oof_records)
    except ValueError as exc:
        raise AssertionError(str(exc)) from exc


def plot_roc(y, oof_records: dict, out_path) -> str:
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, recs in oof_records.items():
        p = _pooled(y, recs)
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title("ROC (pooled out-of-fold predictions)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)
    return str(out_path)


def plot_calibration(y, oof_records: dict, out_path, n_bins=5) -> str:
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, recs in oof_records.items():
        p = _pooled(y, recs)
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed proportion")
    ax.set_title("Calibration (pooled out-of-fold predictions)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)
    return str(out_path)


def plot_dca(y, oof_records: dict, out_path,
             thresholds=np.arange(0.01, 0.99, 0.01)) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(thresholds, net_benefit_treat_all(y, thresholds), "k--", label="Treat all")
    ax.plot(thresholds, net_benefit_treat_none(thresholds), "k:", label="Treat none")
    for name, recs in oof_records.items():
        p = _pooled(y, recs)
        ax.plot(thresholds, net_benefit(y, p, thresholds), label=name)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision Curve Analysis (pooled out-of-fold predictions)")
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(-0.2, 0.8)  # 聚焦临床关注区间，避免 treat-all 深负值压扁图形
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)
    return str(out_path)

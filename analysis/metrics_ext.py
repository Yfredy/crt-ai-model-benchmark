"""扩展评估指标：在现有 evaluation.py / inference.py / dca.py 之上补齐维度。

职责边界（避免同一份数据出现两套口径）：
- DeLong 检验、AUC 自助置信区间、pooled OOF、校准截距/斜率
  → 一律复用 `analysis.inference`；
- 净收益 → 一律复用 `analysis.dca`；
- Youden 灵敏度/特异度 → 一律复用 `analysis.evaluation`；
本模块只提供上面这些地方尚未覆盖的指标：Brier skill score、ICI、
固定特异度下的灵敏度、PPV/NPV、MCC、平均净收益、配对 ΔAUC 自助区间、
以及"折块"级别的聚类自助。
"""
import numpy as np
from scipy.stats import norm
from sklearn.metrics import brier_score_loss, confusion_matrix, matthews_corrcoef
from sklearn.metrics import average_precision_score, roc_auc_score

from analysis.dca import net_benefit
from analysis.evaluation import youden_sens_spec
from analysis.inference import calibration_intercept_slope

EPS = 1e-6
NET_BENEFIT_GRID = np.round(np.arange(0.05, 1.00, 0.05), 2)


def clip_prob(p, eps=EPS):
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)


def logit(p, eps=EPS):
    pc = clip_prob(p, eps)
    return np.log(pc / (1 - pc))


# --------------------------------------------------------------------------
# 校准度补充指标
# --------------------------------------------------------------------------

def brier_skill_score(y_true, y_prob):
    """相对"恒预测患病率"的 Brier 改进比例（>0 表示优于无信息预测）。"""
    y = np.asarray(y_true, dtype=int)
    prevalence = float(np.mean(y))
    ref = float(np.mean((y - prevalence) ** 2))
    bs = float(brier_score_loss(y, clip_prob(y_prob)))
    return float(1 - bs / ref) if ref > 0 else float("nan")


def _loess_fit(x, y, xout, span=0.5):
    """无第三方依赖的局部加权线性回归（tricube 核）。"""
    n = len(x)
    out = np.empty(len(xout))
    for k, xo in enumerate(xout):
        d = np.abs(x - xo)
        r = max(d[np.argsort(d)[int(np.ceil(span * n)) - 1]], 1e-12)
        w = np.clip(1 - (d / r) ** 3, 0, None) ** 3
        if w.sum() <= 1e-12:
            out[k] = np.mean(y)
            continue
        sw = np.sqrt(w)
        Xd = np.column_stack([np.ones_like(x), x]) * sw[:, None]
        beta, *_ = np.linalg.lstsq(Xd, y * sw, rcond=None)
        out[k] = beta[0] + beta[1] * xo
    return np.clip(out, 0.0, 1.0)


def integrated_calibration_index(y_true, y_prob, span=0.5, n_grid=101):
    """ICI：预测概率与 loess 校准后实际概率的平均绝对偏差（越小越好）。"""
    y = np.asarray(y_true, dtype=int)
    p = clip_prob(y_prob)
    grid = np.linspace(0.0, 1.0, n_grid)
    obs = _loess_fit(p, y.astype(float), grid, span=span)
    return float(np.mean(np.abs(obs - grid)))


# --------------------------------------------------------------------------
# 阈值相关指标
# --------------------------------------------------------------------------

def threshold_metrics(y_true, y_prob, thr):
    """在给定阈值下的混淆矩阵衍生指标。"""
    y = np.asarray(y_true, dtype=int)
    pred = (np.asarray(y_prob) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "ppv": tp / (tp + fp) if (tp + fp) else np.nan,
        "npv": tn / (tn + fn) if (tn + fn) else np.nan,
        "fp": int(fp), "fn": int(fn),
    }


def sensitivity_at_specificity(y_true, y_prob, target_spec=0.8):
    """达到指定特异度时能达到的灵敏度（临床常用的工作点表述）。"""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(np.asarray(y_true, dtype=int), np.asarray(y_prob))
    spec = 1 - fpr
    idx = np.where(spec >= target_spec)[0]
    if len(idx) == 0:
        return float("nan")
    return float(tpr[idx].max())


def mean_net_benefit(y_true, y_prob, grid=NET_BENEFIT_GRID):
    """阈值 5%–95% 网格上的平均净收益（便于单数字排序临床效用）。"""
    y = np.asarray(y_true, dtype=int)
    return float(np.nanmean(net_benefit(y, clip_prob(y_prob), grid)))


# --------------------------------------------------------------------------
# 全量指标汇总
# --------------------------------------------------------------------------

def compute_metrics(y_true, y_prob, nb_grid=NET_BENEFIT_GRID):
    """一个样本集上的全部点估计指标（不含置信区间）。"""
    y = np.asarray(y_true, dtype=int)
    p = clip_prob(y_prob)
    if len(np.unique(y)) < 2:
        raise ValueError("y_true must contain both classes")

    sens, spec = youden_sens_spec(y, p)
    intercept, slope = calibration_intercept_slope(y, p)
    tm = threshold_metrics(y, p, 0.5)
    return {
        "auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "brier_skill": brier_skill_score(y, p),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "accuracy": float(np.mean((p >= 0.5).astype(int) == y)),
        "balanced_accuracy": float(0.5 * (tm["sensitivity"] + tm["specificity"])),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "sens_at_spec80": sensitivity_at_specificity(y, p, 0.8),
        "ppv": tm["ppv"],
        "npv": tm["npv"],
        "mcc": float(matthews_corrcoef(y, (p >= 0.5).astype(int))),
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
        "ici": integrated_calibration_index(y, p),
        "mean_net_benefit": mean_net_benefit(y, p, nb_grid),
    }


METRIC_KEYS = list(compute_metrics(np.array([0, 1, 0, 1, 1, 0, 1, 0]),
                                   np.array([0.1, 0.9, 0.4, 0.7, 0.8, 0.2, 0.6, 0.3])
                                   ).keys())

HIGHER_IS_BETTER = {
    "auc": True, "ap": True, "brier": False, "brier_skill": True,
    "log_loss": False, "accuracy": True, "balanced_accuracy": True,
    "sensitivity": True, "specificity": True, "sens_at_spec80": True,
    "ppv": True, "npv": True, "mcc": True,
    "calibration_intercept": False,   # |intercept| / |slope-1| 才是距离，另行处理
    "calibration_slope": False,
    "ici": False, "mean_net_benefit": True,
}


# --------------------------------------------------------------------------
# 配对 ΔAUC 的自助置信区间
# --------------------------------------------------------------------------

def paired_patient_bootstrap_diff(y_true, p_a, p_b, n_boot=2000, alpha=0.05,
                                  seed=2025):
    """同一批样本上两模型 ΔAUC 的患者层面配对自助置信区间。"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true, dtype=int)
    pa, pb = clip_prob(p_a), clip_prob(p_b)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    diffs = np.asarray(diffs, dtype=np.float64)
    if diffs.size == 0:
        return float("nan"), float("nan")
    return (float(np.percentile(diffs, 100 * alpha / 2)),
            float(np.percentile(diffs, 100 * (1 - alpha / 2))))


def cluster_bootstrap_diff_ci(diffs, n_boot=5000, alpha=0.05, seed=2025):
    """"折块"级别自助：适配重复 CV 中同一样本多次出现的配对结构。

    diffs 为例如每个 (repeat, fold) 块上的 AUC 差值数组。
    """
    diffs = np.asarray(diffs, dtype=np.float64)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array([diffs[rng.integers(0, diffs.size, diffs.size)].mean()
                      for _ in range(n_boot)])
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def normal_ci(diffs, alpha=0.05):
    """块差值的正态近似置信区间（自助法的对照，用于交叉核对结论）。"""
    d = np.asarray(diffs, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return float("nan"), float("nan")
    se = d.std(ddof=1) / np.sqrt(d.size)
    z = norm.ppf(1 - alpha / 2)
    return float(d.mean() - z * se), float(d.mean() + z * se)

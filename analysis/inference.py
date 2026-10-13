import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from analysis.config import RANDOM_SEED
from analysis.evaluation import _holm, youden_sens_spec


def _midrank(x):
    """1-based midrank，并列值取平均秩。"""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    r = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and xs[j + 1] == xs[i]:
            j += 1
        r[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = r
    return out


def delong_roc_variance(y_true, y_scores):
    """y_scores: (k, n)。返回 (aucs (k,), cov (k, k))。"""
    y = np.asarray(y_true).astype(int)
    S = np.atleast_2d(np.asarray(y_scores, dtype=float))
    pos = S[:, y == 1]
    neg = S[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = S.shape[0]
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        v01, v10 = pos[r], neg[r]
        tx[r] = _midrank(v01)
        ty[r] = _midrank(v10)
        tz[r] = _midrank(np.concatenate([v01, v10]))
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    # Sun & Xu (2014) 的结构分量必须按 n / m 归一化，否则方差被放大 O(m*n) 倍
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s01 = np.atleast_2d(np.cov(v01) if k > 1 else np.var(v01[0], ddof=1))
    s10 = np.atleast_2d(np.cov(v10) if k > 1 else np.var(v10[0], ddof=1))
    return aucs, s01 / m + s10 / n


def delong_test(y_true, prob_a, prob_b):
    """双侧 DeLong 检验，H0：两模型 AUC 相等。返回 (auc_a, auc_b, z, p)。"""
    aucs, cov = delong_roc_variance(y_true, np.vstack([prob_a, prob_b]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 0.0, 1.0
    z = float((aucs[0] - aucs[1]) / np.sqrt(var))
    return float(aucs[0]), float(aucs[1]), z, float(2 * norm.sf(abs(z)))


def delong_ci(y_true, prob, alpha=0.05):
    """AUC 的 DeLong 解析置信区间。返回 (auc, lo, hi)。"""
    aucs, cov = delong_roc_variance(y_true, np.asarray(prob, dtype=float).reshape(1, -1))
    se = float(np.sqrt(max(cov[0, 0], 0.0)))
    z = norm.ppf(1 - alpha / 2)
    return float(aucs[0]), float(aucs[0] - z * se), float(aucs[0] + z * se)


def bootstrap_auc_ci(y_true, y_prob, n_boot=2000, alpha=0.05, seed=RANDOM_SEED):
    """百分位 bootstrap AUC 置信区间。返回 (auc, lo, hi, boots)。"""
    y = np.asarray(y_true).astype(int); p = np.asarray(y_prob, dtype=float)
    point = float(roc_auc_score(y, p))
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], p[idx]))
    boots = np.asarray(boots, dtype=float)
    if boots.size == 0:
        return point, point, point, boots
    return (point, float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))), boots)


def calibration_intercept_slope(y_true, y_prob):
    """在 logit(p) 上重新拟合 logistic，返回 (截距, 斜率)。理想校准为 (0, 1)。"""
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=10000, tol=1e-10).fit(z, np.asarray(y_true))
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def pooled_oof(y, oof_records):
    """把 [(test_index, proba), ...] 拼回与原样本对齐的 OOF 概率向量；
    多次重复覆盖同一样本时取均值（重复 CV 的 averaged OOF）。"""
    total = np.zeros(len(y)); count = np.zeros(len(y))
    for idx, proba in oof_records:
        total[idx] += proba
        count[idx] += 1
    if not np.all(count > 0):
        raise ValueError("OOF predictions must cover all samples")
    return total / count


def pooled_performance(y, oof_by_model: dict, n_boot=2000, seed=RANDOM_SEED) -> pd.DataFrame:
    """每个模型在 pooled OOF 上的完整性能表。"""
    y = np.asarray(y).astype(int)
    rows = []
    for name, recs in oof_by_model.items():
        p = pooled_oof(y, recs)
        auc, lo, hi, _ = bootstrap_auc_ci(y, p, n_boot=n_boot, seed=seed)
        _, dlo, dhi = delong_ci(y, p)
        intercept, slope = calibration_intercept_slope(y, p)
        sens, spec = youden_sens_spec(y, p)
        rows.append({"model": name, "n": int(len(y)), "events": int(y.sum()),
                     "auc": auc, "auc_ci_boot_lo": lo, "auc_ci_boot_hi": hi,
                     "auc_ci_delong_lo": dlo, "auc_ci_delong_hi": dhi,
                     "auprc": float(average_precision_score(y, p)),
                     "brier": float(np.mean((p - y) ** 2)),
                     "calib_intercept": intercept, "calib_slope": slope,
                     "sensitivity": sens, "specificity": spec})
    return (pd.DataFrame(rows).sort_values("auc", ascending=False)
            .reset_index(drop=True))


def delong_pairwise(y, oof_by_model: dict,
                    baseline="Logistic_LASSO_baseline") -> pd.DataFrame:
    """pooled OOF 上每个模型 vs 基线的 DeLong 检验 + Holm 校正。"""
    y = np.asarray(y).astype(int)
    pooled = {k: pooled_oof(y, v) for k, v in oof_by_model.items()}
    if baseline not in pooled:
        raise KeyError(f"baseline '{baseline}' not in {sorted(pooled)}")
    rows = []
    for name, p in pooled.items():
        if name == baseline:
            continue
        auc_a, auc_b, z, pv = delong_test(y, p, pooled[baseline])
        rows.append({"model": name, "auc": auc_a, "auc_baseline": auc_b,
                     "auc_diff": auc_a - auc_b, "z": z, "p_delong": pv})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = _holm(out["p_delong"].to_numpy())
    return out.sort_values("auc_diff", ascending=False).reset_index(drop=True)

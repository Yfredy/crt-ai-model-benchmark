"""多种重采样验证方案。

单一 CV 方案（哪怕重复多次）仍可能给出偶然结论。临床预测模型实践中常用
三种互补的数据划分方式来交叉验证：
  - LOOCV：留一法，训练折最大、无随机性，小样本下偏差最小；
  - 蒙特卡洛重复留出：100 次随机 80/20 分层划分，考察方差；
  - Bootstrap 乐观度校正（Harrell 1996 / Efron 的 .632 估计）：
    用自抽样估计"表观性能 - 真实性能"的系统性高估，直接给出校正后性能。
三者与重复分层 CV 一致时，结论才可称为稳健。
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def loo_splits(n):
    """留一法：n 个划分，每个测试的size为 1。"""
    all_idx = np.arange(n)
    return [(np.setdiff1d(all_idx, [i]), np.array([i])) for i in range(n)]


def montecarlo_splits(X, y, n_splits=100, test_size=0.2, seed=2025):
    """蒙特卡洛重复留出：n_splits 次随机分层 80/20 划分。"""
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,
                                 random_state=seed)
    return list(sss.split(X, y))


def repeated_stratified_splits(X, y, n_splits=5, n_repeats=5, seed=2025):
    from sklearn.model_selection import RepeatedStratifiedKFold
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=seed)
    return list(rskf.split(X, y))


def evaluate_splits(X, y, estimator, splits, metric=roc_auc_score,
                    return_predictions=False):
    """在给定划分上评估单个估计器，返回每个测试折的指标（与 OOF 记录）。

    estimator 会被 clone，因此同一个实例可安全复用于多组划分。
    """
    from sklearn.base import clone
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    scores, records = [], []
    for i, (tr, te) in enumerate(splits):
        est = clone(estimator).fit(X[tr], y[tr])
        proba = est.predict_proba(X[te])[:, 1]
        if len(np.unique(y[te])) < 2:      # LOOCV 下单个样本无法算 AUC
            scores.append(np.nan)
        else:
            scores.append(float(metric(y[te], proba)))
        if return_predictions:
            records.append({"repeat": 0, "fold": i, "idx": te.copy(),
                            "proba": proba.copy()})
    if return_predictions:
        return np.array(scores, dtype=np.float64), records
    return np.array(scores, dtype=np.float64)


def bootstrap_optimism(X, y, estimator, n_boot=200, alpha=0.05, seed=2025,
                       metric=roc_auc_score):
    """Bootstrap 乐观度校正（Harrell 1996）与 .632 估计。

    返回 dict(apparent, optimism, corrected, corrected_ci, b632, per_boot)。
    corrected = 全数据拟合的表观性能 - 平均乐观度。
    """
    from sklearn.base import clone
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n = len(y)
    rng = np.random.default_rng(seed)

    apparent_model = clone(estimator).fit(X, y)
    apparent = float(metric(y, apparent_model.predict_proba(X)[:, 1]))

    train_scores, test_scores = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        Xb, yb = X[idx], y[idx]
        if len(np.unique(yb)) < 2:
            continue
        est = clone(estimator).fit(Xb, yb)
        try:
            s_train = float(metric(yb, est.predict_proba(Xb)[:, 1]))
            s_test = float(metric(y, est.predict_proba(X)[:, 1]))
        except ValueError:
            continue
        train_scores.append(s_train)
        test_scores.append(s_test)

    train_scores = np.asarray(train_scores, dtype=np.float64)
    test_scores = np.asarray(test_scores, dtype=np.float64)
    if len(train_scores) == 0:
        return {"apparent": apparent, "optimism": np.nan, "corrected": np.nan,
                "corrected_ci": (np.nan, np.nan), "b632": np.nan,
                "n_boot_used": 0}

    optimism = float(np.mean(train_scores - test_scores))
    corrected = float(apparent - optimism)
    # 校正值的自助分布：固定"表观性能"，让乐观度估计的变异决定区间宽度。
    # （若改为 train_scores - optimism，则区间中心会偏离 corrected 本身。）
    corrected_samples = apparent - (train_scores - test_scores)
    lo = float(np.percentile(corrected_samples, 100 * alpha / 2))
    hi = float(np.percentile(corrected_samples, 100 * (1 - alpha / 2)))
    b632 = float(0.368 * apparent + 0.632 * np.mean(test_scores))
    return {"apparent": apparent, "optimism": optimism, "corrected": corrected,
            "corrected_ci": (lo, hi), "b632": b632,
            "n_boot_used": int(len(train_scores))}


def loo_auc(X, y, estimator, metric=roc_auc_score):
    """留一法的 AUC：单个留出样本算不出 AUC，因此先把 163 个 LOO 预测拼成
    一条完整的折外概率向量，再一次性计算（这是 LOO 报告 AUC 的标准做法）。"""
    from sklearn.base import clone
    y = np.asarray(y)
    p = np.full(len(y), np.nan, dtype=np.float64)
    _, records = evaluate_splits(X, y, clone(estimator), loo_splits(len(y)),
                                 return_predictions=True)
    for r in records:
        p[r["idx"]] = np.asarray(r["proba"], dtype=np.float64)
    if np.any(~np.isfinite(p)):
        raise ValueError("LOO predictions do not cover all samples")
    return float(metric(y, p))


def multi_seed_scores(X, y, estimator, seeds=(1, 2, 3, 4, 5), n_splits=5,
                      n_repeats=2, metric=roc_auc_score):
    """在多个随机种子下重复跑重复分层 CV，用于秩稳定性分析。

    返回 {seed: 平均指标}。
    """
    from sklearn.base import clone
    out = {}
    for s in seeds:
        splits = repeated_stratified_splits(X, y, n_splits=n_splits,
                                           n_repeats=n_repeats, seed=s)
        sc = evaluate_splits(X, y, clone(estimator), splits, metric=metric)
        out[s] = float(np.nanmean(sc))
    return out

import numpy as np
from scipy.stats import chi2
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from analysis.preprocessing import AutoPreprocessor


class UnivariateLRFilter(BaseEstimator, TransformerMixin):
    """对每列拟合单变量 logistic，似然比检验 P<p_threshold 则保留（论文 Methods 原方法）。"""

    def __init__(self, p_threshold=0.05):
        self.p_threshold = p_threshold

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        p1 = np.clip(y.mean(), 1e-9, 1 - 1e-9)
        ll_null = np.sum(y * np.log(p1) + (1 - y) * np.log(1 - p1))
        kept = []
        for j in range(X.shape[1]):
            xj = X[:, [j]]
            if np.nanstd(xj) == 0:
                continue
            xj = np.where(np.isnan(xj), np.nanmedian(xj), xj)
            lr = LogisticRegression(max_iter=1000).fit(xj, y)
            p_hat = np.clip(lr.predict_proba(xj)[:, 1], 1e-9, 1 - 1e-9)
            ll = np.sum(y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat))
            p_value = chi2.sf(max(2 * (ll - ll_null), 0.0), 1)
            if p_value < self.p_threshold:
                kept.append(j)
        if not kept:  # 无显著特征时退化为全部保留，保证下游可用
            kept = list(range(X.shape[1]))
        self.kept_idx_ = kept
        return self

    def transform(self, X):
        return np.asarray(X, dtype=np.float64)[:, self.kept_idx_]


def make_baseline_pipeline(seed=2025) -> Pipeline:
    return Pipeline([
        ("pre", AutoPreprocessor()),
        ("univar", UnivariateLRFilter(p_threshold=0.05)),
        ("clf", LogisticRegression(
            penalty="l1", solver="liblinear", C=1.0,
            max_iter=5000, random_state=seed,
        )),
    ])

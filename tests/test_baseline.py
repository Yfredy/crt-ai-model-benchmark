import numpy as np
from sklearn.pipeline import Pipeline

from analysis.baseline import UnivariateLRFilter, make_baseline_pipeline

rng = np.random.default_rng(0)
n = 120
y = rng.integers(0, 2, n)
X = np.column_stack([
    y * 2.0 + rng.normal(0, 0.3, n),   # 强信号
    rng.normal(0, 1, n),               # 噪声
    rng.normal(0, 1, n),               # 噪声
])

def test_filter_keeps_signal_drops_noise():
    f = UnivariateLRFilter(p_threshold=0.05).fit(X, y)
    assert 0 in f.kept_idx_
    assert 1 not in f.kept_idx_
    assert 2 not in f.kept_idx_

def test_filter_fallback_when_nothing_significant():
    X_noise = rng.normal(0, 1, (60, 3))
    y_noise = rng.integers(0, 2, 60)
    f = UnivariateLRFilter().fit(X_noise, y_noise)
    assert f.kept_idx_ == [0, 1, 2]  # 全保留，不允许输出 0 列

def test_transform_shape():
    f = UnivariateLRFilter().fit(X, y)
    Xt = f.transform(X)
    assert Xt.shape[0] == n and Xt.shape[1] == len(f.kept_idx_)

def test_baseline_pipeline_is_pipeline_and_fits():
    pipe = make_baseline_pipeline(seed=2025)
    assert isinstance(pipe, Pipeline)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (n, 2)

def test_baseline_on_real_data_selects_features():
    # 论文报告单变量筛选后 14 个显著变量
    from analysis.data_loader import load_data, split_xy
    from analysis.preprocessing import AutoPreprocessor
    X_real, y_real, _ = split_xy(load_data())
    f = UnivariateLRFilter(p_threshold=0.05).fit(AutoPreprocessor().fit_transform(X_real, y_real), y_real)
    assert len(f.kept_idx_) > 0
    print(f"selected {len(f.kept_idx_)} variables (paper: 14)")

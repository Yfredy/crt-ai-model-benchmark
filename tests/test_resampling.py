import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis.resampling import (bootstrap_optimism, evaluate_splits, loo_splits,
                                 montecarlo_splits, multi_seed_scores,
                                 repeated_stratified_splits)


def _data(n=120, p=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ np.r_[[1.5, -1.0], np.zeros(p - 2)])))).astype(int)
    return X, y


def _est():
    return Pipeline([("s", StandardScaler()), ("c", LogisticRegression())])


def test_loo_splits_covers_each_sample_once():
    splits = loo_splits(30)
    assert len(splits) == 30
    assert all(len(te) == 1 for _, te in splits)
    assert sorted(int(te[0]) for _, te in splits) == list(range(30))
    for tr, _ in splits:
        assert len(tr) == 29


def test_montecarlo_splits_stratified_count():
    X, y = _data(n=100)
    splits = montecarlo_splits(X, y, n_splits=20, test_size=0.2, seed=1)
    assert len(splits) == 20
    for tr, te in splits:
        assert len(te) == 20 and len(tr) == 80
        assert abs(y[te].mean() - y.mean()) < 0.2


def test_evaluate_splits_shapes_and_finite():
    X, y = _data(n=90)
    splits = repeated_stratified_splits(X, y, n_splits=3, n_repeats=2, seed=2)
    scores, records = evaluate_splits(X, y, _est(), splits, return_predictions=True)
    assert len(scores) == 6 and len(records) == 6
    assert np.all(np.isfinite(scores))


def test_repeated_stratified_splits_deterministic():
    X, y = _data(n=60)
    a = repeated_stratified_splits(X, y, n_splits=5, n_repeats=2, seed=11)
    b = repeated_stratified_splits(X, y, n_splits=5, n_repeats=2, seed=11)
    assert all(np.array_equal(x[1], y_[1]) for x, y_ in zip(a, b))


def test_bootstrap_optimism_corrects_downwards():
    X, y = _data(n=100, p=6)
    res = bootstrap_optimism(X, y, _est(), n_boot=15, seed=4)
    assert res["n_boot_used"] > 0
    assert res["optimism"] >= -1e-6      # 表观性能通常高于样本外性能
    assert res["corrected"] <= res["apparent"] + 1e-6
    lo, hi = res["corrected_ci"]
    assert lo <= hi


def test_multi_seed_scores_length_and_range():
    X, y = _data(n=80)
    out = multi_seed_scores(X, y, _est(), seeds=(1, 2, 3), n_splits=5,
                            n_repeats=1)
    assert len(out) == 3
    assert all(0.0 <= v <= 1.0 for v in out.values())

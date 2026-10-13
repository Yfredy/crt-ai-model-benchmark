import time

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from analysis.models import get_models, get_grids, make_stacking_pipeline

EXPECTED_KEYS = {
    "Logistic_LASSO_baseline", "Logistic_L2", "RF", "SVM_rbf",
    "XGBoost", "LightGBM", "MLP", "TabNet", "Stacking",
}

rng = np.random.default_rng(0)
X = rng.normal(0, 1, (80, 6))
y = (X[:, 0] - X[:, 1] + rng.normal(0, 0.5, 80) > 0).astype(int)

@pytest.fixture(scope="module")
def fitted():
    models = get_models(seed=2025, light=True)
    out = {}
    for name, m in models.items():
        m.fit(X, y)
        out[name] = m
    return out

def test_model_keys():
    assert set(get_models().keys()) == EXPECTED_KEYS

def test_all_models_predict_proba(fitted):
    for name, m in fitted.items():
        proba = m.predict_proba(X)
        assert proba.shape == (80, 2), name
        assert np.all((proba >= 0) & (proba <= 1)), name
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5), name

def test_grids_match_models():
    grids = get_grids(light=True)
    assert grids == {}
    grids_full = get_grids(seed=1)
    assert set(grids_full.keys()) <= EXPECTED_KEYS
    assert "Logistic_LASSO_baseline" not in grids_full

def test_determinism(fitted):
    m1 = get_models(seed=2025, light=True)["RF"]
    m2 = get_models(seed=2025, light=True)["RF"]
    p1 = m1.fit(X, y).predict_proba(X)
    p2 = m2.fit(X, y).predict_proba(X)
    assert np.allclose(p1, p2)


# ---------------------------------------------------------------- Stacking ---

def _synthetic_10col(n=80, seed=2025):
    """n=80、10 列、线性可分带噪的合成数据（seed 固定）。"""
    rng = np.random.default_rng(seed)
    Xs = rng.normal(0, 1, (n, 10))
    signal = Xs[:, 0] - 0.8 * Xs[:, 1] + 0.5 * Xs[:, 2]
    ys = (signal + rng.normal(0, 0.5, n) > 0).astype(int)
    return Xs, ys


def test_stacking_is_registered():
    assert "Stacking" in get_models(light=True)


def test_stacking_predict_proba_shape_and_rowsums():
    Xs, ys = _synthetic_10col()
    m = get_models(seed=2025, light=True)["Stacking"]
    proba = m.fit(Xs, ys).predict_proba(Xs)
    assert proba.shape == (80, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)


def test_stacking_grid_keys_are_settable():
    grids = get_grids(seed=2025)["Stacking"]
    pipe = make_stacking_pipeline(2025)
    for params in grids:
        for k, v in params.items():
            pipe.set_params(**{k: v[0]})
            assert pipe.get_params()[k] == v[0], k


def test_stacking_determinism():
    Xs, ys = _synthetic_10col()
    p1 = make_stacking_pipeline(2025, light=True).fit(Xs, ys).predict_proba(Xs)
    p2 = make_stacking_pipeline(2025, light=True).fit(Xs, ys).predict_proba(Xs)
    np.testing.assert_array_equal(p1, p2)


def test_stacking_is_cloneable():
    cloned = clone(make_stacking_pipeline(2025))
    assert isinstance(cloned, Pipeline)
    assert not hasattr(cloned, "classes_")
    assert cloned.get_params(deep=True)["clf__cv"] == 3


def test_stacking_light_fit_predict_under_30s():
    Xs, ys = _synthetic_10col()
    m = make_stacking_pipeline(2025, light=True)
    t0 = time.perf_counter()
    m.fit(Xs, ys)
    m.predict_proba(Xs)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0, f"light-mode Stacking took {elapsed:.2f}s"

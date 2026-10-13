import numpy as np
import pytest

from analysis.models import get_models, get_grids

EXPECTED_KEYS = {
    "Logistic_LASSO_baseline", "Logistic_L2", "RF", "SVM_rbf",
    "XGBoost", "LightGBM", "MLP", "TabNet",
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

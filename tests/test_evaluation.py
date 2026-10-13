import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from analysis.evaluation import (calibration_slope, youden_sens_spec,
                                 run_comparison, summarize, compare_to_baseline)
from analysis.preprocessing import AutoPreprocessor

rng = np.random.default_rng(2025)
n = 120
X = rng.normal(0, 1, (n, 6))
y = (X[:, 0] * 1.5 - X[:, 1] + rng.normal(0, 0.5, n) > 0).astype(int)

def test_calibration_slope_perfect_ranking():
    p = np.clip(y * 0.9 + 0.05, 1e-6, 1 - 1e-6)
    assert calibration_slope(y, p) > 5.0

def test_youden_perfect():
    p = y.astype(float)
    sens, spec = youden_sens_spec(y, p)
    assert sens == 1.0 and spec == 1.0

def test_youden_range():
    rng2 = np.random.default_rng(1)
    p = rng2.random(n)
    sens, spec = youden_sens_spec(y, p)
    assert 0.0 <= sens <= 1.0 and 0.0 <= spec <= 1.0

@pytest.fixture(scope="module")
def result():
    models = {
        "L2": Pipeline([("pre", AutoPreprocessor()),
                        ("clf", LogisticRegression(max_iter=1000))]),
        "RF": Pipeline([("pre", AutoPreprocessor()),
                        ("clf", RandomForestClassifier(n_estimators=30, random_state=1))]),
    }
    return run_comparison(X, y, models, grids=None,
                          n_folds=2, n_repeats=2, inner_folds=2, seed=2025)

def test_run_comparison_structure(result):
    fold_df, oof = result
    assert set(fold_df.columns) >= {"model","repeat","fold","auc","brier","sensitivity","specificity","calibration_slope"}
    assert len(fold_df) == 2 * 2 * 2  # models x folds x repeats
    assert fold_df["auc"].between(0, 1).all()
    for name, recs in oof.items():
        total = sum(len(idx) for idx, _ in recs)
        assert total == n  # OOF 覆盖全部样本

def test_determinism(result):
    fold_df, _ = result
    models = {
        "L2": Pipeline([("pre", AutoPreprocessor()),
                        ("clf", LogisticRegression(max_iter=1000))]),
    }
    fold_df2, _ = run_comparison(X, y, models, grids=None,
                                 n_folds=2, n_repeats=2, inner_folds=2, seed=2025)
    a = fold_df[fold_df.model == "L2"]["auc"].to_numpy()
    b = fold_df2["auc"].to_numpy()
    assert np.allclose(a, b)

def test_summarize_sorted(result):
    s = summarize(result[0])
    assert list(s["mean_auc"]) == sorted(s["mean_auc"], reverse=True)
    assert set(s["model"]) == {"L2", "RF"}

def test_compare_to_baseline(result):
    comp = compare_to_baseline(result[0], baseline="L2")
    assert list(comp["model"]) == ["RF"]
    assert "p_holm" in comp.columns and comp["p_holm"].between(0, 1).all()

def test_grid_search_path():
    # 提供 grids 时应走 GridSearchCV 且不报错
    models = {
        "RF": Pipeline([("pre", AutoPreprocessor()),
                        ("clf", RandomForestClassifier(random_state=1))]),
    }
    grids = {"RF": [{"clf__n_estimators": [10, 30]}]}
    fold_df, oof = run_comparison(X, y, models, grids=grids,
                                  n_folds=2, n_repeats=1, inner_folds=2, seed=2025)
    assert len(fold_df) == 2
    assert sum(len(idx) for idx, _ in oof["RF"]) == n

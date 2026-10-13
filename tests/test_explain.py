import numpy as np

from analysis.explain import explain_best
from analysis.models import get_models

rng = np.random.default_rng(0)
X = rng.normal(0, 1, (60, 4))
y = (X[:, 0] - X[:, 1] + rng.normal(0, 0.3, 60) > 0).astype(int)
names = [f"f{i}" for i in range(4)]

def test_explain_rf(tmp_path):
    model = get_models(seed=1, light=True)["RF"]
    model.fit(X, y)
    result = explain_best(model, X, names, tmp_path, seed=1)
    assert result["method"] == "TreeExplainer"
    assert set(result["importance"].index) == set(names)
    assert list(result["importance"].values) == sorted(
        result["importance"].values, reverse=True)  # 按均值|SHAP|降序
    # 信号列 f0/f1 应排在噪声列前面
    top2 = set(result["importance"].index[:2])
    assert top2 == {"f0", "f1"}
    assert (tmp_path / "shap_summary.png").exists()
    assert (tmp_path / "shap_bar.png").exists()

def test_explain_xgboost(tmp_path):
    model = get_models(seed=1, light=True)["XGBoost"]
    model.fit(X, y)
    result = explain_best(model, X, names, tmp_path, seed=1)
    assert result["method"] == "TreeExplainer"
    assert len(result["importance"]) == 4

def test_explain_kernel_path(tmp_path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from analysis.preprocessing import AutoPreprocessor
    model = Pipeline([("pre", AutoPreprocessor()),
                      ("clf", LogisticRegression(max_iter=500))]).fit(X, y)
    result = explain_best(model, X, names, tmp_path, seed=1)
    assert result["method"] == "KernelExplainer"
    assert len(result["importance"]) == 4

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from analysis.models_ext import FAMILY, get_ext_grids, get_ext_models


def _data(n=140, p=20, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:4] = [1.5, -1.2, 0.9, 0.7]
    logit = X @ beta
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


ALL = ["Logistic_LASSO_baseline", "Logistic_LASSO_full", "Logistic_Ridge",
       "Logistic_ElasticNet", "Stepwise_Logistic_AIC", "LDA", "LDA_Shrinkage",
       "QDA", "GaussianNB", "Logistic_Poly2", "SVM_linear", "SVM_rbf",
       "SVM_poly", "GaussianProcess_RBF", "KNN", "DecisionTree",
       "RandomForest", "ExtraTrees", "GradientBoosting",
       "HistGradientBoosting", "XGBoost", "LightGBM", "AdaBoost",
       "Bagging_Trees", "MLP_Small", "MLP", "MLP_Deep", "TabNet",
       "TabNet_Deep", "FT_Transformer", "Voting_Soft_Hetero",
       "Stacking_Logit", "SuperLearner_NNLS", "SVM_rbf_screened",
       "XGBoost_screened", "MLP_screened", "RandomForest_screened"]


def test_zoo_size_and_families():
    models = get_ext_models()
    assert len(models) >= 35
    fams = {FAMILY[m] for m in models}
    for f in ["Linear & Regularized", "Kernel & Bayes", "Trees & Ensembles",
              "Deep Learning", "Ensemble of Ensembles", "Ablation (screening)"]:
        assert f in fams, f
    assert set(models) == set(ALL)


@pytest.mark.parametrize("name", ALL)
def test_every_model_fits_and_scores(name):
    X, y = _data()
    est = get_ext_models(light=True)[name]
    est.fit(X[:100], y[:100])
    proba = est.predict_proba(X[100:])
    assert proba.shape[1] == 2
    assert np.all(np.isfinite(proba))
    assert np.all(proba >= 0) and np.all(proba <= 1)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_stepwise_respects_max_k():
    from analysis.models_ext import StepwiseLogistic
    X, y = _data(n=120, p=12)
    m = StepwiseLogistic(max_k=3).fit(X, y)
    assert 1 <= len(m.selected_) <= 3


def test_stacking_uses_inner_cv_meta_features():
    from analysis.models_ext import LeakSafeStacking
    X, y = _data(n=120, p=10)
    m = LeakSafeStacking(base_models=["lr", "lda"], n_inner_folds=3).fit(X, y)
    assert len(m.fitted_bases_) == 2
    assert np.all(np.isfinite(m.predict_proba(X)))


def test_superlearner_weights_nonneg_and_sum_one():
    from analysis.models_ext import NegativeEnsembleMeta
    rng = np.random.default_rng(0)
    Z = rng.uniform(size=(80, 4))
    y = (rng.random(80) < 0.5).astype(int)
    m = NegativeEnsembleMeta(constrain=True).fit(Z, y)
    assert np.all(m.coef_ >= -1e-8)
    assert abs(m.coef_.sum() - 1.0) < 1e-6


def test_ft_transformer_deterministic():
    from analysis.models_ext import FTTransformerClassifier
    X, y = _data(n=100, p=8)
    a = FTTransformerClassifier(max_epochs=12, seed=5).fit(X, y)
    b = FTTransformerClassifier(max_epochs=12, seed=5).fit(X, y)
    pa = a.predict_proba(X)[:, 1]
    pb = b.predict_proba(X)[:, 1]
    assert np.allclose(pa, pb, atol=1e-6)


def test_grids_reference_existing_params():
    models = get_ext_models()
    grids = get_ext_grids()
    for name, grid in grids.items():
        assert name in models, f"grid for unknown model {name}"
        est = models[name]
        for g in grid:
            for key in g:
                # 逐步校验嵌套参数名是否存在（如 clf__estimator__C）
                obj = est
                for part in key.split("__"):
                    if part == "clf":
                        obj = est.named_steps["clf"]
                    elif hasattr(obj, "get_params") and part in obj.get_params():
                        obj = obj.get_params()[part]
                    else:
                        raise AssertionError(f"{name}: unknown param {key}")

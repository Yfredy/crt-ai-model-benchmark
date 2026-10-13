import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from analysis.config import RANDOM_SEED
from analysis.preprocessing import AutoPreprocessor
from analysis.baseline import make_baseline_pipeline


class TabNetWrapper(BaseEstimator, ClassifierMixin):
    """TabNetClassifier 的 sklearn 兼容封装：固定种子、内部留出集早停、有 GPU 用 GPU。"""

    def __init__(self, n_d=8, n_a=8, n_steps=3, gamma=1.5, lambda_sparse=1e-3,
                 max_epochs=100, patience=15, batch_size=32, seed=RANDOM_SEED):
        self.n_d = n_d; self.n_a = n_a; self.n_steps = n_steps
        self.gamma = gamma; self.lambda_sparse = lambda_sparse
        self.max_epochs = max_epochs; self.patience = patience
        self.batch_size = batch_size; self.seed = seed

    def fit(self, X, y):
        import torch
        from pytorch_tabnet.tab_model import TabNetClassifier
        from sklearn.model_selection import train_test_split

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_ = TabNetClassifier(
            n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps,
            gamma=self.gamma, lambda_sparse=self.lambda_sparse,
            seed=self.seed, verbose=0, device_name=device,
        )
        self.model_.fit(
            X_tr, y_tr, eval_set=[(X_val, y_val)],
            eval_metric=["auc"],
            max_epochs=self.max_epochs, patience=self.patience,
            batch_size=min(self.batch_size, len(X_tr)),
            virtual_batch_size=min(16, max(2, len(X_tr) // 2)),
        )
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        proba = np.asarray(self.model_.predict_proba(np.asarray(X, dtype=np.float64)))
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba
        p = proba.ravel()
        return np.column_stack([1 - p, p])


def _pipe(clf) -> Pipeline:
    return Pipeline([("pre", AutoPreprocessor()), ("clf", clf)])


def get_models(seed=RANDOM_SEED, light=False) -> dict:
    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16), alpha=0.5, activation="relu",
        max_iter=50 if light else 3000, early_stopping=not light,
        validation_fraction=0.15, random_state=seed,
    )
    tabnet = TabNetWrapper(
        seed=seed,
        max_epochs=5 if light else 200,
        patience=2 if light else 25,
    )
    rf = RandomForestClassifier(
        n_estimators=30 if light else 500, random_state=seed, n_jobs=1)
    xgb = XGBClassifier(
        n_estimators=10 if light else 200, max_depth=2, learning_rate=0.1,
        subsample=0.8, reg_lambda=1.0, eval_metric="logloss",
        random_state=seed, n_jobs=1, verbosity=0)
    lgbm = LGBMClassifier(
        n_estimators=10 if light else 200, num_leaves=7,
        min_child_samples=10, random_state=seed, n_jobs=1, verbose=-1)
    return {
        "Logistic_LASSO_baseline": make_baseline_pipeline(seed),
        "Logistic_L2": _pipe(LogisticRegression(
            penalty="l2", solver="lbfgs", C=1.0, max_iter=5000, random_state=seed)),
        "RF": _pipe(rf),
        "SVM_rbf": _pipe(SVC(probability=True, random_state=seed)),
        "XGBoost": _pipe(xgb),
        "LightGBM": _pipe(lgbm),
        "MLP": _pipe(mlp),
        "TabNet": _pipe(tabnet),
    }


def get_grids(seed=RANDOM_SEED, light=False) -> dict:
    if light:
        return {}
    return {
        "RF": [
            {"clf__n_estimators": [300, 500], "clf__max_features": ["sqrt", "log2"]},
        ],
        "SVM_rbf": [
            {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", 0.01]},
        ],
        "XGBoost": [
            {"clf__max_depth": [2, 3], "clf__n_estimators": [100, 200]},
        ],
        "LightGBM": [
            {"clf__num_leaves": [7, 15], "clf__min_child_samples": [10, 20]},
        ],
        "MLP": [
            {"clf__alpha": [0.1, 1.0],
             "clf__hidden_layer_sizes": [(32, 16), (16, 8)]},
        ],
        "TabNet": [
            {"clf__n_steps": [3, 4], "clf__n_d": [8, 16]},
        ],
    }

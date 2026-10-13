import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from pathlib import Path

TREE_MODELS = ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier")


def _final_estimator(model):
    return model[-1] if hasattr(model, "steps") else model


def _transform_features(model, X):
    if hasattr(model, "steps") and model.steps[0][0] == "pre":
        return model.steps[0][1].transform(X)
    return np.asarray(X, dtype=np.float64)


def explain_best(model, X, feature_names, out_dir, seed=2025) -> dict:
    X = np.asarray(X, dtype=np.float64)
    Xt = _transform_features(model, X)
    clf = _final_estimator(model)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if type(clf).__name__ in TREE_MODELS:
        method = "TreeExplainer"
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xt)
    else:
        method = "KernelExplainer"
        bg = shap.sample(Xt, min(60, len(Xt)), random_state=seed)
        explainer = shap.KernelExplainer(
            lambda d: clf.predict_proba(d)[:, 1], bg, seed=seed)
        sv = explainer.shap_values(Xt, nsamples=100)

    sv = np.asarray(sv)
    if sv.ndim == 3:          # (n, features, classes) 或 (classes, n, features)
        if sv.shape[-1] == 2:
            sv = sv[..., 1]
        elif sv.shape[0] == 2:
            sv = sv[1]

    importance = pd.Series(np.abs(sv).mean(axis=0), index=feature_names)
    importance = importance.sort_values(ascending=False)

    plt.figure()
    shap.summary_plot(sv, Xt, feature_names=feature_names, show=False)
    plt.tight_layout(); plt.savefig(out_dir / "shap_summary.png", dpi=200)
    plt.close("all")

    plt.figure()
    importance[::-1].plot(kind="barh", figsize=(8, max(4, 0.3 * len(feature_names))))
    plt.xlabel("mean |SHAP value|")
    plt.tight_layout(); plt.savefig(out_dir / "shap_bar.png", dpi=200)
    plt.close("all")

    return {"importance": importance, "method": method}

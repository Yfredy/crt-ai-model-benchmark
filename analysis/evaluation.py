import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold

from analysis.config import N_FOLDS, N_INNER_FOLDS, N_REPEATS, RANDOM_SEED


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_slope(y_true, y_prob) -> float:
    z = _logit(np.asarray(y_prob)).reshape(-1, 1)
    y = np.asarray(y_true)
    lr = LogisticRegression(C=1e6, max_iter=10000, tol=1e-10).fit(z, y)
    return float(lr.coef_[0, 0])


def youden_sens_spec(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(tpr[i]), float(1 - fpr[i])  # sensitivity, specificity


def _fold_metrics(y_true, y_prob):
    sens, spec = youden_sens_spec(y_true, y_prob)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "sensitivity": sens,
        "specificity": spec,
        "calibration_slope": calibration_slope(y_true, y_prob),
    }


def run_comparison(X, y, models, grids=None, n_folds=N_FOLDS, n_repeats=N_REPEATS,
                   inner_folds=N_INNER_FOLDS, seed=RANDOM_SEED):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    grids = grids or {}
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats,
                                   random_state=seed)
    rows = []
    oof_records = {}
    for name, est in models.items():
        oof_records[name] = []
        splits = list(rskf.split(X, y))
        for i, (tr, te) in enumerate(splits):
            rep, fold = i // n_folds, i % n_folds
            grid = grids.get(name)
            if grid:
                gs = GridSearchCV(
                    clone(est), grid,
                    cv=StratifiedKFold(inner_folds, shuffle=True, random_state=seed),
                    scoring="roc_auc", n_jobs=1, refit=True)
                gs.fit(X[tr], y[tr])
                fitted = gs.best_estimator_
            else:
                fitted = clone(est).fit(X[tr], y[tr])
            proba = fitted.predict_proba(X[te])[:, 1]
            metrics = _fold_metrics(y[te], proba)
            rows.append({"model": name, "repeat": rep, "fold": fold,
                         "n_test": len(te), **metrics})
            oof_records[name].append((te.copy(), proba.copy()))
    return pd.DataFrame(rows), oof_records


def summarize(fold_df: pd.DataFrame) -> pd.DataFrame:
    g = fold_df.groupby("model")["auc"]
    out = pd.DataFrame({
        "model": g.mean().index,
        "mean_auc": g.mean().to_numpy(),
        "sd_auc": g.std().to_numpy(),
        "min_auc": g.min().to_numpy(),
        "max_auc": g.max().to_numpy(),
    }).sort_values("mean_auc", ascending=False).reset_index(drop=True)
    return out


def _holm(pvals):
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted_sorted = np.maximum.accumulate((m - np.arange(m)) * pvals[order])
    adjusted = np.minimum(adjusted_sorted, 1.0)
    out = np.empty(m)
    out[order] = adjusted
    return out


def compare_to_baseline(fold_df: pd.DataFrame,
                        baseline="Logistic_LASSO_baseline") -> pd.DataFrame:
    pivot = fold_df.pivot_table(index=["repeat", "fold"], columns="model",
                                values="auc")
    rows = []
    for model in pivot.columns:
        if model == baseline:
            continue
        diff = pivot[model] - pivot[baseline]
        if np.allclose(diff, 0):
            stat, p = np.nan, 1.0
        else:
            stat, p = wilcoxon(diff)
        rows.append({"model": model, "mean_auc_diff": diff.mean(), "p_wilcoxon": p})
    comp = pd.DataFrame(rows)
    if len(comp):
        comp["p_holm"] = _holm(comp["p_wilcoxon"].to_numpy())
    return comp.sort_values("mean_auc_diff", ascending=False).reset_index(drop=True)

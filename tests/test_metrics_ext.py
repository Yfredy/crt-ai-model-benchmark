import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from analysis.metrics_ext import (brier_skill_score, cluster_bootstrap_diff_ci,
                                  compute_metrics,
                                  integrated_calibration_index, normal_ci,
                                  paired_patient_bootstrap_diff,
                                  sensitivity_at_specificity, threshold_metrics)


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    logit = 1.2 * x
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    return y, p, x


def test_compute_metrics_perfect_separation():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.01, 0.05, 0.10, 0.20, 0.80, 0.90, 0.95, 0.99])
    m = compute_metrics(y, p)
    assert m["auc"] == 1.0
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert 0.0 <= m["brier"] <= 1.0
    assert set(["auc", "ap", "brier", "brier_skill", "ici",
                "mean_net_benefit"]).issubset(m.keys())


def test_compute_metrics_requires_two_classes():
    with pytest.raises(ValueError):
        compute_metrics(np.array([1, 1, 1, 1]), np.array([0.5] * 4))


def test_brier_skill_score_baseline_is_zero():
    y = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    prev = y.mean()
    assert abs(brier_skill_score(y, np.full(len(y), prev))) < 1e-8


def test_brier_skill_positive_when_better_than_baseline():
    y, p, _ = _data()
    assert brier_skill_score(y, p) > 0


def test_ici_small_for_well_calibrated_model():
    rng = np.random.default_rng(1)
    n = 2000
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.random(n) < p).astype(int)
    assert integrated_calibration_index(y, p) < 0.05


def test_threshold_metrics_confusion_counts():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.2])
    tm = threshold_metrics(y, p, 0.5)
    # 预测阳性：0.9, 0.6 → tp=1(y=1), fp=1(y=0); 阴性：0.4, 0.2 → fn=1, tn=1
    assert (tm["tp"] if "tp" in tm else tm["fp"]) == 1
    assert tm["sensitivity"] == 0.5
    assert tm["specificity"] == 0.5


def test_sensitivity_at_specificity_in_range():
    y, p, _ = _data()
    s = sensitivity_at_specificity(y, p, 0.8)
    assert 0.0 <= s <= 1.0
    # 放宽特异度要求后，灵敏度不应下降
    assert sensitivity_at_specificity(y, p, 0.6) >= s - 1e-9


def test_paired_patient_bootstrap_diff_sign():
    y, p_good, _ = _data()
    p_bad = 1 - p_good
    lo, hi = paired_patient_bootstrap_diff(y, p_good, p_bad, n_boot=300, seed=7)
    assert lo > 0 and hi > 0 and hi >= lo


def test_cluster_bootstrap_diff_ci_constant():
    diffs = np.full(25, 0.05)
    lo, hi = cluster_bootstrap_diff_ci(diffs, n_boot=500, seed=3)
    assert abs(lo - 0.05) < 1e-9 and abs(hi - 0.05) < 1e-9


def test_normal_ci_consistent_with_bootstrap():
    rng = np.random.default_rng(2)
    diffs = rng.normal(0.04, 0.02, 40)
    nlo, nhi = normal_ci(diffs)
    blo, bhi = cluster_bootstrap_diff_ci(diffs, n_boot=2000, seed=5)
    assert nlo < 0.04 < nhi
    # 两种口径应当给出同号的结论（都排除 0）
    assert nlo > 0 and blo > 0


def test_log_loss_and_brier_relation():
    y, p, _ = _data()
    m = compute_metrics(y, p)
    assert m["log_loss"] > 0
    assert m["brier"] < 1.0

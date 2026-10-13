import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from analysis.evaluation import calibration_slope
from analysis.inference import (_midrank, bootstrap_auc_ci,
                                calibration_intercept_slope, delong_ci,
                                delong_pairwise, delong_roc_variance,
                                delong_test, pooled_oof, pooled_performance)

rng = np.random.default_rng(2025)
n = 120
y = rng.integers(0, 2, n)
# 有信号但不完美的预测，以及一个纯随机预测
signal = np.clip(y * 0.6 + rng.normal(0, 0.25, n), 1e-4, 1 - 1e-4)
noise = rng.random(n)


# 1. midrank 与并列值
def test_midrank_ties():
    assert np.allclose(_midrank([10, 20, 20, 30]), [1, 2.5, 2.5, 4])


# 2. 单模型 AUC 与 sklearn 一致；协方差矩阵对称、对角非负
def test_delong_variance_matches_roc_auc_score():
    aucs, cov = delong_roc_variance(y, signal.reshape(1, -1))
    assert aucs.shape == (1,) and cov.shape == (1, 1)
    assert aucs[0] == pytest.approx(roc_auc_score(y, signal), abs=1e-10)
    assert np.allclose(cov, cov.T)
    assert np.all(np.diag(cov) >= 0)


def test_delong_variance_multi_model():
    aucs, cov = delong_roc_variance(y, np.vstack([signal, noise]))
    assert aucs.shape == (2,) and cov.shape == (2, 2)
    assert np.allclose(aucs, [roc_auc_score(y, signal), roc_auc_score(y, noise)],
                       atol=1e-10)
    assert np.allclose(cov, cov.T)
    assert np.all(np.diag(cov) >= 0)


# 3. 同一预测两次 → z == 0, p == 1
def test_delong_test_identical_predictions():
    auc_a, auc_b, z, p = delong_test(y, signal, signal.copy())
    assert auc_a == pytest.approx(auc_b, abs=1e-12)
    assert z == 0.0 and p == 1.0


# 4. 完美预测 vs 随机预测 → p < 0.05
def test_delong_test_perfect_vs_random():
    perfect = np.where(y == 1, 0.9, 0.1) + np.arange(n) * 1e-6
    auc_a, auc_b, z, p = delong_test(y, perfect, noise)
    assert auc_a == pytest.approx(1.0, abs=1e-10)
    assert auc_a > auc_b
    assert p < 0.05


def test_delong_ci_brackets_point_estimate():
    auc, lo, hi = delong_ci(y, signal)
    assert auc == pytest.approx(roc_auc_score(y, signal), abs=1e-10)
    assert lo < auc < hi


def test_delong_se_is_on_the_scale_of_a_probability():
    """DeLong SE 必须与 bootstrap SE 同量级（防结构分量漏掉 1/n、1/m 归一化）。"""
    _, cov = delong_roc_variance(y, signal.reshape(1, -1))
    se = float(np.sqrt(cov[0, 0]))
    boots = bootstrap_auc_ci(y, signal, n_boot=400, seed=3)[3]
    boot_se = float(np.std(boots, ddof=1))
    assert 0.0 < se < 0.5
    assert se == pytest.approx(boot_se, rel=0.25)


# 5. bootstrap 可复现且 lo <= auc <= hi
def test_bootstrap_auc_ci_reproducible():
    r1 = bootstrap_auc_ci(y, signal, n_boot=200, seed=7)
    r2 = bootstrap_auc_ci(y, signal, n_boot=200, seed=7)
    assert len(r1) == 4 and len(r2) == 4
    auc1, lo1, hi1, b1 = r1
    auc2, lo2, hi2, b2 = r2
    assert (auc1, lo1, hi1) == (auc2, lo2, hi2)
    assert np.array_equal(b1, b2)
    assert lo1 <= auc1 <= hi1
    assert auc1 == pytest.approx(roc_auc_score(y, signal), abs=1e-12)


# 6. 校准斜率与 evaluation.calibration_slope 一致
def test_calibration_intercept_slope_matches_evaluation():
    intercept, slope = calibration_intercept_slope(y, signal)
    assert slope == pytest.approx(calibration_slope(y, signal), abs=1e-8)
    assert np.isfinite(intercept)


# 7. 覆盖不全 → ValueError
def test_pooled_oof_incomplete_coverage_raises():
    recs = [(np.arange(n // 2), rng.random(n // 2))]
    with pytest.raises(ValueError, match="cover all samples"):
        pooled_oof(y, recs)


def test_pooled_oof_averages_repeated_coverage():
    idx = np.arange(n)
    recs = [(idx, np.full(n, 0.2)), (idx, np.full(n, 0.8))]
    assert np.allclose(pooled_oof(y, recs), 0.5)


# 8. 合成 oof_by_model → pooled_performance / delong_pairwise
def _synthetic_oof(n_samples=80, seed=11):
    r = np.random.default_rng(seed)
    yy = r.integers(0, 2, n_samples)
    half = n_samples // 2
    folds = [np.arange(half), np.arange(half, n_samples)]
    base = np.clip(yy * 0.5 + r.normal(0, 0.3, n_samples), 1e-3, 1 - 1e-3)
    good = np.clip(yy * 0.8 + r.normal(0, 0.2, n_samples), 1e-3, 1 - 1e-3)
    oof = {
        "Logistic_LASSO_baseline": [(f, base[f]) for f in folds],
        "SVM_rbf": [(f, good[f]) for f in folds],
    }
    return yy, oof


PERF_COLS = {"model", "n", "events", "auc", "auc_ci_boot_lo", "auc_ci_boot_hi",
             "auc_ci_delong_lo", "auc_ci_delong_hi", "auprc", "brier",
             "calib_intercept", "calib_slope", "sensitivity", "specificity"}


def test_pooled_performance_table():
    yy, oof = _synthetic_oof()
    tab = pooled_performance(yy, oof, n_boot=200, seed=11)
    assert len(tab) == 2
    assert set(tab.columns) == PERF_COLS
    assert set(tab["model"]) == set(oof)
    assert tab["auc"].between(0, 1).all()
    assert (tab["auc_ci_boot_lo"] <= tab["auc"]).all()
    assert (tab["auc"] <= tab["auc_ci_boot_hi"]).all()
    assert (tab["auc_ci_delong_lo"] <= tab["auc"]).all()
    assert (tab["auc"] <= tab["auc_ci_delong_hi"]).all()
    assert list(tab["n"]) == [len(yy)] * 2
    assert list(tab["events"]) == [int(yy.sum())] * 2
    assert list(tab["auc"]) == sorted(tab["auc"], reverse=True)


def test_delong_pairwise_holm_and_baseline_error():
    yy, oof = _synthetic_oof()
    comp = delong_pairwise(yy, oof)
    assert list(comp["model"]) == ["SVM_rbf"]
    assert {"auc", "auc_baseline", "auc_diff", "z", "p_delong", "p_holm"} <= set(comp.columns)
    assert (comp["p_holm"] >= comp["p_delong"]).all()
    assert comp["p_holm"].between(0, 1).all()
    assert np.allclose(comp["auc_diff"], comp["auc"] - comp["auc_baseline"])
    with pytest.raises(KeyError):
        delong_pairwise(yy, oof, baseline="NoSuchModel")

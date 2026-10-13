import re

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from analysis.config import RANDOM_SEED, TARGET_COL
from analysis.data_loader import column_kinds, load_data, split_xy
from analysis.preprocessing import AutoPreprocessor
from analysis.stats_tables import (
    _median_iqr,
    _smd,
    irls_logistic,
    table1_by_group,
    univariate_lr_table,
)

df = load_data()


def _synthetic(seed=RANDOM_SEED, n=400, coef=(-0.5, 1.2)):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(coef[0] + coef[1] * x)))
    y = (rng.uniform(size=n) < p).astype(int)
    return x.reshape(-1, 1), y


def test_irls_matches_sklearn_unpenalised():
    """1. IRLS 与 sklearn 近无惩罚解在合成数据上一致。"""
    X, y = _synthetic()
    fit = irls_logistic(X, y)
    sk = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000).fit(X, y)
    assert fit["coef"][0] == pytest.approx(float(sk.intercept_[0]), abs=1e-3)
    assert fit["coef"][1] == pytest.approx(float(sk.coef_[0][0]), abs=1e-3)


def test_irls_wald_self_consistent():
    """2. p 与 2*norm.sf(|z|) 自洽；置信区间包住系数。"""
    X, y = _synthetic()
    fit = irls_logistic(X, y)
    np.testing.assert_allclose(fit["p"], 2 * norm.sf(np.abs(fit["z"])), rtol=0, atol=0)
    np.testing.assert_allclose(fit["z"], fit["coef"] / fit["se"], rtol=1e-12)
    assert np.all(fit["ci_lo"] < fit["coef"])
    assert np.all(fit["coef"] < fit["ci_hi"])
    assert np.all(fit["se"] > 0)
    assert np.all((fit["p"] >= 0) & (fit["p"] <= 1))


def test_perfect_separation_raises_and_table_survives():
    """3. 完全分离 → ValueError；整表不崩溃并标注 not estimable。"""
    rng = np.random.default_rng(RANDOM_SEED)
    x = rng.normal(size=200)
    y = (x > 0).astype(int)
    with pytest.raises(ValueError):
        irls_logistic(x.reshape(-1, 1), y)

    sep = pd.DataFrame({
        "sep": x,
        "ok": rng.normal(size=200),
        TARGET_COL: y,
    })
    tab = univariate_lr_table(sep)
    assert len(tab) == 2
    row = tab.loc[tab["variable"] == "sep"].iloc[0]
    assert "not estimable" in row["note"]
    assert np.isnan(row["p_wald"]) and np.isnan(row["or_per_sd"])
    assert not bool(row["selected"])
    # 其余列仍然照常估计
    other = tab.loc[tab["variable"] == "ok"].iloc[0]
    assert np.isfinite(other["p_wald"])


def test_constant_column_noted():
    const = pd.DataFrame({"c": np.ones(50), "x": np.arange(50.0),
                          TARGET_COL: np.r_[np.zeros(25), np.ones(25)]})
    tab = univariate_lr_table(const)
    row = tab.loc[tab["variable"] == "c"].iloc[0]
    assert row["note"] == "constant column"
    assert np.isnan(row["p_wald"])


def test_univariate_table_on_real_data():
    """4. 真实数据上的单因素 Wald 表。"""
    tab = univariate_lr_table(df)
    assert len(tab) == 28
    assert list(tab.columns) == ["variable", "n", "or_per_sd", "ci_lo", "ci_hi",
                                 "p_wald", "selected", "note"]
    assert set(tab["variable"]) == set(split_xy(df)[2])
    assert int(tab["selected"].sum()) >= 1
    assert bool((tab["p_wald"].isna() | tab["p_wald"].between(0, 1)).all())
    ok = tab.dropna(subset=["or_per_sd"])
    assert not ok.empty
    assert bool((ok["ci_lo"] <= ok["or_per_sd"]).all())
    assert bool((ok["or_per_sd"] <= ok["ci_hi"]).all())
    assert bool((ok["or_per_sd"] > 0).all())
    assert int(tab["n"].nunique()) == 1 and int(tab["n"].iloc[0]) == 163
    # 按 p_wald 升序排列，NaN（若有）只能排在尾部
    p = tab["p_wald"].to_numpy(dtype=float)
    finite = p[~np.isnan(p)]
    assert np.all(np.diff(finite) >= 0)
    nan_idx = np.flatnonzero(np.isnan(p))
    assert nan_idx.size == 0 or nan_idx[0] == len(p) - nan_idx.size
    assert all(n in {"binary (OR)", "OR per 1 SD", "constant column"}
               or n.startswith("not estimable") for n in tab["note"])
    kinds = column_kinds(df)
    assert set(tab.loc[tab["note"] == "binary (OR)", "variable"]) == set(kinds["binary"])


def test_table1_on_real_data():
    """5. 真实数据上的 Table 1。"""
    t1 = table1_by_group(df)
    assert len(t1) == 28
    assert list(t1.columns) == ["variable", "type", "overall", "responder",
                                "non_responder", "p_value", "smd"]
    assert bool(t1["p_value"].between(0, 1).all())
    assert bool(t1["smd"].notna().all())
    kinds = column_kinds(df)
    assert set(t1.loc[t1["type"] == "binary", "variable"]) == set(kinds["binary"])
    assert set(t1.loc[t1["type"] == "continuous", "variable"]) == set(kinds["continuous"])
    for _, r in t1.iterrows():
        if r["type"] == "binary":
            assert "%" in r["overall"]
            assert r["overall"].startswith(f"{int(df[r['variable']].sum())}/163 (")
        else:
            assert re.fullmatch(r"-?\d+\.\d{2} \(-?\d+\.\d{2}--?\d+\.\d{2}\)",
                                r["overall"]), r["overall"]
    # 应答组 n=93、非应答组 n=70
    binrows = t1[t1["type"] == "binary"]
    assert binrows["responder"].str.contains(r"/93 \(").all()
    assert binrows["non_responder"].str.contains(r"/70 \(").all()


def test_column_kinds_matches_auto_preprocessor():
    """6. 关键一致性：column_kinds 与 AutoPreprocessor.binary_mask_ 完全相同。"""
    X, _, names = split_xy(df)
    pre = AutoPreprocessor().fit(X)
    from_pre = {names[i] for i in range(len(names)) if pre.binary_mask_[i]}
    kinds = column_kinds(df)
    assert set(kinds["binary"]) == from_pre
    assert set(kinds["continuous"]) == set(names) - from_pre
    assert len(kinds["binary"]) + len(kinds["continuous"]) == 28
    assert TARGET_COL not in kinds["binary"] and TARGET_COL not in kinds["continuous"]


def test_column_kinds_synthetic_edge_cases():
    small = pd.DataFrame({
        "bin": [0, 1, np.nan, 1],
        "cont": [0.0, 1.5, 2.0, np.nan],
        "allnan": [np.nan] * 4,
        TARGET_COL: [0, 1, 0, 1],
    })
    kinds = column_kinds(small)
    assert kinds["binary"] == ["bin", "allnan"]
    assert kinds["continuous"] == ["cont"]


def test_helpers():
    assert _median_iqr([1, 2, 3, 4]) == "2.50 (1.75-3.25)"
    assert _median_iqr([]) == "NA"
    assert _median_iqr([np.nan, np.nan]) == "NA"
    assert _smd([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    assert np.isnan(_smd([1.0], [1.0, 2.0]))
    assert _smd([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx(-3.0)
    assert _smd([1.0, 1.0], [1.0, 1.0]) == 0.0  # pooled == 0


def test_table1_degenerate_column_does_not_crash():
    """退化连续列（两组取值完全相同）→ p_value 记 NaN，而不是抛异常。"""
    rng = np.random.default_rng(RANDOM_SEED)
    n = 40
    deg = pd.DataFrame({
        "same": np.full(n, 5.0),
        "x": rng.normal(size=n),
        TARGET_COL: np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int),
    })
    t1 = table1_by_group(deg)
    assert len(t1) == 2
    row = t1.loc[t1["variable"] == "same"].iloc[0]
    assert row["type"] == "continuous"
    assert np.isnan(row["p_value"])
    assert row["overall"] == "5.00 (5.00-5.00)"
    assert np.isfinite(t1.loc[t1["variable"] == "x", "p_value"].iloc[0])

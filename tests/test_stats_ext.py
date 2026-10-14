import numpy as np
import pandas as pd
import pytest

from analysis.stats_ext import (average_ranks, friedman_test,
                                nemenyi_critical_difference, nemenyi_vs_control,
                                rank_stability)


def _blocks(n_blocks=25, gap=0.05):
    """构造块数据：model_a 恒定优于 model_b，model_c 恒定居中。"""
    rng = np.random.default_rng(0)
    base = rng.uniform(0.6, 0.8, n_blocks)
    return pd.DataFrame({
        "model_a": base + gap,
        "model_b": base - gap,
        "model_c": base,
    })


def test_average_ranks_order():
    df = _blocks()
    ranks = average_ranks(df)
    assert ranks.index[0] == "model_a"
    assert ranks.loc["model_a"] < ranks.loc["model_c"] < ranks.loc["model_b"]


def test_friedman_detects_difference():
    df = _blocks()
    res = friedman_test(df)
    assert res["n_models"] == 3
    assert res["n_blocks"] == 25
    assert res["p_value"] < 1e-6


def test_friedman_requires_at_least_three_models():
    df = _blocks()[["model_a", "model_b"]]
    with pytest.raises(ValueError):
        friedman_test(df)


def test_nemenyi_critical_difference_positive():
    cd = nemenyi_critical_difference(3, 25)
    assert cd > 0


def test_nemenyi_vs_control_flags_worse_model():
    df = _blocks()
    post, cd = nemenyi_vs_control(df, "model_a")
    worse = post.set_index("model").loc["model_b"]
    better = post.set_index("model").loc["model_c"]
    assert worse["significant"] or worse["p_holm"] < 0.05
    assert better["mean_rank"] < worse["mean_rank"]


def test_rank_stability_counts():
    rng = np.random.default_rng(1)
    seed_df = pd.DataFrame({
        "m1": [0.80, 0.79, 0.81, 0.78, 0.80],
        "m2": [0.60, 0.61, 0.59, 0.62, 0.60],
    })
    ranks = seed_df.rank(axis=1, ascending=False)
    out = rank_stability(ranks)
    best = out.set_index("model").loc["m1"]
    assert best["times_best"] == 5
    assert best["mean_rank"] < out.set_index("model").loc["m2"]["mean_rank"]

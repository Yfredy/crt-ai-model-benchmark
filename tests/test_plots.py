import os

import numpy as np

from analysis.plots import plot_roc, plot_calibration, plot_dca

rng = np.random.default_rng(1)
y = rng.integers(0, 2, 60)
oof = {
    "A": [(np.arange(60), np.clip(y * 0.8 + rng.normal(0, 0.2, 60), 0, 1))],
    "B": [(np.arange(60), rng.random(60))],
}

def test_plots_write_files(tmp_path):
    p1 = plot_roc(y, oof, tmp_path / "roc.png")
    p2 = plot_calibration(y, oof, tmp_path / "cal.png")
    p3 = plot_dca(y, oof, tmp_path / "dca.png")
    for p in (p1, p2, p3):
        assert os.path.exists(p) and os.path.getsize(p) > 0

def test_multiple_repeats_pooled(tmp_path):
    # 多次重复的 OOF 记录：同一单元格取均值而非覆盖
    y2 = rng.integers(0, 2, 40)
    idx = np.arange(40)
    p_rep1 = np.full(40, 0.2)
    p_rep2 = np.full(40, 0.8)
    oof2 = {"M": [(idx, p_rep1), (idx, p_rep2)]}
    from analysis.plots import _pooled
    pooled = _pooled(y2, oof2["M"])
    assert np.allclose(pooled, 0.5)  # (0.2+0.8)/2
    p = plot_roc(y2, oof2, tmp_path / "roc2.png")
    assert os.path.exists(p) and os.path.getsize(p) > 0

def test_incomplete_coverage_raises(tmp_path):
    y3 = np.zeros(10, dtype=int)
    oof3 = {"M": [(np.arange(5), np.random.random(5))]}  # 只覆盖一半
    import pytest
    with pytest.raises(AssertionError):
        plot_roc(y3, oof3, tmp_path / "bad.png")

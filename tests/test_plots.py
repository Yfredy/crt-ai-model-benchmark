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
    # 多次重复的 OOF 记录也能正常绘图（后写覆盖同位置单元格）
    y2 = rng.integers(0, 2, 40)
    idx1 = np.arange(0, 20); idx2 = np.arange(20, 40)
    oof2 = {
        "M": [(idx1, rng.random(20)), (idx2, rng.random(20))],
    }
    p = plot_roc(y2, oof2, tmp_path / "roc2.png")
    assert os.path.exists(p) and os.path.getsize(p) > 0

def test_incomplete_coverage_raises(tmp_path):
    y3 = np.zeros(10, dtype=int)
    oof3 = {"M": [(np.arange(5), np.random.random(5))]}  # 只覆盖一半
    import pytest
    with pytest.raises(AssertionError):
        plot_roc(y3, oof3, tmp_path / "bad.png")

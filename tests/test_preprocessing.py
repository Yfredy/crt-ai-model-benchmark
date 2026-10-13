import numpy as np

from analysis.preprocessing import AutoPreprocessor

X = np.array([
    [1.0, 0.0, 10.0],
    [0.0, 1.0, 20.0],
    [1.0, 0.0, np.nan],   # 连续列缺失
    [0.0, np.nan, 40.0],  # 二值列缺失
])

def test_output_shape_and_finite():
    p = AutoPreprocessor().fit(X)
    Xt = p.transform(X)
    assert Xt.shape == X.shape
    assert np.all(np.isfinite(Xt)), "no NaN may remain after imputation"

def test_binary_columns_stay_binary_after_imputation():
    p = AutoPreprocessor().fit(X)
    Xt = p.transform(X)
    # 二值列填补后取值仍应为 {0,1}
    for j in (0, 1):
        assert set(np.unique(Xt[:, j])) <= {0.0, 1.0}

def test_binary_mask_detection():
    p = AutoPreprocessor().fit(X)
    assert list(p.binary_mask_) == [True, True, False]

def test_continuous_columns_standardized():
    p = AutoPreprocessor().fit(X)
    Xt = p.transform(X)
    cont = Xt[:, 2]
    assert abs(cont.mean()) < 1.5 and abs(cont.std()) < 2.5

def test_feature_names_out():
    p = AutoPreprocessor().fit(X)
    names = p.feature_names_out(["bin_a", "bin_b", "cont_c"])
    assert names == ["bin_a", "bin_b", "cont_c"]  # 逐列变换，顺序不变

def test_real_data_passthrough():
    # 真实数据零缺失：输出应有限且形状一致
    from analysis.data_loader import load_data, split_xy
    X_real, _, _ = split_xy(load_data())
    p = AutoPreprocessor().fit(X_real)
    Xt = p.transform(X_real)
    assert Xt.shape == X_real.shape
    assert np.all(np.isfinite(Xt))
    assert list(p.binary_mask_) == [
        True,   # Gender_male
        False,  # Age
        True,   # DCM
        True,   # Hypertension
        True,   # Diabetes
        False,  # QRSd_ms
        True,   # NS_VT
        False,  # NT_proBNP ordinal
        False,  # EDV
        False,  # ESV
        False,  # LVEDD
        False,  # LVESD
        False,  # LVEF_SPECT
        False,  # LVEF_Echo
        False,  # Rest_Scar_Burden
        False,  # systolic_PSD
        False,  # systolic_PBW
        False,  # diastolic_PSD
        False,  # diastolic_PBW
        True,   # systolic_match
        True,   # diastolic_match
        True,   # lead_in_scar_burden (实际为 0/1 编码：导线是否位于瘢痕区)
        False,  # NYHA
        True,   # ARB
        True,   # ACEI
        True,   # βblocker
        True,   # Diuretic
        True,   # Spironolactone
    ]

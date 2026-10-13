import numpy as np
import pandas as pd
import pytest

from analysis.data_loader import load_data, split_xy, data_quality_notes
from analysis.config import TARGET_COL

df = load_data()

def test_shape():
    assert df.shape == (163, 29)

def test_target_binary():
    assert set(df[TARGET_COL].unique()) == {0, 1}

def test_target_counts_match_paper():
    # 论文：93 应答者 (57.1%)，其余 70 例不应答
    assert int(df[TARGET_COL].sum()) == 93
    assert int((df[TARGET_COL] == 0).sum()) == 70

def test_all_predictors_numeric():
    X, y, names = split_xy(df)
    assert X.dtype == np.float64
    assert X.shape == (163, 28)
    assert len(names) == 28
    assert y.shape == (163,)

def test_special_column_names_present():
    assert "βblocker" in df.columns
    assert "NT_proBNP(<1000=1; 1000~2499=2; 2500~4999=3; 5000~1000=4; >10000=5)" in df.columns

def test_load_rejects_wrong_file(tmp_path):
    bad = tmp_path / "bad.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(bad, index=False)
    with pytest.raises(ValueError):
        load_data(bad)

def test_data_quality_notes_flags_age_outlier():
    notes = data_quality_notes(df)
    assert any("Age" in n and "165" in n for n in notes)

def test_data_quality_notes_clean_data():
    clean = pd.DataFrame({"response": [0, 1], "Age": [50, 60]})
    assert data_quality_notes(clean) == []

from analysis.config import DATA_PATH, TARGET_COL, RANDOM_SEED

def test_data_file_exists():
    assert DATA_PATH.exists(), f"data.xlsx not found at {DATA_PATH}"

def test_constants():
    assert TARGET_COL == "response"
    assert isinstance(RANDOM_SEED, int)

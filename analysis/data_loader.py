import warnings

import numpy as np
import pandas as pd

from analysis.config import DATA_PATH, TARGET_COL

EXPECTED_ROWS = 163
EXPECTED_COLS = 29

# 已知生理上不可能的范围（仅警告，不阻断：保留与已发表队列一致的原值）
RANGE_CHECKS = {"Age": (18, 120)}


def validate(df: pd.DataFrame) -> None:
    if df.shape != (EXPECTED_ROWS, EXPECTED_COLS):
        raise ValueError(
            f"Expected data shape ({EXPECTED_ROWS}, {EXPECTED_COLS}), got {df.shape}"
        )
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found")
    if not set(df[TARGET_COL].unique()) <= {0, 1}:
        raise ValueError("Target must be binary 0/1")
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric columns found: {non_numeric}")


def data_quality_notes(df: pd.DataFrame) -> list[str]:
    """返回仅用于报告的数据质量备注（不修改数据、不阻断加载）。"""
    notes = []
    for col, (lo, hi) in RANGE_CHECKS.items():
        if col in df.columns:
            bad = df[(df[col] < lo) | (df[col] > hi)][col]
            for idx, val in bad.items():
                note = (f"{col}={val} at row {idx + 2} outside plausible range "
                        f"[{lo}, {hi}] (suspected entry error; kept as-is to match "
                        f"the published cohort)")
                notes.append(note)
                warnings.warn(note)
    n_missing = int(df.isna().sum().sum())
    if n_missing:
        notes.append(f"missing cells: {n_missing} (median-imputed inside CV folds)")
    return notes


def load_data(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_excel(path)
    validate(df)
    return df


def split_xy(df: pd.DataFrame):
    feature_names = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_names].to_numpy(dtype=np.float64)
    y = df[TARGET_COL].to_numpy(dtype=np.int64)
    return X, y, feature_names


def column_kinds(df) -> dict:
    """返回 {'binary': [...], 'continuous': [...]}；判定标准与 AutoPreprocessor 一致：
    取值 ⊆ {0,1}（忽略 NaN）视为二值。不含 TARGET_COL。"""
    feats = [c for c in df.columns if c != TARGET_COL]
    binary, cont = [], []
    for c in feats:
        v = df[c].dropna().unique()
        (binary if set(v) <= {0, 1} else cont).append(c)
    return {"binary": binary, "continuous": cont}

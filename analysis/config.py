from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

TARGET_COL = "response"
RANDOM_SEED = 2025
N_FOLDS = 5          # 外层 CV 折数
N_REPEATS = 5        # 外层 CV 重复次数
N_INNER_FOLDS = 3    # 内层调参折数

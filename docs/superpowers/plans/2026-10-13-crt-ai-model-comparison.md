# CRT 应答预测：传统基线 vs 多种 AI 模型基准对比 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 163 例 CRT 患者数据上，系统对比"传统 LASSO/logistic 基线"与多种 AI 模型（随机森林、SVM、XGBoost、LightGBM、正则化 MLP、TabNet）的判别能力，产出结果表、图形与 SHAP 可解释性分析，支撑回应编辑意见的新 article 与综述。

**Architecture:** 纯 Python 批处理分析管道（无 web 服务）。所有模型封装为 sklearn 兼容 Pipeline（预处理内嵌于 Pipeline 中杜绝数据泄漏），用嵌套重复分层交叉验证（外层 5 折 × 5 次重复做泛化评估，内层 3 折做超参调优）得到折外（out-of-fold）预测，再汇总 AUC/Brier/校准斜率/Youden 敏感性特异性，绘制 ROC/校准/DCA 曲线，对最优模型做 SHAP 分析，最终生成 `outputs/report.md`。

**Tech Stack:** Python 3.10+；pandas、numpy、scikit-learn、scipy、xgboost、lightgbm、pytorch-tabnet（torch CPU 版）、shap、matplotlib、pytest。

**Spec:** `背景说明.md`（项目背景、编辑意见、数据集说明）。计划据此argues：编辑要求"增加 AI 算法、深度学习、大数据挖掘内容"；聊天记录要求"测试哪种 AI 模型与当前数据最适配，response 是因变量"。

## 执行环境（重要：使用本机已有环境，勿创建/安装新环境）

- **解释器**：`C:\Users\yaoyue\miniconda3\python.exe`（Python 3.12.4，conda base）。
- **已具备全部依赖**：numpy 2.4.6 / pandas 2.3.3 / scipy 1.18.0 / scikit-learn 1.9.0 / xgboost 3.4.1 / lightgbm 4.7.0 / shap 0.53.0 / pytorch-tabnet / torch 2.6.0+cu124 / matplotlib 3.11.1 / pytest 9.1.1 / tabulate 0.10.0 / openpyxl 3.1.5。
- **GPU**：NVIDIA RTX 4060 Laptop（CUDA 12.4 可用）。仍按 Global Constraints 以 CPU 运行（n=163 GPU 无收益）；如切 GPU，改 `analysis/models.py` 中 TabNet `device_name`。
- **规则**：本计划文本中凡出现的 `.venv/Scripts/python` 一律替换为上面的解释器；**禁止** `pip install` 或新建虚拟环境（用户明确要求）。

## Global Constraints

- 数据文件 `data.xlsx` **只读**，任何任务不得修改/移动/重命名它。
- 随机种子统一为 `RANDOM_SEED = 2025`，所有模型与 CV 分割必须带种子，保证可复现。
- 全程 CPU 计算（`device_name="cpu"`），不要求 GPU。
- 预处理（标准化、缺失填补、单变量筛选、超参搜索）**必须发生在 CV 训练折内部**，严禁在全数据上先做再切分（数据泄漏）。
- 因变量为 `response`（1=应答 93 例，0=不应答 71 例）；候选自变量为其余 28 列，全部为数值型。
- 列名必须使用精确原名，特别是含特殊字符的两列：
  - `NT_proBNP(<1000=1; 1000~2499=2; 2500~4999=3; 5000~1000=4; >10000=5)`
  - `βblocker`
- 所有输出写入 `outputs/`（表格 `outputs/tables/`，图 `outputs/figures/`），报告为 `outputs/report.md`。
- 统计比较：各模型 vs 基线的配对 Wilcoxon 检验 + Holm 校正，比较单位为"每个外层折的 AUC"（同一折配对）。
- 完整运行预计 1–3 小时（CPU）；提供 `--quick` 模式（1 次重复、缩小网格）用于冒烟验证。
- 每个任务结束必须 `git commit`。

---

### Task 1: 环境搭建与项目骨架

**Files:**
- Create: `requirements.txt`
- Create: `analysis/__init__.py`
- Create: `analysis/config.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`
- Create: `.gitignore`

**Interfaces:**
- Consumes: 无（起始任务）
- Produces: `analysis.config` 模块，导出 `PROJECT_ROOT, DATA_PATH, OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, RANDOM_SEED, TARGET_COL, N_FOLDS, N_REPEATS, N_INNER_FOLDS`；`pytest` 可运行。

- [ ] **Step 1: 初始化 git 仓库与 .gitignore**

```bash
cd "C:\Users\yaoyue\Desktop\20261013-article"
git init
```

创建 `.gitignore`：

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
outputs/
```

- [ ] **Step 2: 创建虚拟环境并安装依赖**

创建 `requirements.txt`：

```
pandas>=2.0
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
openpyxl>=3.1
xgboost>=2.0
lightgbm>=4.0
matplotlib>=3.8
shap>=0.44
pytorch-tabnet>=4.1
pytest>=7.4
```

torch 单独用 CPU 源安装（避免拉 CUDA 大包）：

```bash
python -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

若本机无 Python 3.10+，先从 python.org 安装。验证：

```bash
.venv/Scripts/python -c "import sklearn, xgboost, lightgbm, shap, pytorch_tabnet, torch; print('ok')"
```

Expected: 输出 `ok`

- [ ] **Step 3: 创建 config 模块**

`analysis/__init__.py` 为空文件。创建 `analysis/config.py`：

```python
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
```

- [ ] **Step 4: 写 config 冒烟测试**

创建 `tests/__init__.py`（空）与 `tests/test_config.py`：

```python
from analysis.config import DATA_PATH, TARGET_COL, RANDOM_SEED

def test_data_file_exists():
    assert DATA_PATH.exists(), f"data.xlsx not found at {DATA_PATH}"

def test_constants():
    assert TARGET_COL == "response"
    assert isinstance(RANDOM_SEED, int)
```

- [ ] **Step 5: 运行测试验证通过**

```bash
.venv/Scripts/python -m pytest tests/test_config.py -v
```

Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add .gitignore requirements.txt analysis/ tests/
git commit -m "chore: project scaffold with venv, config module and pytest"
```

---

### Task 2: 数据加载与校验

**Files:**
- Create: `analysis/data_loader.py`
- Create: `tests/test_data_loader.py`

**Interfaces:**
- Consumes: `analysis.config.DATA_PATH, TARGET_COL`
- Produces:
  - `load_data(path=DATA_PATH) -> pd.DataFrame`（163 行 × 29 列，校验失败抛 `ValueError`）
  - `split_xy(df) -> tuple[np.ndarray, np.ndarray, list[str]]`（返回 X 浮点矩阵、y 0/1 数组、特征名列表）

- [ ] **Step 1: 写失败测试**

创建 `tests/test_data_loader.py`：

```python
import numpy as np
import pandas as pd
import pytest

from analysis.data_loader import load_data, split_xy
from analysis.config import TARGET_COL

df = load_data()

def test_shape():
    assert df.shape == (163, 29)

def test_target_binary():
    assert set(df[TARGET_COL].unique()) == {0, 1}

def test_target_counts_match_paper():
    # 论文：93 应答者 (57.1%)
    assert int(df[TARGET_COL].sum()) == 93

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
```

- [ ] **Step 2: 运行测试确认失败**

```bash
.venv/Scripts/python -m pytest tests/test_data_loader.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'analysis.data_loader'`

- [ ] **Step 3: 实现 data_loader**

创建 `analysis/data_loader.py`：

```python
import numpy as np
import pandas as pd

from analysis.config import DATA_PATH, TARGET_COL

EXPECTED_ROWS = 163
EXPECTED_COLS = 29


def load_data(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_excel(path)
    validate(df)
    return df


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


def split_xy(df: pd.DataFrame):
    feature_names = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_names].to_numpy(dtype=np.float64)
    y = df[TARGET_COL].to_numpy(dtype=np.int64)
    return X, y, feature_names
```

- [ ] **Step 4: 运行测试确认通过**

```bash
.venv/Scripts/python -m pytest tests/test_data_loader.py -v
```

Expected: 7 passed（若 `test_shape` 因实际列数不符而失败，以真实数据为准修正 `EXPECTED_COLS` 并在 commit message 里说明）

- [ ] **Step 5: Commit**

```bash
git add analysis/data_loader.py tests/test_data_loader.py
git commit -m "feat: data loader with validation against published cohort"
```

---

### Task 3: 自动预处理器（无泄漏的标准化 + 缺失填补）

**Files:**
- Create: `analysis/preprocessing.py`
- Create: `tests/test_preprocessing.py`

**Interfaces:**
- Consumes: numpy 数组
- Produces: `AutoPreprocessor(BaseEstimator, TransformerMixin)` — `fit(X, y)` / `transform(X) -> np.ndarray`，形状不变；自动识别二值列（取值 ⊆ {0,1}）做中位数填补后 passthrough，其余列做中位数填补 + StandardScaler；属性 `binary_mask_`、`feature_names_out(names) -> list[str]` 供 SHAP 使用。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_preprocessing.py`：

```python
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

def test_binary_columns_pass_through_values():
    p = AutoPreprocessor().fit(X)
    Xt = p.transform(X)
    # 二值列填补后仍应为 0/1
    assert set(p.transform(X[:, :2].reshape(4, 2) * 0 + Xt[:, 1][~np.isnan(X[:, 1])])) <= {0.0, 1.0}

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
```

- [ ] **Step 2: 运行测试确认失败**

```bash
.venv/Scripts/python -m pytest tests/test_preprocessing.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现 AutoPreprocessor**

创建 `analysis/preprocessing.py`：

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class AutoPreprocessor(BaseEstimator, TransformerMixin):
    """逐列预处理：二值列(取值⊆{0,1})中位数填补后原样保留；其余列中位数填补+标准化。

    逐列、无标签泄漏（fit 只看训练折），输出形状与输入一致，列顺序不变。
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.binary_mask_ = np.all((X == 0) | (X == 1) | np.isnan(X), axis=0)
        self.imputer_ = SimpleImputer(strategy="median").fit(X)
        Xi = self.imputer_.transform(X)
        self.scaler_ = StandardScaler().fit(Xi[:, ~self.binary_mask_])
        return self

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != len(self.binary_mask_):
            raise ValueError(f"Expected {len(self.binary_mask_)} columns, got {X.shape[1]}")
        Xi = self.imputer_.transform(X)
        out = Xi.copy()
        out[:, ~self.binary_mask_] = self.scaler_.transform(Xi[:, ~self.binary_mask_])
        return out

    def feature_names_out(self, names):
        return list(names)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
.venv/Scripts/python -m pytest tests/test_preprocessing.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add analysis/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: leakage-free per-column auto preprocessor"
```

---

### Task 4: 单变量筛选 + LASSO 基线（复现论文方法）

**Files:**
- Create: `analysis/baseline.py`
- Create: `tests/test_baseline.py`

**Interfaces:**
- Consumes: numpy 数组
- Produces:
  - `UnivariateLRFilter(BaseEstimator, TransformerMixin)` — `fit(X, y)` 对每个特征拟合单变量 logistic 并做似然比检验，保留 P<`p_threshold`(默认 0.05) 的特征列索引；若无特征通过则保留全部（防退化）。属性 `kept_idx_`。
  - `make_baseline_pipeline(seed) -> sklearn.Pipeline`：`AutoPreprocessor → UnivariateLRFilter → LogisticRegression(penalty="l1", solver="liblinear", C=1.0)`，即论文的"单变量筛选 + LASSO"。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_baseline.py`：

```python
import numpy as np
from sklearn.pipeline import Pipeline

from analysis.baseline import UnivariateLRFilter, make_baseline_pipeline

rng = np.random.default_rng(0)
n = 120
y = rng.integers(0, 2, n)
X = np.column_stack([
    y * 2.0 + rng.normal(0, 0.3, n),   # 强信号
    rng.normal(0, 1, n),               # 噪声
    rng.normal(0, 1, n),               # 噪声
])

def test_filter_keeps_signal_drops_noise():
    f = UnivariateLRFilter(p_threshold=0.05).fit(X, y)
    assert 0 in f.kept_idx_
    assert 1 not in f.kept_idx_
    assert 2 not in f.kept_idx_

def test_filter_fallback_when_nothing_significant():
    X_noise = rng.normal(0, 1, (60, 3))
    y_noise = rng.integers(0, 2, 60)
    f = UnivariateLRFilter().fit(X_noise, y_noise)
    assert f.kept_idx_ == [0, 1, 2]  # 全保留，不允许输出 0 列

def test_transform_shape():
    f = UnivariateLRFilter().fit(X, y)
    Xt = f.transform(X)
    assert Xt.shape[0] == n and Xt.shape[1] == len(f.kept_idx_)

def test_baseline_pipeline_is_pipeline_and_fits():
    pipe = make_baseline_pipeline(seed=2025)
    assert isinstance(pipe, Pipeline)
    pipe.fit(X, y)  # 不抛异常
    proba = pipe.predict_proba(X)
    assert proba.shape == (n, 2)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
.venv/Scripts/python -m pytest tests/test_baseline.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现 baseline**

创建 `analysis/baseline.py`：

```python
import numpy as np
from scipy.stats import chi2
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from analysis.preprocessing import AutoPreprocessor


class UnivariateLRFilter(BaseEstimator, TransformerMixin):
    """对每列拟合单变量 logistic，似然比检验 P<p_threshold 则保留（论文 Methods 原方法）。"""

    def __init__(self, p_threshold=0.05):
        self.p_threshold = p_threshold

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        p1 = np.clip(y.mean(), 1e-9, 1 - 1e-9)
        ll_null = np.sum(y * np.log(p1) + (1 - y) * np.log(1 - p1))
        kept = []
        for j in range(X.shape[1]):
            xj = X[:, [j]]
            if np.nanstd(xj) == 0:
                continue
            xj = np.where(np.isnan(xj), np.nanmedian(xj), xj)
            lr = LogisticRegression(max_iter=1000).fit(xj, y)
            p_hat = np.clip(lr.predict_proba(xj)[:, 1], 1e-9, 1 - 1e-9)
            ll = np.sum(y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat))
            p_value = chi2.sf(max(2 * (ll - ll_null), 0.0), 1)
            if p_value < self.p_threshold:
                kept.append(j)
        if not kept:  # 无显著特征时退化为全部保留，保证下游可用
            kept = list(range(X.shape[1]))
        self.kept_idx_ = kept
        return self

    def transform(self, X):
        return np.asarray(X, dtype=np.float64)[:, self.kept_idx_]


def make_baseline_pipeline(seed=2025) -> Pipeline:
    return Pipeline([
        ("pre", AutoPreprocessor()),
        ("univar", UnivariateLRFilter(p_threshold=0.05)),
        ("clf", LogisticRegression(
            penalty="l1", solver="liblinear", C=1.0,
            max_iter=5000, random_state=seed,
        )),
    ])
```

- [ ] **Step 4: 运行测试确认通过**

```bash
.venv/Scripts/python -m pytest tests/test_baseline.py -v
```

Expected: 4 passed

- [ ] **Step 5: 在真实数据上冒烟（判别力应接近论文 AUC 0.845 的量级）**

```bash
.venv/Scripts/python -c "
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from analysis.data_loader import load_data, split_xy
from analysis.baseline import make_baseline_pipeline
df = load_data(); X, y, _ = split_xy(df)
scores = cross_val_score(make_baseline_pipeline(2025), X, y,
                         cv=StratifiedKFold(5, shuffle=True, random_state=2025), scoring='roc_auc')
print('AUC per fold:', np.round(scores, 3), 'mean:', scores.mean().round(3))
"
```

Expected: 平均 AUC 约 0.75–0.85（论文 0.845 为全数据拟合 + bootstrap，CV 下略低属正常；若 <0.65 需检查列名/预处理）

- [ ] **Step 6: Commit**

```bash
git add analysis/baseline.py tests/test_baseline.py
git commit -m "feat: univariate-filter + LASSO baseline replicating published method"
```

---

### Task 5: AI 模型库（含深度学习 MLP 与 TabNet）

**Files:**
- Create: `analysis/models.py`
- Create: `tests/test_models.py`

**Interfaces:**
- Consumes: `AutoPreprocessor`、`make_baseline_pipeline`
- Produces:
  - `TabNetWrapper(BaseEstimator, ClassifierMixin)` — sklearn 风格 `fit(X, y)` / `predict_proba(X) -> (n,2)`，内部自动留 20% 做早停验证，CPU 运行。
  - `get_models(seed=2025, light=False) -> dict[str, sklearn estimator]`，键固定为：
    `"Logistic_LASSO_baseline", "Logistic_L2", "RF", "SVM_rbf", "XGBoost", "LightGBM", "MLP", "TabNet"`
  - `get_grids(seed=2025, light=False) -> dict[str, list[dict]]`（与 `get_models` 键对应；基线两个模型无网格；`light=True` 返回 `{}` 供测试跳过调参）

- [ ] **Step 1: 写失败测试**

创建 `tests/test_models.py`：

```python
import numpy as np
import pytest

from analysis.models import get_models, get_grids

EXPECTED_KEYS = {
    "Logistic_LASSO_baseline", "Logistic_L2", "RF", "SVM_rbf",
    "XGBoost", "LightGBM", "MLP", "TabNet",
}

rng = np.random.default_rng(0)
X = rng.normal(0, 1, (80, 6))
y = (X[:, 0] - X[:, 1] + rng.normal(0, 0.5, 80) > 0).astype(int)

@pytest.fixture(scope="module")
def fitted():
    models = get_models(seed=2025, light=True)
    out = {}
    for name, m in models.items():
        m.fit(X, y)
        out[name] = m
    return out

def test_model_keys(fitted):
    assert set(get_models().keys()) == EXPECTED_KEYS

def test_all_models_predict_proba(fitted):
    for name, m in fitted.items():
        proba = m.predict_proba(X)
        assert proba.shape == (80, 2), name
        assert np.all((proba >= 0) & (proba <= 1)), name
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5), name

def test_grids_match_models():
    grids = get_grids(light=True)
    assert grids == {}
    grids_full = get_grids(seed=1)
    assert set(grids_full.keys()) <= EXPECTED_KEYS
    assert "Logistic_LASSO_baseline" not in grids_full

def test_determinism(fitted):
    m1 = get_models(seed=2025, light=True)["RF"]
    m2 = get_models(seed=2025, light=True)["RF"]
    p1 = m1.fit(X, y).predict_proba(X)
    p2 = m2.fit(X, y).predict_proba(X)
    assert np.allclose(p1, p2)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
.venv/Scripts/python -m pytest tests/test_models.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现模型库**

创建 `analysis/models.py`：

```python
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from analysis.config import RANDOM_SEED
from analysis.preprocessing import AutoPreprocessor
from analysis.baseline import make_baseline_pipeline


class TabNetWrapper(BaseEstimator, ClassifierMixin):
    """TabNetClassifier 的 sklearn 兼容封装：CPU、固定种子、内部留出集早停。"""

    def __init__(self, n_d=8, n_a=8, n_steps=3, gamma=1.5, lambda_sparse=1e-3,
                 max_epochs=100, patience=15, batch_size=32, seed=RANDOM_SEED):
        self.n_d = n_d; self.n_a = n_a; self.n_steps = n_steps
        self.gamma = gamma; self.lambda_sparse = lambda_sparse
        self.max_epochs = max_epochs; self.patience = patience
        self.batch_size = batch_size; self.seed = seed

    def fit(self, X, y):
        import torch
        from pytorch_tabnet.tab_model import TabNetClassifier
        from sklearn.model_selection import train_test_split

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.seed)
        self.model_ = TabNetClassifier(
            n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps,
            gamma=self.gamma, lambda_sparse=self.lambda_sparse,
            seed=self.seed, verbose=0, device_name="cpu",
        )
        self.model_.fit(
            X_tr, y_tr, eval_set=[(X_val, y_val)],
            max_epochs=self.max_epochs, patience=self.patience,
            batch_size=min(self.batch_size, len(X_tr)),
            virtual_batch_size=min(16, max(2, len(X_tr) // 2)),
        )
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        proba = np.asarray(self.model_.predict_proba(np.asarray(X, dtype=np.float64)))
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba
        p = proba.ravel()
        return np.column_stack([1 - p, p])


def _pipe(clf) -> Pipeline:
    return Pipeline([("pre", AutoPreprocessor()), ("clf", clf)])


def get_models(seed=RANDOM_SEED, light=False) -> dict:
    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16), alpha=0.5, activation="relu",
        max_iter=50 if light else 3000, early_stopping=not light,
        validation_fraction=0.15, random_state=seed,
    )
    tabnet = TabNetWrapper(
        seed=seed,
        max_epochs=5 if light else 200,
        patience=2 if light else 25,
    )
    rf = RandomForestClassifier(
        n_estimators=30 if light else 500, random_state=seed, n_jobs=1)
    xgb = XGBClassifier(
        n_estimators=10 if light else 200, max_depth=2, learning_rate=0.1,
        subsample=0.8, reg_lambda=1.0, eval_metric="logloss",
        random_state=seed, n_jobs=1, verbosity=0)
    lgbm = LGBMClassifier(
        n_estimators=10 if light else 200, num_leaves=7,
        min_child_samples=10, random_state=seed, n_jobs=1, verbose=-1)
    return {
        "Logistic_LASSO_baseline": make_baseline_pipeline(seed),
        "Logistic_L2": _pipe(LogisticRegression(
            penalty="l2", solver="lbfgs", C=1.0, max_iter=5000, random_state=seed)),
        "RF": _pipe(rf),
        "SVM_rbf": _pipe(SVC(probability=True, random_state=seed)),
        "XGBoost": _pipe(xgb),
        "LightGBM": _pipe(lgbm),
        "MLP": _pipe(mlp),
        "TabNet": _pipe(tabnet),
    }


def get_grids(seed=RANDOM_SEED, light=False) -> dict:
    if light:
        return {}
    return {
        "RF": [
            {"clf__n_estimators": [300, 500], "clf__max_features": ["sqrt", "log2"]},
        ],
        "SVM_rbf": [
            {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", 0.01]},
        ],
        "XGBoost": [
            {"clf__max_depth": [2, 3], "clf__n_estimators": [100, 200]},
        ],
        "LightGBM": [
            {"clf__num_leaves": [7, 15], "clf__min_child_samples": [10, 20]},
        ],
        "MLP": [
            {"clf__alpha": [0.1, 1.0],
             "clf__hidden_layer_sizes": [(32, 16), (16, 8)]},
        ],
        "TabNet": [
            {"clf__n_steps": [3, 4], "clf__n_d": [8, 16]},
        ],
    }
```

注意：`MLPClassifier` 在网格里调 `alpha` 与 `hidden_layer_sizes` 时 `early_stopping=True` 保持默认即可（构造时已设）。

- [ ] **Step 4: 运行测试确认通过**

```bash
.venv/Scripts/python -m pytest tests/test_models.py -v
```

Expected: 4 passed（TabNet 首次运行需编译 torch 图，约 1–2 分钟属正常）

- [ ] **Step 5: Commit**

```bash
git add analysis/models.py tests/test_models.py
git commit -m "feat: model zoo with baselines, boosting, SVM, regularized MLP and TabNet"
```

---

### Task 6: DCA 决策曲线分析模块

**Files:**
- Create: `analysis/dca.py`
- Create: `tests/test_dca.py`

**Interfaces:**
- Consumes: numpy 数组
- Produces:
  - `net_benefit(y_true, y_prob, thresholds: np.ndarray) -> np.ndarray`（模型曲线，阈值外返回 NaN）
  - `net_benefit_treat_all(y_true, thresholds) -> np.ndarray`（treat-all 曲线）
  - `net_benefit_treat_none(thresholds) -> np.ndarray`（恒为 0）

- [ ] **Step 1: 写失败测试**

创建 `tests/test_dca.py`：

```python
import numpy as np

from analysis.dca import net_benefit, net_benefit_treat_all, net_benefit_treat_none

y = np.array([1, 1, 0, 0])
p = np.array([0.9, 0.8, 0.2, 0.1])
thr = np.array([0.5])

def test_perfect_case_threshold_05():
    # tp=2, fp=0, n=4 -> nb = 2/4 = 0.5
    assert np.isclose(net_benefit(y, p, thr)[0], 0.5)

def test_treat_all_formula():
    # treat-all: prevalence - (1-prevalence)*pt/(1-pt)
    assert np.isclose(net_benefit_treat_all(y, np.array([0.5]))[0], 0.5)
    # pt=0.8: 0.5 - 0.5*0.8/0.2 = -1.5
    assert np.isclose(net_benefit_treat_all(y, np.array([0.8]))[0], -1.5)

def test_treat_none_zero():
    assert np.allclose(net_benefit_treat_none(np.array([0.1, 0.9])), 0.0)

def test_false_positive_penalty():
    # y=[1,0], p=[0.9,0.9], pt=0.5: tp=1, fp=1, n=2 -> 0.5 - 0.5 = 0
    yy = np.array([1, 0]); pp = np.array([0.9, 0.9])
    assert np.isclose(net_benefit(yy, pp, np.array([0.5]))[0], 0.0)

def test_nan_outside_valid_range():
    thr = np.array([0.0, 0.5, 1.0])
    out = net_benefit(y, p, thr)
    assert np.isnan(out[0]) and np.isnan(out[2]) and not np.isnan(out[1])
```

- [ ] **Step 2: 运行测试确认失败**

```bash
.venv/Scripts/python -m pytest tests/test_dca.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现 DCA**

创建 `analysis/dca.py`：

```python
import numpy as np


def _count(y_true, y_prob, pt):
    pred_pos = y_prob >= pt
    tp = np.sum((pred_pos) & (y_true == 1))
    fp = np.sum((pred_pos) & (y_true == 0))
    return tp, fp


def net_benefit(y_true, y_prob, thresholds) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    out = np.full(len(thresholds), np.nan)
    for i, pt in enumerate(thresholds):
        if pt <= 0.0 or pt >= 1.0:
            continue  # pt/(1-pt) 在 0/1 处无定义
        tp, fp = _count(y_true, y_prob, pt)
        out[i] = tp / len(y_true) - fp / len(y_true) * (pt / (1 - pt))
    return out


def net_benefit_treat_all(y_true, thresholds) -> np.ndarray:
    prevalence = np.mean(np.asarray(y_true) == 1)
    out = np.full(len(thresholds), np.nan)
    for i, pt in enumerate(thresholds):
        if pt <= 0.0 or pt >= 1.0:
            continue
        out[i] = prevalence - (1 - prevalence) * (pt / (1 - pt))
    return out


def net_benefit_treat_none(thresholds) -> np.ndarray:
    return np.zeros(len(thresholds))
```

- [ ] **Step 4: 运行测试确认通过**

```bash
.venv/Scripts/python -m pytest tests/test_dca.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add analysis/dca.py tests/test_dca.py
git commit -m "feat: decision curve analysis (net benefit) module"
```

---

### Task 7: 嵌套重复分层交叉验证评估引擎

**Files:**
- Create: `analysis/evaluation.py`
- Create: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `get_models, get_grids`
- Produces:
  - `calibration_slope(y_true, y_prob) -> float`（y 对 logit(p) 回归的系数）
  - `youden_sens_spec(y_true, y_prob) -> tuple[float, float]`
  - `run_comparison(X, y, models, grids=None, n_folds=5, n_repeats=5, inner_folds=3, seed=2025) -> tuple[pd.DataFrame, dict]`
    - 返回 `fold_df`：列 `["model","repeat","fold","n_test","auc","brier","sensitivity","specificity","calibration_slope"]`，每行 = 模型 × repeat × fold
    - 返回 `oof_records: dict[str, list[tuple[np.ndarray, np.ndarray]]]`：每模型的 (测试索引数组, 预测概率数组) 列表，用于绘制 pooled OOF 曲线
  - `summarize(fold_df) -> pd.DataFrame`（每模型 AUC mean±sd，按均值降序）
  - `compare_to_baseline(fold_df, baseline="Logistic_LASSO_baseline") -> pd.DataFrame`（配对 Wilcoxon + Holm 校正，列 `["model","mean_auc_diff","p_wilcoxon","p_holm"]`）

- [ ] **Step 1: 写失败测试**

创建 `tests/test_evaluation.py`：

```python
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from analysis.evaluation import (calibration_slope, youden_sens_spec,
                                 run_comparison, summarize, compare_to_baseline)
from analysis.preprocessing import AutoPreprocessor

rng = np.random.default_rng(2025)
n = 120
X = rng.normal(0, 1, (n, 6))
y = (X[:, 0] * 1.5 - X[:, 1] + rng.normal(0, 0.5, n) > 0).astype(int)

def test_calibration_slope_perfect_ranking():
    p = np.clip(y * 0.9 + 0.05, 1e-6, 1 - 1e-6)
    assert calibration_slope(y, p) > 5.0

def test_youden_perfect():
    p = y.astype(float)
    sens, spec = youden_sens_spec(y, p)
    assert sens == 1.0 and spec == 1.0

@pytest.fixture(scope="module")
def result():
    models = {
        "L2": Pipeline([("pre", AutoPreprocessor()),
                        ("clf", LogisticRegression(max_iter=1000))]),
        "RF": Pipeline([("pre", AutoPreprocessor()),
                        ("clf", RandomForestClassifier(n_estimators=30, random_state=1))]),
    }
    return run_comparison(X, y, models, grids=None,
                          n_folds=2, n_repeats=2, inner_folds=2, seed=2025)

def test_run_comparison_structure(result):
    fold_df, oof = result
    assert set(fold_df.columns) >= {"model","repeat","fold","auc","brier","sensitivity","specificity","calibration_slope"}
    assert len(fold_df) == 2 * 2 * 2  # models x folds x repeats
    assert fold_df["auc"].between(0, 1).all()
    for name, recs in oof.items():
        total = sum(len(idx) for idx, _ in recs)
        assert total == n  # OOF 覆盖全部样本

def test_determinism(result):
    fold_df, _ = result
    models = {
        "L2": Pipeline([("pre", AutoPreprocessor()),
                        ("clf", LogisticRegression(max_iter=1000))]),
    }
    fold_df2, _ = run_comparison(X, y, models, grids=None,
                                 n_folds=2, n_repeats=2, inner_folds=2, seed=2025)
    a = fold_df[fold_df.model == "L2"]["auc"].to_numpy()
    b = fold_df2["auc"].to_numpy()
    assert np.allclose(a, b)

def test_summarize_sorted(result):
    s = summarize(result[0])
    assert list(s["mean_auc"]) == sorted(s["mean_auc"], reverse=True)
    assert set(s["model"]) == {"L2", "RF"}

def test_compare_to_baseline(result):
    comp = compare_to_baseline(result[0], baseline="L2")
    assert list(comp["model"]) == ["RF"]
    assert "p_holm" in comp.columns and comp["p_holm"].between(0, 1).all()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
.venv/Scripts/python -m pytest tests/test_evaluation.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现评估引擎**

创建 `analysis/evaluation.py`：

```python
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold

from analysis.config import N_FOLDS, N_INNER_FOLDS, N_REPEATS, RANDOM_SEED


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_slope(y_true, y_prob) -> float:
    z = _logit(np.asarray(y_prob)).reshape(-1, 1)
    y = np.asarray(y_true)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(lr.coef_[0, 0])


def youden_sens_spec(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(tpr[i]), float(1 - fpr[i])  # sensitivity, specificity


def _fold_metrics(y_true, y_prob):
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "sensitivity": youden_sens_spec(y_true, y_prob)[0],
        "specificity": youden_sens_spec(y_true, y_prob)[1],
        "calibration_slope": calibration_slope(y_true, y_prob),
    }


def run_comparison(X, y, models, grids=None, n_folds=N_FOLDS, n_repeats=N_REPEATS,
                   inner_folds=N_INNER_FOLDS, seed=RANDOM_SEED):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    grids = grids or {}
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats,
                                   random_state=seed)
    rows = []
    oof_records = {}
    for name, est in models.items():
        oof_records[name] = []
        splits = list(rskf.split(X, y))
        for i, (tr, te) in enumerate(splits):
            rep, fold = i // n_folds, i % n_folds
            grid = grids.get(name)
            if grid:
                gs = GridSearchCV(
                    clone(est), grid,
                    cv=StratifiedKFold(inner_folds, shuffle=True, random_state=seed),
                    scoring="roc_auc", n_jobs=1, refit=True)
                gs.fit(X[tr], y[tr])
                fitted = gs.best_estimator_
            else:
                fitted = clone(est).fit(X[tr], y[tr])
            proba = fitted.predict_proba(X[te])[:, 1]
            metrics = _fold_metrics(y[te], proba)
            rows.append({"model": name, "repeat": rep, "fold": fold,
                         "n_test": len(te), **metrics})
            oof_records[name].append((te.copy(), proba.copy()))
    return pd.DataFrame(rows), oof_records


def summarize(fold_df: pd.DataFrame) -> pd.DataFrame:
    g = fold_df.groupby("model")["auc"]
    out = pd.DataFrame({
        "model": g.mean().index,
        "mean_auc": g.mean().to_numpy(),
        "sd_auc": g.std().to_numpy(),
        "min_auc": g.min().to_numpy(),
        "max_auc": g.max().to_numpy(),
    }).sort_values("mean_auc", ascending=False).reset_index(drop=True)
    return out


def _holm(pvals):
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted_sorted = np.maximum.accumulate((m - np.arange(m)) * pvals[order])
    adjusted = np.minimum(adjusted_sorted, 1.0)
    out = np.empty(m)
    out[order] = adjusted
    return out


def compare_to_baseline(fold_df: pd.DataFrame,
                        baseline="Logistic_LASSO_baseline") -> pd.DataFrame:
    pivot = fold_df.pivot_table(index=["repeat", "fold"], columns="model",
                                values="auc")
    rows = []
    for model in pivot.columns:
        if model == baseline:
            continue
        diff = pivot[model] - pivot[baseline]
        if np.allclose(diff, 0):
            stat, p = np.nan, 1.0
        else:
            stat, p = wilcoxon(diff)
        rows.append({"model": model, "mean_auc_diff": diff.mean(), "p_wilcoxon": p})
    comp = pd.DataFrame(rows)
    if len(comp):
        comp["p_holm"] = _holm(comp["p_wilcoxon"].to_numpy())
    return comp.sort_values("mean_auc_diff", ascending=False).reset_index(drop=True)
```

注意：`evaluation.py` 顶部需补一行 `from sklearn.linear_model import LogisticRegression`（`calibration_slope` 使用）。

- [ ] **Step 4: 运行测试确认通过**

```bash
.venv/Scripts/python -m pytest tests/test_evaluation.py -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add analysis/evaluation.py tests/test_evaluation.py
git commit -m "feat: nested repeated-stratified CV engine with metrics and paired stats"
```

---

### Task 8: 绘图模块（ROC / 校准曲线 / DCA）

**Files:**
- Create: `analysis/plots.py`
- Create: `tests/test_plots.py`

**Interfaces:**
- Consumes: `oof_records`（Task 7）、`dca` 模块、`matplotlib`
- Produces:
  - `plot_roc(y, oof_records, out_path) -> str`（各模型 pooled OOF ROC，图例含 AUC，返回 out_path 字符串）
  - `plot_calibration(y, oof_records, out_path, n_bins=5) -> str`
  - `plot_dca(y, oof_records, out_path, thresholds=np.arange(0.01, 0.99, 0.01)) -> str`
  - 均内部 `matplotlib.use("Agg")`，无头运行；文件成功写出即返回路径。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_plots.py`：

```python
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
    import os
    for p in (p1, p2, p3):
        assert os.path.exists(p) and os.path.getsize(p) > 0
```

- [ ] **Step 2: 运行测试确认失败**

```bash
.venv/Scripts/python -m pytest tests/test_plots.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现绘图**

创建 `analysis/plots.py`：

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve

from analysis.dca import net_benefit, net_benefit_treat_all, net_benefit_treat_none


def _pooled(y, oof_records):
    """把每折 (索引, 概率) 拼回与原样本对齐的 OOF 概率向量。"""
    p = np.full(len(y), np.nan)
    for idx, proba in oof_records:
        p[idx] = proba
    assert not np.any(np.isnan(p)), "OOF predictions must cover all samples"
    return p


def plot_roc(y, oof_records: dict, out_path) -> str:
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, recs in oof_records.items():
        p = _pooled(y, recs)
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title("ROC (pooled out-of-fold predictions)")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)
    return str(out_path)


def plot_calibration(y, oof_records: dict, out_path, n_bins=5) -> str:
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, recs in oof_records.items():
        p = _pooled(y, recs)
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed proportion")
    ax.set_title("Calibration (pooled out-of-fold predictions)")
    ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)
    return str(out_path)


def plot_dca(y, oof_records: dict, out_path,
             thresholds=np.arange(0.01, 0.99, 0.01)) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(thresholds, net_benefit_treat_all(y, thresholds), "k--", label="Treat all")
    ax.plot(thresholds, net_benefit_treat_none(thresholds), "k:", label="Treat none")
    for name, recs in oof_records.items():
        p = _pooled(y, recs)
        ax.plot(thresholds, net_benefit(y, p, thresholds), label=name)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision Curve Analysis (pooled out-of-fold predictions)")
    ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)
    return str(out_path)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
.venv/Scripts/python -m pytest tests/test_plots.py -v
```

Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add analysis/plots.py tests/test_plots.py
git commit -m "feat: ROC, calibration and DCA plotting from pooled OOF predictions"
```

---

### Task 9: SHAP 可解释性分析

**Files:**
- Create: `analysis/explain.py`
- Create: `tests/test_explain.py`

**Interfaces:**
- Consumes: 训练好的 Pipeline（含 `pre` 步）、`feature_names`
- Produces:
  - `explain_best(model, X, feature_names, out_dir, seed=2025) -> dict`
    - 在全数据上：树模型（XGBoost/LightGBM/RandomForest）用 `shap.TreeExplainer`，其余用 `shap.KernelExplainer`（背景采样 ≤60 行）
    - 产出 `shap_summary.png`（beeswarm）与 `shap_bar.png`（均值 |SHAP| 条形图）
    - 返回 `{"importance": pd.Series(按均值|SHAP|降序), "method": "TreeExplainer"|"KernelExplainer"}`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_explain.py`：

```python
import numpy as np

from analysis.explain import explain_best
from analysis.models import get_models

rng = np.random.default_rng(0)
X = rng.normal(0, 1, (60, 4))
y = (X[:, 0] - X[:, 1] + rng.normal(0, 0.3, 60) > 0).astype(int)
names = [f"f{i}" for i in range(4)]

def test_explain_rf(tmp_path):
    model = get_models(seed=1, light=True)["RF"]
    model.fit(X, y)
    result = explain_best(model, X, names, tmp_path, seed=1)
    assert result["method"] == "TreeExplainer"
    assert list(result["importance"].index) == names or len(result["importance"]) == 4
    assert (tmp_path / "shap_summary.png").exists()
    assert (tmp_path / "shap_bar.png").exists()

def test_explain_kernel_path(tmp_path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from analysis.preprocessing import AutoPreprocessor
    model = Pipeline([("pre", AutoPreprocessor()),
                      ("clf", LogisticRegression(max_iter=500))]).fit(X, y)
    result = explain_best(model, X, names, tmp_path, seed=1)
    assert result["method"] == "KernelExplainer"
    assert len(result["importance"]) == 4
```

- [ ] **Step 2: 运行测试确认失败**

```bash
.venv/Scripts/python -m pytest tests/test_explain.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现 explain**

创建 `analysis/explain.py`：

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

TREE_MODELS = ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier")


def _final_estimator(model):
    return model[-1] if hasattr(model, "steps") else model


def _transform_features(model, X):
    if hasattr(model, "steps") and model.steps[0][0] == "pre":
        return model.steps[0][1].transform(X)
    return np.asarray(X, dtype=np.float64)


def explain_best(model, X, feature_names, out_dir, seed=2025) -> dict:
    X = np.asarray(X, dtype=np.float64)
    Xt = _transform_features(model, X)
    clf = _final_estimator(model)
    out_dir.mkdir(parents=True, exist_ok=True)

    if type(clf).__name__ in TREE_MODELS:
        method = "TreeExplainer"
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xt)
    else:
        method = "KernelExplainer"
        bg = shap.sample(Xt, min(60, len(Xt)), random_state=seed)
        explainer = shap.KernelExplainer(
            lambda d: clf.predict_proba(d)[:, 1], bg, seed=seed)
        sv = explainer.shap_values(Xt, nsamples=100)

    sv = np.asarray(sv)
    if sv.ndim == 3:          # (n, features, classes) 或 (classes, n, features)
        if sv.shape[-1] == 2:
            sv = sv[..., 1]
        elif sv.shape[0] == 2:
            sv = sv[1]

    importance = pd.Series(np.abs(sv).mean(axis=0), index=feature_names)
    importance = importance.sort_values(ascending=False)

    plt.figure()
    shap.summary_plot(sv, Xt, feature_names=feature_names, show=False)
    plt.tight_layout(); plt.savefig(out_dir / "shap_summary.png", dpi=200)
    plt.close("all")

    plt.figure()
    importance[::-1].plot(kind="barh", figsize=(8, max(4, 0.3 * len(feature_names))))
    plt.xlabel("mean |SHAP value|")
    plt.tight_layout(); plt.savefig(out_dir / "shap_bar.png", dpi=200)
    plt.close("all")

    return {"importance": importance, "method": method}
```

- [ ] **Step 4: 运行测试确认通过**

```bash
.venv/Scripts/python -m pytest tests/test_explain.py -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add analysis/explain.py tests/test_explain.py
git commit -m "feat: SHAP explainability for best model (tree + kernel paths)"
```

---

### Task 10: 主入口与报告生成

**Files:**
- Create: `analysis/report.py`
- Create: `analysis/run_analysis.py`
- Create: `tests/test_report.py`

**Interfaces:**
- Consumes: 前面全部模块
- Produces:
  - `write_report(summary_df, comp_df, best_model, shap_method, out_path, extra: dict) -> str`
  - CLI：`.venv/Scripts/python -m analysis.run_analysis [--quick]`，完整运行产出：
    - `outputs/tables/fold_results.csv`、`outputs/tables/model_summary.csv`、`outputs/tables/model_comparison_vs_baseline.csv`
    - `outputs/figures/roc_curves.png`、`calibration_curves.png`、`dca_curves.png`、`shap_summary.png`、`shap_bar.png`
    - `outputs/report.md`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_report.py`：

```python
import pandas as pd

from analysis.report import write_report

summary = pd.DataFrame({
    "model": ["TabNet", "Logistic_LASSO_baseline"],
    "mean_auc": [0.85, 0.80], "sd_auc": [0.05, 0.06],
    "min_auc": [0.75, 0.70], "max_auc": [0.92, 0.88],
})
comp = pd.DataFrame({
    "model": ["TabNet"], "mean_auc_diff": [0.05],
    "p_wilcoxon": [0.03], "p_holm": [0.03],
})

def test_write_report(tmp_path):
    out = tmp_path / "report.md"
    write_report(summary, comp, "TabNet", "TreeExplainer", out,
                 extra={"n_repeats": 5, "n_folds": 5})
    text = out.read_text(encoding="utf-8")
    assert "TabNet" in text and "0.850" in text
    assert "TreeExplainer" in text
    assert "| model |" in text or "|model|" in text
```

- [ ] **Step 2: 运行测试确认失败**

```bash
.venv/Scripts/python -m pytest tests/test_report.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 实现 report 与主入口**

创建 `analysis/report.py`：

```python
import pandas as pd


def write_report(summary_df, comp_df, best_model, shap_method, out_path,
                 extra=None) -> str:
    extra = extra or {}
    lines = [
        "# CRT 应答预测：传统基线 vs AI 模型对比报告",
        "",
        f"- 交叉验证方案：外层 {extra.get('n_folds', 5)} 折 × {extra.get('n_repeats', 5)} 次重复分层 CV，"
        f"内层 {extra.get('inner_folds', 3)} 折调参（嵌套，无泄漏）",
        f"- 样本：163 例 CRT 患者（应答 {extra.get('n_events', 93)} 例）",
        f"- **最优模型：{best_model}**（SHAP 方法：{shap_method}）",
        "",
        "## 模型汇总（外层折 AUC）",
        "",
        summary_df.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## 与 LASSO 基线的配对比较（同一折配对 Wilcoxon + Holm 校正）",
        "",
        comp_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 图形",
        "",
        "- ![ROC](figures/roc_curves.png)",
        "- ![Calibration](figures/calibration_curves.png)",
        "- ![DCA](figures/dca_curves.png)",
        "- ![SHAP summary](figures/shap_summary.png)",
        "- ![SHAP bar](figures/shap_bar.png)",
        "",
        "## 方法说明（供论文 Methods 复用）",
        "",
        "All models were embedded in leak-free pipelines and evaluated with nested "
        "repeated stratified cross-validation (outer " + str(extra.get('n_folds', 5)) +
        "-fold × " + str(extra.get('n_repeats', 5)) + " repeats; inner " +
        str(extra.get('inner_folds', 3)) +
        "-fold grid search optimizing AUC). Discrimination (AUC), calibration "
        "(Brier score, calibration slope), decision-curve net benefit, and Youden "
        "sensitivity/specificity were computed per outer fold. Pairwise differences "
        "versus the LASSO baseline were tested with Wilcoxon signed-rank tests and "
        "Holm correction. SHAP was applied to the best-performing model.",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)
```

创建 `analysis/run_analysis.py`：

```python
import argparse

from analysis.config import (FIGURE_DIR, N_FOLDS, N_INNER_FOLDS, N_REPEATS,
                             RANDOM_SEED, TABLE_DIR)
from analysis.data_loader import load_data, split_xy
from analysis.models import get_models, get_grids
from analysis.evaluation import run_comparison, summarize, compare_to_baseline
from analysis.plots import plot_roc, plot_calibration, plot_dca
from analysis.explain import explain_best
from analysis.report import write_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="1 repeat, no grids - smoke test of the full pipeline")
    args = ap.parse_args()

    n_folds, n_repeats, inner_folds = N_FOLDS, N_REPEATS, N_INNER_FOLDS
    light = False
    if args.quick:
        n_repeats, inner_folds, light = 1, 2, True

    df = load_data()
    X, y, feature_names = split_xy(df)
    models = get_models(seed=RANDOM_SEED, light=light)
    grids = {} if light else get_grids(seed=RANDOM_SEED)

    fold_df, oof = run_comparison(X, y, models, grids=grids,
                                  n_folds=n_folds, n_repeats=n_repeats,
                                  inner_folds=inner_folds, seed=RANDOM_SEED)
    summary = summarize(fold_df)
    comp = compare_to_baseline(fold_df)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(TABLE_DIR / "fold_results.csv", index=False)
    summary.to_csv(TABLE_DIR / "model_summary.csv", index=False)
    comp.to_csv(TABLE_DIR / "model_comparison_vs_baseline.csv", index=False)

    plot_roc(y, oof, FIGURE_DIR / "roc_curves.png")
    plot_calibration(y, oof, FIGURE_DIR / "calibration_curves.png")
    plot_dca(y, oof, FIGURE_DIR / "dca_curves.png")

    best_model = summary.iloc[0]["model"]
    best_est = models[best_model].fit(X, y)  # 全数据拟合，仅用于 SHAP 展示
    shap_res = explain_best(best_est, X, feature_names, FIGURE_DIR, seed=RANDOM_SEED)

    report_path = write_report(
        summary, comp, best_model, shap_res["method"],
        TABLE_DIR.parent / "report.md",
        extra={"n_folds": n_folds, "n_repeats": n_repeats,
               "inner_folds": inner_folds, "n_events": int(y.sum())})

    print(summary.to_string(index=False))
    print(f"\nBest model: {best_model}")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
```

注意：`run_comparison` 返回的 `oof` 中若 best_model 存在，直接用；此处 `models[best_model]` 是未拟合的原始对象，重新全数据拟合仅供 SHAP。

- [ ] **Step 4: 运行测试确认通过**

```bash
.venv/Scripts/python -m pytest tests/test_report.py -v
```

Expected: 1 passed（`to_markdown` 需要 `tabulate`，若报错先 `.venv/Scripts/python -m pip install tabulate` 并把 `tabulate` 加入 `requirements.txt`）

- [ ] **Step 5: 全套测试回归**

```bash
.venv/Scripts/python -m pytest tests/ -v
```

Expected: 全部 passed

- [ ] **Step 6: quick 模式冒烟运行完整管道**

```bash
.venv/Scripts/python -m analysis.run_analysis --quick
```

Expected: 打印模型汇总表（8 个模型），无异常退出；`outputs/report.md`、`outputs/tables/*.csv`、`outputs/figures/*.png` 全部生成。TabNet 在 quick 下也应在几分钟内完成。

- [ ] **Step 7: Commit**

```bash
git add analysis/report.py analysis/run_analysis.py tests/test_report.py requirements.txt
git commit -m "feat: main entry point with quick mode and markdown report generation"
```

---

### Task 11: 完整运行与结果交付

**Files:**
- Create: `outputs/report.md`（运行产物，加入 git）
- Create: `outputs/tables/*.csv`、`outputs/figures/*.png`（运行产物，加入 git）

**Interfaces:**
- Consumes: 全部模块
- Produces: 最终交付物——完整对比结果与报告（可直接粘贴进论文草稿/回复编辑）

- [ ] **Step 1: 完整运行（后台，预计 1–3 小时）**

```bash
.venv/Scripts/python -m analysis.run_analysis 2>&1 | tee outputs/run_log.txt
```

注意：先 `mkdir -p outputs`。运行期间不要修改 `data.xlsx`。

- [ ] **Step 2: 检查结果合理性**

```bash
.venv/Scripts/python -c "
import pandas as pd
s = pd.read_csv('outputs/tables/model_summary.csv')
print(s)
c = pd.read_csv('outputs/tables/model_comparison_vs_baseline.csv')
print(c)
"
```

Expected 检查点：
- 每个模型 mean_auc 在 0.6–0.95 之间；基线（LASSO）约 0.75–0.85（参考论文全数据 0.845）
- 最优 AI 模型相对基线的 mean_auc_diff 为正则更有说服力；p_holm < 0.05 时可在论文中主张"显著优于基线"
- 若 TabNet/MLP 的 AUC 显著低于树模型（小样本常见），这正是论文可写的发现："在 n=163 的小样本场景下，梯度提升树优于深度学习"——同样回应编辑的 AI 要求，不必强行让深度学习胜出

- [ ] **Step 3: 人工查看五张图**

用 Read 工具查看 `outputs/figures/*.png`：ROC 图各模型曲线应可分辨；校准曲线应大致沿对角线；DCA 图在合理阈值区间内模型曲线应高于 treat-all/treat-none；SHAP 图特征名应为中文数据集原始列名（含 `βblocker` 等）。

- [ ] **Step 4: 提交全部产物**

```bash
sed -i 's|^outputs/$||' .gitignore
git add outputs/ .gitignore
git commit -m "results: full nested-CV comparison of 8 models with SHAP on best model"
```

- [ ] **Step 5: 更新 背景说明.md 的"下一步任务"**

把第 7 节标记为已完成，并附最优模型名称与 AUC，供后续写 article 初稿引用。

```bash
git add 背景说明.md
git commit -m "docs: mark AI model comparison task as done in background notes"
```

---

## Self-Review 记录

- **Spec 覆盖**：编辑要求（AI 算法✓ Task 5、深度学习✓ MLP+TabNet Task 5、大数据挖掘—以多模型基准对比与 SHAP 覆盖，论文叙述层面落实）、聊天记录要求（测试哪种 AI 模型适配✓ Task 7/11、response 因变量✓ Task 2）均有对应任务；综述素材由 Task 11 的报告与图表支撑。
- **占位符扫描**：所有代码步骤均给出完整代码；无 TBD/TODO。
- **类型一致性**：`get_models` 的 8 个键在 Task 5/7/10 一致；`fold_df` 列名在 Task 7/10 一致；`oof_records` 结构 `dict[str, list[(idx_array, proba_array)]]` 在 Task 7/8/10 一致；`explain_best` 返回 dict 的键在 Task 9/10 一致。
- **已知风险**（已在任务内写明处理）：① `data.xlsx` 列数若与 29 不符 → Task 2 校验会显式报错，以真实数据修正常量；② `to_markdown` 依赖 `tabulate` → Task 10 Step 4 处理；③ TabNet 全网格运行慢 → 提供 `--quick` 与缩小网格；④ 无人回答环境下 `shap.summary_plot` 偶发字号警告不影响产物。

# ITM 投稿：CRT 应答预测的表格机器学习 + SHAP 可解释性 实施计划（v1，已作废）

> **本计划已被 `2026-10-13-itm-crt-ai-v2.md` 取代，不要再执行。** v1 假设仓库是空白的、要新建 `src/crt_ai/` 包；实际仓库已有完整且 49 测试全绿的 `analysis/` 实现，用户裁定改为**扩展 `analysis/`**（见 v2 计划与 SDD ledger 的 Ruling R5）。以下内容仅作历史归档保留。

**Goal:** 在已有的 163 例 CRT 患者 SPECT 相位分析数据上，构建一套泄漏安全（leakage-safe）的表格机器学习基准 + SHAP 可解释性流水线，产出满足 *Intelligent Translational Medicine*（ITM）"AI 算法层"要求的 article 级结果工件。

**Architecture:** 单一 Python 包 `src/crt_ai/`，按"数据 → 预处理/切分 → 指标 → 基线 → 特征选择 → 模型注册表 → 嵌套 CV 基准 → 统计比较 → SHAP → 冻结 → 报告 → 应用"分层，每层一个文件、一个测试文件。所有特征选择与超参调优只发生在训练折内（嵌套 CV：外层 5 折 × 10 重复，内层 5 折调参），以 163 例 / 70 事件的样本量控制乐观偏倚。

**Tech Stack:** 复用本机已有 miniconda base 环境 `C:/Users/yaoyue/miniconda3/python.exe`（Python 3.12.4；numpy 2.4.6、pandas 2.3.3、scipy 1.18.0、scikit-learn 1.9.0、xgboost 3.4.1、lightgbm 4.7.0、shap 0.53.0、matplotlib 3.11.1、openpyxl 3.1.5、pytest 9.1.1、joblib 1.5.3、torch 2.6.0+cu124/CUDA 可用）。**不建 venv、不安装任何新包**：缺 statsmodels → Task 6 自写 IRLS Wald；缺 gradio → Task 13 改用标准库 `http.server`。

**Spec:** 本文件夹即规格来源 —— `背景说明.md`（ITM scope + 三人分工 + 已发表 LASSO 列线图论文 Jiang et al., QIMS 2025;15(5):4247-4261, DOI 10.21037/qims-2024-2700）、`编辑主题评价.jpg`（编辑意见：原稿无 AI/深度学习/影像智能分析/大数据挖掘，需补充）、`data.xlsx`（163 行 × 29 列，`response` 为因变量，其余为候选自变量）。

## Global Constraints

- 目标期刊 ITM 要求稿件必须包含 AI 算法层（编辑意见原文），纯 LASSO+列线图不算。
- `data.xlsx` 是只读文件（权限 `-r--r--r--`）：禁止任何写操作，一切清洗只发生在内存或派生 CSV。
- 样本量 n=163、事件数 70、建模变量 26 个（EPV≈2.7）：单因素筛选、共线性聚类、超参调优**必须**在训练折内完成，禁止全样本选择后再交叉验证。
- 随机种子统一 `splits.SEED = 42`；同一命令两次运行必须得到逐位相同的结果。
- 报告遵循 TRIPOD+AI；正文**不得**声称外部验证（本数据无外部队列、无日期字段）。
- 不引入需要 GPU 或 >2GB 内存的依赖；全部模型 CPU 可跑。
- **零新安装**：解释器固定为 `C:/Users/yaoyue/miniconda3/python.exe`（miniconda base，已含全部建模依赖与 CUDA torch）；禁止创建 venv、禁止 pip/conda 安装任何包；缺失能力一律自写实现（Task 6 的 IRLS Wald、Task 13 的标准库 HTTP）。
- 沟通用中文，稿件正文用英文。
- 每个 Task 结束执行一次 git commit；commit message 用 `feat:` / `test:` / `fix:` 前缀。
- 完整基准（12 模型 × 50 外折 × 内层网格）预计 30–60 分钟；`--smoke` 模式（3 模型 × 2 重复）约 2 分钟，测试一律走 smoke 规模或合成数据。

## 范围检查（Scope Check）

本计划只覆盖 **article 的建模与结果工件**。以下两项是独立交付物，**不在本计划内**，完成后另起计划：

1. 配套综述（"AI + 心血管/癌症/感官"大主题）——无代码依赖，单独选题与写作计划。
2. 影像级深度学习（直接吃 gated SPECT 原始图像）——需要向于院索取原始 DICOM/相位图，数据尚不存在。

## 文件结构

```
20261013-article/
├── data.xlsx                      # 只读源数据（已存在，纳入 git）
├── requirements.txt               # 新建：固定版本依赖
├── pytest.ini                     # 新建：pytest 配置
├── .gitignore                     # 新建
├── docs/
│   ├── superpowers/plans/2026-10-13-itm-crt-tableml-shap.md   # 本计划
│   ├── data-fingerprint.txt       # 新建：data.xlsx 的 sha256
│   └── questions-for-collaborators.md  # 新建：需向于院索取的字段清单
├── src/crt_ai/
│   ├── __init__.py
│   ├── data.py                    # 读取/表头归一/异常值/近常量剔除
│   ├── pipeline.py                # 无泄漏预处理器（按变量类型分块）
│   ├── splits.py                  # 嵌套重复分层 CV 切分
│   ├── evaluate.py                # AUC/AUPRC/Brier/校准/DCA 净获益
│   ├── baseline.py                # 复现已发表 LASSO 列线图（含乐观偏倚对照）
│   ├── features.py                # CV 内单因素筛选 + 共线性聚类
│   ├── models.py                  # 12 个候选模型注册表 + stacking
│   ├── harness.py                 # 嵌套 CV 基准测试台（调参在折内）
│   ├── compare.py                 # bootstrap CI + DeLong 检验
│   ├── explain.py                 # SHAP 全局/个体解释
│   ├── final.py                   # 最优模型冻结 + 敏感性分析
│   ├── report.py                  # Table 1 / 性能表 / 三张图
│   ├── app.py                     # gradio 在线预测器（数字健康落点）
│   └── main.py                    # CLI 入口（--smoke / --models / --repeats）
├── tests/
│   ├── test_data.py
│   ├── test_pipeline.py
│   ├── test_splits.py
│   ├── test_evaluate.py
│   ├── test_baseline.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_harness.py
│   ├── test_compare.py
│   ├── test_explain.py
│   ├── test_final.py
│   └── test_report.py
└── outputs/                       # 运行产物（gitignore）
    ├── model_summary.csv
    ├── oof_by_repeat.csv
    ├── table1.csv
    └── figures/
```

---

### Task 1: 仓库骨架、环境记录与数据指纹（零安装）

**Files:**
- Create: `pytest.ini`, `.gitignore`, `conftest.py`, `src/crt_ai/__init__.py`, `docs/environment.md`, `docs/data-fingerprint.txt`, `docs/questions-for-collaborators.md`
- Test: 无（交付物是骨架与环境记录，验证靠 import 冒烟）

**Interfaces:**
- Consumes: 无
- Produces: `docs/environment.md`（解释器路径 + 实测版本，后续所有任务的唯一环境事实源）；`docs/data-fingerprint.txt` 中的 sha256；`conftest.py` 把 `src` 注入 `sys.path`

- [ ] **Step 1: 写配置与骨架文件**

```ini
# pytest.ini
[pytest]
testpaths = tests
addopts = -q
```

```text
# .gitignore
__pycache__/
*.pyc
outputs/
.pytest_cache/
.superpowers/
```

```python
# src/crt_ai/__init__.py
__version__ = "0.1.0"
```

```python
# conftest.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
```

- [ ] **Step 2: 探测环境并写 docs/environment.md（禁止安装）**

Run:
```bash
/c/Users/yaoyue/miniconda3/python.exe - <<'EOF' > docs/environment.md
import sys, importlib
print("# environment (probed; install nothing)")
print("interpreter:", sys.executable)
print("python:", sys.version.split()[0])
for m in ["numpy","pandas","scipy","sklearn","xgboost","lightgbm","shap","matplotlib","openpyxl","pytest","joblib","torch"]:
    try:
        mod = importlib.import_module(m)
        print(f"{m}=={getattr(mod,'__version__','?')}")
    except Exception:
        print(f"{m}=MISSING")
import torch
print("cuda:", torch.cuda.is_available())
EOF
```
Expected: 无 `=MISSING` 行，且 `cuda: True`。

- [ ] **Step 3: import 冒烟**

Run:
```bash
/c/Users/yaoyue/miniconda3/python.exe -c "import numpy,pandas,scipy,sklearn,xgboost,lightgbm,shap,matplotlib,pytest;print('ok')"
```
Expected: `ok`。若任一 import 失败：**不要安装**，向控制器报告并等待裁定。

- [ ] **Step 4: 记录数据指纹与协作问题清单**

Run:
```bash
/c/Users/yaoyue/miniconda3/python.exe -c "import hashlib,pathlib;print(hashlib.sha256(pathlib.Path('data.xlsx').read_bytes()).hexdigest())" > docs/data-fingerprint.txt
```

写 `docs/questions-for-collaborators.md`：

```markdown
# 需向于院/合作者确认的字段（阻塞项）

1. 每位患者的 CRT 植入年份与随访截止日期 —— 当前数据无日期字段，无法做时间切分验证。
2. PVC 计数（已发表摘要报告了 1,392.98 vs 2,283.60，但 data.xlsx 中不存在该列）。
3. Age=165 这一行的真实年龄（其余最大 86，判定为录入错误）。
4. systolic_match / diastolic_match 的精确定义（导线是否位于最晚收缩/舒张段）。
5. 是否存在第二中心队列或原始 gated SPECT 相位图（DICOM）——决定能否做外部验证或影像级深度学习。
```

- [ ] **Step 5: git init、建分支并提交**

Run:
```bash
git init
git checkout -b feat/itm-crt-ai
git add -A
git commit -m "chore: project skeleton, environment record, data fingerprint"
```
Expected: 分支 `feat/itm-crt-ai` 上的首次提交包含 `data.xlsx`、`背景说明.md`、`编辑主题评价.jpg` 与骨架文件。

---

### Task 2: 数据加载与清洗（data.py）

**Files:**
- Create: `src/crt_ai/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `data.xlsx`（只读）
- Produces:
  - `load_raw(path=DATA_XLSX) -> pd.DataFrame`：原始 29 列，表头已归一
  - `load_meta(path=DATA_XLSX) -> pd.DataFrame`：列 `source_header, column, n_missing, n_unique, minority_prevalence`
  - `load_dataset(path=DATA_XLSX) -> tuple[pd.DataFrame, pd.Series]`：`(X, y)`，X 为 163×26 float，y 为 0/1 Series
  - 常量 `DATA_XLSX`, `AGE_MAX_VALID = 110`, `NEAR_CONSTANT_MINORITY = 0.05`

清洗规则（全部为无监督规则，不接触 y，故可在加载期执行）：表头 `*blocker` → `beta_blocker`、`NT_proBNP*` → `nt_proBNP_grade`；`Age > 110` → NaN；二值列中少数类占比 < 0.05 者剔除（命中 `beta_blocker` 4.9%、`Diuretic` 4.9%；`Spironolactone` 5.5% 保留）。被剔除的两个药物变量仍会在 Task 12 的 Table 1 中报告（Table 1 用 `load_raw`，模型矩阵用 `load_dataset`）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_data.py
import numpy as np
import pytest

from crt_ai import data


def test_shape_and_outcome():
    x, y = data.load_dataset()
    assert x.shape == (163, 26)
    assert int(y.sum()) == 93


def test_age_outlier_becomes_missing():
    x, _ = data.load_dataset()
    assert x["Age"].isna().sum() == 1


def test_near_constant_dropped():
    x, _ = data.load_dataset()
    assert "beta_blocker" not in x.columns
    assert "Diuretic" not in x.columns
    assert "Spironolactone" in x.columns


def test_headers_normalized():
    x, _ = data.load_dataset()
    assert "nt_proBNP_grade" in x.columns
    assert not any(c.startswith("NT_proBNP(") for c in x.columns)


def test_no_other_missing():
    x, _ = data.load_dataset()
    assert x.drop(columns=["Age"]).isna().sum().sum() == 0


def test_meta_frame_contents():
    meta = data.load_meta()
    row = meta[meta["column"] == "beta_blocker"].iloc[0]
    assert row["minority_prevalence"] == pytest.approx(0.049, abs=0.002)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `/c/Users/yaoyue/miniconda3/python.exe -m pytest tests/test_data.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'crt_ai'` 或 import error。

- [ ] **Step 3: 写最小实现**

```python
# src/crt_ai/data.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_XLSX = Path(__file__).resolve().parents[2] / "data.xlsx"
AGE_MAX_VALID = 110
NEAR_CONSTANT_MINORITY = 0.05


def _normalize_header(h: str) -> str:
    h = str(h).strip()
    if h.endswith("blocker"):
        return "beta_blocker"
    if h.startswith("NT_proBNP"):
        return "nt_proBNP_grade"
    return h


def load_raw(path: Path = DATA_XLSX) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    df.columns = [_normalize_header(c) for c in df.columns]
    return df


def load_meta(path: Path = DATA_XLSX) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    rows = []
    for c in raw.columns:
        vals = raw[c].dropna()
        is_binary = set(vals.unique()) <= {0, 1}
        rows.append(
            {
                "source_header": str(c),
                "column": _normalize_header(c),
                "n_missing": int(raw[c].isna().sum()),
                "n_unique": int(vals.nunique()),
                "minority_prevalence": (
                    float(min(vals.mean(), 1 - vals.mean())) if is_binary else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def load_dataset(path: Path = DATA_XLSX) -> tuple[pd.DataFrame, pd.Series]:
    df = load_raw(path)
    if "response" not in df.columns:
        raise KeyError(f"response column missing; first columns: {list(df.columns)[:5]}")
    y = df["response"].astype(int).rename("response")
    x = df.drop(columns=["response"]).astype(float)
    x.loc[x["Age"] > AGE_MAX_VALID, "Age"] = np.nan
    dropped = []
    for c in list(x.columns):
        vals = x[c].dropna()
        if set(vals.unique()) <= {0.0, 1.0}:
            minority = min(float((vals == 1).mean()), float((vals == 0).mean()))
            if minority < NEAR_CONSTANT_MINORITY:
                dropped.append(c)
    x = x.drop(columns=dropped)
    return x, y
```

- [ ] **Step 4: 让测试可导入包并运行通过**

在项目根创建 `conftest.py`：

```python
# conftest.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
```

Run: `/c/Users/yaoyue/miniconda3/python.exe -m pytest tests/test_data.py -v`
Expected: 6 passed。

- [ ] **Step 5: 提交**

```bash
git add conftest.py src/crt_ai/data.py tests/test_data.py
git commit -m "feat: dataset loader with header normalization, age outlier and near-constant rules"
```

---

### Task 3: 无泄漏预处理与嵌套 CV 切分（pipeline.py, splits.py）

**Files:**
- Create: `src/crt_ai/pipeline.py`, `src/crt_ai/splits.py`
- Test: `tests/test_pipeline.py`, `tests/test_splits.py`

**Interfaces:**
- Consumes: `data.load_dataset`
- Produces:
  - `pipeline.split_feature_kinds(x) -> dict[str, list[str]]`：键 `numeric / ordinal / binary`
  - `pipeline.make_preprocessor(x) -> ColumnTransformer`：numeric 块 = 中位数插补 + 标准化；ordinal 与 binary 块 = 仅中位数插补；空块自动省略；列顺序与列名保持不变（无 one-hot，因所有预测变量已编码为 0/1 或有序/连续）
  - `splits.SEED = 42`, `splits.N_OUTER = 5`, `splits.N_REPEATS = 10`, `splits.N_INNER = 5`
  - `splits.Fold`（NamedTuple：`repeat, fold, train_index, test_index, inner_cv`）
  - `splits.make_outer_cv(seed, n_repeats) -> RepeatedStratifiedKFold`
  - `splits.make_inner_cv(seed) -> StratifiedKFold`
  - `splits.iter_nested_cv(x, y, seed, n_repeats) -> Iterator[Fold]`：共 `n_repeats * 5` 折，内层 cv 种子 = `seed + repeat`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_pipeline.py
import numpy as np

from crt_ai import data
from crt_ai.pipeline import make_preprocessor, split_feature_kinds


def test_feature_kinds():
    x, _ = data.load_dataset()
    kinds = split_feature_kinds(x)
    assert kinds["ordinal"] == ["nt_proBNP_grade", "NYHA"]
    assert "QRSd_ms" in kinds["numeric"]
    assert "NS_VT" in kinds["binary"]


def test_no_nan_after_transform_and_names_preserved():
    x, y = data.load_dataset()
    pre = make_preprocessor(x)
    xt = pre.fit_transform(x, y)
    assert np.isnan(xt).sum() == 0
    assert xt.shape == x.shape


def test_imputer_uses_train_median_only():
    x, y = data.load_dataset()
    tr = x.dropna()
    te = x[x["Age"].isna()]
    pre = make_preprocessor(tr)
    pre.fit(tr, y.loc[tr.index])
    filled = te.copy()
    filled["Age"] = tr["Age"].median()
    assert np.allclose(pre.transform(te), pre.transform(filled))
```

```python
# tests/test_splits.py
from crt_ai import data, splits


def test_fold_count_and_partition():
    x, y = data.load_dataset()
    folds = list(splits.iter_nested_cv(x, y))
    assert len(folds) == 50
    for f in folds:
        assert f.fold in range(5)
        assert set(f.train_index) & set(f.test_index) == set()
        assert len(f.test_index) in (32, 33)
        assert int(y.iloc[f.test_index].sum()) >= 10


def test_deterministic():
    x, y = data.load_dataset()
    a = [f.test_index.tolist() for f in splits.iter_nested_cv(x, y)]
    b = [f.test_index.tolist() for f in splits.iter_nested_cv(x, y)]
    assert a == b


def test_inner_cv_is_stratified():
    x, y = data.load_dataset()
    fold = next(splits.iter_nested_cv(x, y))
    assert fold.inner_cv.n_splits == 5
```

- [ ] **Step 2: 运行测试确认失败**

Run: `/c/Users/yaoyue/miniconda3/python.exe -m pytest tests/test_pipeline.py tests/test_splits.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'crt_ai.pipeline'`。

- [ ] **Step 3: 写最小实现**

```python
# src/crt_ai/pipeline.py
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ORDINAL_COLS = ("nt_proBNP_grade", "NYHA")


def split_feature_kinds(x) -> dict[str, list[str]]:
    numeric, ordinal, binary = [], [], []
    for c in x.columns:
        if c in ORDINAL_COLS:
            ordinal.append(c)
        elif set(np.unique(x[c].dropna())) <= {0.0, 1.0}:
            binary.append(c)
        else:
            numeric.append(c)
    return {"numeric": numeric, "ordinal": ordinal, "binary": binary}


def make_preprocessor(x) -> ColumnTransformer:
    kinds = split_feature_kinds(x)
    transformers = []
    if kinds["numeric"]:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                kinds["numeric"],
            )
        )
    if kinds["ordinal"]:
        transformers.append(("ordinal", SimpleImputer(strategy="median"), kinds["ordinal"]))
    if kinds["binary"]:
        transformers.append(("binary", SimpleImputer(strategy="median"), kinds["binary"]))
    return ColumnTransformer(transformers=transformers, remainder="drop")
```

```python
# src/crt_ai/splits.py
from __future__ import annotations

from typing import Iterator, NamedTuple

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

SEED = 42
N_OUTER = 5
N_REPEATS = 10
N_INNER = 5


class Fold(NamedTuple):
    repeat: int
    fold: int
    train_index: tuple[int, ...]
    test_index: tuple[int, ...]
    inner_cv: StratifiedKFold


def make_outer_cv(seed: int = SEED, n_repeats: int = N_REPEATS) -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(n_splits=N_OUTER, n_repeats=n_repeats, random_state=seed)


def make_inner_cv(seed: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=seed)


def iter_nested_cv(x, y, seed: int = SEED, n_repeats: int = N_REPEATS) -> Iterator[Fold]:
    for i, (tr, te) in enumerate(make_outer_cv(seed, n_repeats).split(x, y)):
        repeat = i // N_OUTER
        yield Fold(
            repeat=repeat,
            fold=i % N_OUTER,
            train_index=tuple(int(v) for v in tr),
            test_index=tuple(int(v) for v in te),
            inner_cv=make_inner_cv(seed + repeat),
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `/c/Users/yaoyue/miniconda3/python.exe -m pytest tests/test_pipeline.py tests/test_splits.py -v`
Expected: 6 passed。

- [ ] **Step 5: 提交**

```bash
git add src/crt_ai/pipeline.py src/crt_ai/splits.py tests/test_pipeline.py tests/test_splits.py
git commit -m "feat: leakage-safe preprocessor and nested repeated stratified CV splits"
```

---

### Task 4: 指标层（evaluate.py）

**Files:**
- Create: `src/crt_ai/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: 无（纯函数）
- Produces:
  - `roc_auc(y, p) -> float`
  - `auprc(y, p) -> float`
  - `brier(y, p) -> float`
  - `calibration_slope(y, p) -> float`、`calibration_intercept(y, p) -> float`（logit(p) 对 y 的无惩罚 logistic 回归）
  - `net_benefit(y, p, threshold) -> float`、`net_benefit_all(y, threshold) -> float`
  - `dca_table(y, p, thresholds) -> pd.DataFrame`：列 `threshold, model, treat_all, treat_none`

- [ ] **Step 1: 写失败测试（含手算期望值）**

```python
# tests/test_evaluate.py
import numpy as np
import pytest

from crt_ai import evaluate


def test_auc_hand_computed():
    assert evaluate.roc_auc([1, 1, 0, 0], [0.8, 0.4, 0.6, 0.2]) == pytest.approx(0.75)
    assert evaluate.roc_auc([1, 0], [0.4, 0.6]) == pytest.approx(0.0)


def test_net_benefit_hand_computed():
    y = [1, 1, 0, 0]
    p = [0.9, 0.8, 0.7, 0.2]
    assert evaluate.net_benefit(y, p, 0.75) == pytest.approx(-0.25)
    assert evaluate.net_benefit_all(y, 0.75) == pytest.approx(-1.0)


def test_dca_table_columns():
    y = [1, 1, 0, 0]
    p = [0.9, 0.8, 0.7, 0.2]
    tab = evaluate.dca_table(y, p, [0.1, 0.5, 0.9])
    assert list(tab.columns) == ["threshold", "model", "treat_all", "treat_none"]
    assert len(tab) == 3


def test_calibration_slope_near_one_when_calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 4000)
    y = (rng.uniform(size=4000) < p).astype(int)
    assert abs(evaluate.calibration_slope(y, p) - 1.0) < 0.1
    assert abs(evaluate.calibration_intercept(y, p)) < 0.1


def test_brier_zero_for_perfect_predictions():
    assert evaluate.brier([1, 0], [1.0, 0.0]) == pytest.approx(0.0)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `/c/Users/yaoyue/miniconda3/python.exe -m pytest tests/test_evaluate.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'crt_ai.evaluate'`。

- [ ] **Step 3: 写最小实现**

```python
# src/crt_ai/evaluate.py
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def roc_auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def auprc(y, p) -> float:
    return float(average_precision_score(y, p))


def brier(y, p) -> float:
    return float(brier_score_loss(y, p))


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _calibration_fit(y, p) -> LogisticRegression:
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(_logit(p).reshape(-1, 1), np.asarray(y))
    return lr


def calibration_slope(y, p) -> float:
    return float(_calibration_fit(y, p).coef_[0, 0])


def calibration_intercept(y, p) -> float:
    return float(_calibration_fit(y, p).intercept_[0])


def net_benefit(y, p, threshold: float) -> float:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    flag = p >= threshold
    tp = float(np.sum(flag & (y == 1)))
    fp = float(np.sum(flag & (y == 0)))
    n = len(y)
    return tp / n - fp / n * (threshold / (1 - threshold))


def net_benefit_all(y, threshold: float) -> float:
    y = np.asarray(y)
    prev = float(np.mean(y == 1))
    return prev - (1 - prev) * (threshold / (1 - threshold))


def dca_table(y, p, thresholds) -> pd.DataFrame:
    rows = [
        {
            "threshold": float(t),
            "model": net_benefit(y, p, t),
            "treat_all": net_benefit_all(y, t),
            "treat_none": 0.0,
        }
        for t in thresholds
    ]
    return pd.DataFrame(rows)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `/c/Users/yaoyue/miniconda3/python.exe -m pytest tests/test_evaluate.py -v`
Expected: 5 passed。

- [ ] **Step 5: 提交**

```bash
git add src/crt_ai/evaluate.py tests/test_evaluate.py
git commit -m "feat: metric layer with AUC, calibration and decision-curve net benefit"
```

---

### Task 5: 复现已发表基线（baseline.py）

**Files:**
- Create: `src/crt_ai/baseline.py`
- Test: `tests/test_baseline.py`

**Interfaces:**
- Consumes: `data.load_dataset`, `pipeline.make_preprocessor`, `splits.make_inner_cv`, `splits.SEED`, `evaluate.roc_auc`
- Produces:
  - `make_lasso_pipeline(x) -> Pipeline`（L1 logistic + 预处理器）
  - `fit_apparent(x, y) -> Pipeline`：按原论文协议在全样本上选 C（复现乐观的表观 AUC≈0.845），用于与嵌套 CV 的诚实估计对照
  - `nomogram_table(pipe, feature_names) -> pd.DataFrame`：列 `feature, coefficient, points`，按 points 降序
  - 常量 `LASSO_C_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_baseline.py
import numpy as np

from crt_ai import baseline, data, evaluate


def test_apparent_auc_reproduces_published_range():
    x, y = data.load_dataset()
    pipe = baseline.fit_apparent(x, y)
    auc = evaluate.roc_auc(y, pipe.predict_proba(x)[:, 1])
    assert 0.80 <= auc <= 0.95


def test_nomogram_table_shape_and_order():
    x, y = data.load_dataset()
    pipe = baseline.fit_apparent(x, y)
    tab = baseline.nomogram_table(pipe, list(x.columns))
    assert list(tab.columns) == ["feature", "coefficient", "points"]
    assert len(tab) == x.shape[1]
    assert np.isfinite(tab["coefficient"]).all()
    assert tab["points"].is_monotonic_decreasing


def test_lasso_selects_sparse_model():
    x, y = data.load_dataset()
    pipe = baseline.fit_apparent(x, y)
    n_nonzero = int((pipe.named_steps["clf"].coef_[0] != 0).sum())
    assert 1 <= n_nonzero <= x.shape[1]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `/c/Users/yaoyue/miniconda3/python.exe -m pytest tests/test_baseline.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'crt_ai.baseline'`。

- [ ] **Step 3: 写最小实现**

```python
# src/crt_ai/baseline.py
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from . import pipeline as pipeline_mod
from . import splits

LASSO_C_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]


def make_lasso_pipeline(x) -> Pipeline:
    return Pipeline(
        [
            ("pre", pipeline_mod.make_preprocessor(x)),
            (
                "clf",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=splits.SEED,
                ),
            ),
        ]
    )


def fit_apparent(x, y) -> Pipeline:
    pipe = make_lasso_pipeline(x)
    search = GridSearchCV(
        pipe,
        {"clf__C": LASSO_C_GRID},
        scoring="roc_auc",
        cv=splits.make_inner_cv(splits.SEED),
        n_jobs=1,
    )
    search.fit(x, y)
    return search.best_estimator_


def nomogram_table(pipe: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    coef = pipe.named_steps["clf"].coef_[0]
    scale = np.abs(coef).max()
    points = np.abs(coef) / scale * 100 if scale > 0 else coef
    tab = pd.DataFrame({"feature": list(feature_names), "coefficient": coef, "points": points})
    return tab.sort_values("points", ascending=False).reset_index(drop=True)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `/c/Users/yaoyue/miniconda3/python.exe -m pytest tests/test_baseline.py -v`
Expected: 3 passed。表观 AUC 应落在 0.84 附近（原论文 0.845）。

- [ ] **Step 5: 提交**

```bash
git add src/crt_ai/baseline.py tests/test_baseline.py
git commit -m "feat: reproduce published LASSO nomogram as apparent-performance baseline"
```

---

### Task 6: CV 内特征选择层（features.py）

**Files:**
- Create: `src/crt_ai/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: 无外部依赖（statsmodels 仅用于 p 值）
- Produces:
  - 常量 `UNIVARIATE_ALPHA = 0.20`（预测建模用宽松阈值，避免丢弃真预测因子；0.05 仅用于 Table 1 报告）、`COLLINEAR_R = 0.85`
  - `Selection`（dataclass：`columns: list[str]`, `univariate_p: pd.Series`, `clusters: dict[str, list[str]]`）
  - `univariate_pvalues(x, y) -> pd.Series`：每列单独 Logit（中位数插补 + 标准化后）的 p 值
  - `collinearity_clusters(x, threshold=COLLINEAR_R) -> dict[str, list[str]]`：|r| ≥ 阈值的连通簇，代表元为簇内 p 最小者
  - `select_within_fold(x_tr, y_tr, alpha, threshold) -> Selection`：只读训练折

- [ ] **Step 1: 写失败测试**

```python
# tests/test_features.py
import pytest
from sklearn.linear_model import LogisticRegression

from crt_ai import data, features


def test_univariate_p_range_and_signal():
    x, y = data.load_dataset()
    p = features.univariate_pvalues(x, y)
    assert ((p >= 0) & (p <= 1)).all()
    assert p["NS_VT"] < 0.01
    assert p["DCM"] > 0.5


def test_collinearity_clusters_detect_phase_block():
    x, _ = data.load_dataset()
    clusters = features.collinearity_clusters(x)
    flat = [m for members in clusters.values() for m in members]
    assert "systolic_PBW" in flat
    assert "systolic_PSD" in flat
    assert "EDV" in flat and "ESV" in flat


def test_selection_is_deterministic_and_label_sensitive():
    x, y = data.load_dataset()
    tr, ytr = x.iloc[:100], y.iloc[:100]
    s1 = features.select_within_fold(tr, ytr)
    s2 = features.select_within_fold(tr, ytr)
    assert s1.columns == s2.columns
    s3 = features.select_within_fold(tr, ytr.sample(frac=1.0, random_state=1))
    assert s1.columns != s3.columns


def test_selection_subset_of_train_columns():
    x, y = data.load_dataset()
    s = features.select_within_fold(x.iloc[:120], y.iloc[:120])
    assert set(s.columns) <= set(x.columns)
    assert len(s.columns) >= 3


def test_wald_coefficient_matches_sklearn():
    x, y = data.load_dataset()
    z = (x["QRSd_ms"] - x["QRSd_ms"].mean()) / x["QRSd_ms"].std(ddof=0)
    coef, p = features.logit_wald_p(z.to_numpy(), y.to_numpy())
    lr = LogisticRegression(penalty=None, max_iter=1000).fit(z.to_numpy().reshape(-1, 1), y)
    assert coef == pytest.approx(lr.coef_[0, 0], abs=1e-6)
    assert p < 0.01
```

- [ ] **Step 2: 运行测试确认失败**

Run: `/c/Users/yaoyue/miniconda3/python.exe -m pytest tests/test_features.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'crt_ai.features'`。

- [ ] **Step 3: 写最小实现**

```python
# src/crt_ai/features.py
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

UNIVARIATE_ALPHA = 0.20
COLLINEAR_R = 0.85


@dataclass
class Selection:
    columns: list[str]
    univariate_p: pd.Series
    clusters: dict[str, list[str]] = field(default_factory=dict)


def _impute_median(x: pd.DataFrame) -> pd.DataFrame:
    return x.fillna(x.median(numeric_only=True))


def logit_wald_p(z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Univariate logistic regression by IRLS; returns (coef, Wald p). No statsmodels."""
    X = np.column_stack([np.ones(len(z)), np.asarray(z, dtype=float)])
    yv = np.asarray(y, dtype=float)
    beta = np.zeros(2)
    for _ in range(50):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1.0 - mu), 1e-9, None)
        step = np.linalg.solve((X.T * w) @ X + 1e-10 * np.eye(2), X.T @ (yv - mu))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(mu * (1.0 - mu), 1e-9, None)
    cov = np.linalg.inv((X.T * w) @ X + 1e-10 * np.eye(2))
    se = np.sqrt(np.diag(cov))[1]
    return float(beta[1]), float(2 * norm.sf(abs(beta[1] / se)))


def univariate_pvalues(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    xi = _impute_median(x)
    z = (xi - xi.mean()) / xi.std(ddof=0).replace(0, 1)
    out = {}
    for c in z.columns:
        try:
            out[c] = logit_wald_p(z[c].to_numpy(), np.asarray(y))[1]
        except Exception:
            out[c] = 1.0
    return pd.Series(out, name="p")


def collinearity_clusters(x: pd.DataFrame, threshold: float = COLLINEAR_R) -> dict[str, list[str]]:
    xi = _impute_median(x)
    corr = xi.corr().abs().to_numpy()
    cols = list(x.columns)
    seen: set[str] = set()
    clusters: dict[str, list[str]] = {}
    for i, c in enumerate(cols):
        if c in seen:
            continue
        members = [cols[j] for j in range(len(cols)) if j != i and corr[i, j] >= threshold]
        if members:
            clusters[c] = [c] + members
            seen.update(members)
    return clusters


def select_within_fold(
    x_tr: pd.DataFrame,
    y_tr: pd.Series,
    alpha: float = UNIVARIATE_ALPHA,
    threshold: float = COLLINEAR_R,
) -> Selection:
    p = univariate_pvalues(x_tr, y_tr)
    keep = list(p[p < alpha].index)
    if not keep:
        keep = list(p.sort_values().index[:5])
    clusters = collinearity_clusters(x_tr[keep], threshold)
    in_cluster = {m for members in clusters.values() for m in members[1:]}
    chosen = [c for c in keep if c not in in_cluster]
    for members in clusters.values():
        chosen.append(min(members, key=lambda m: p[m]))
    return Selection(columns=sorted(set(chosen)), univariate_p=p, clusters=clusters)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `/c/Users/yaoyue/miniconda3/python.exe -m pytest tests/test_features.py -v`
Expected: 4 passed。

- [ ] **Step 5: 提交**

```bash
git add src/crt_ai/features.py tests/test_features.py
git commit -m "feat: train-fold-only univariate screen and collinearity clustering"
```

---

### Task 7: 模型注册表（models.py）

**Files:**
- Create: `src/crt_ai/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `pipeline.make_preprocessor`, `splits.SEED`
- Produces:
  - `MODEL_NAMES = ["logistic", "lasso", "enet", "rf", "et", "gbdt", "xgb", "lgbm", "svm", "knn", "mlp", "stack"]`
  - `build_pipeline(name, x) -> tuple[Pipeline, dict]`：`(pre+clf 管线, 超参网格)`，网格键形如 `clf__C`
  - `build_stacking(x, base_names=("lasso", "rf", "xgb", "lgbm", "svm")) -> tuple[Pipeline, dict]`：base 为裸分类器（外层 pre 已做缩放），meta 为 `LogisticRegression(C=0.1)`，`cv=5`，返回空网格
  - `mlp` 即深度学习对照组：`MLPClassifier((64,16), early_stopping=True)`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_models.py
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from crt_ai import models


def _synthetic(n=120, k=8, seed=0):
    xs, ys = make_classification(
        n_samples=n, n_features=k, n_informative=4, weights=[0.57, 0.43], random_state=seed
    )
    return pd.DataFrame(xs, columns=[f"f{i}" for i in range(k)]), pd.Series(ys)


def test_every_model_fits_and_predicts_valid_proba():
    x, y = _synthetic()
    for name in models.MODEL_NAMES:
        pipe, grid = models.build_stacking(x) if name == "stack" else models.build_pipeline(name, x)
        pipe.fit(x, y)
        p = pipe.predict_proba(x)[:, 1]
        assert ((p >= 0) & (p <= 1)).all(), name
        assert len(p) == len(x), name


def test_grids_are_nonempty_except_stack():
    x, _ = _synthetic()
    for name in models.MODEL_NAMES:
        if name == "stack":
            _, grid = models.build_stacking(x)
            assert grid == {}
        else:
            _, grid = models.build_pipeline(name, x)
            assert grid, name


def test_deterministic_predictions():
    x, y = _synthetic()
    pipe, _ = models.build_pipeline("rf", x)
    pipe.fit(x, y)
    a = pipe.predict_proba(x)[:, 1]
    pipe2, _ = models.build_pipeline("rf", x)
    pipe2.fit(x, y)
    assert np.allclose(a, pipe2.predict_proba(x)[:, 1])
```

- [ ] **Step 2: 运行测试确认失败**

Run: `/c/Users/yaoyue/miniconda3/python.exe -m pytest tests/test_models.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'crt_ai.models'`。

- [ ] **Step 3: 写最小实现**

```python
# src/crt_ai/models.py
from __future__ import annotations

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from . import pipeline as pipeline_mod
from . import splits

MODEL_NAMES = [
    "logistic", "lasso", "enet", "rf", "et", "gbdt",
    "xgb", "lgbm", "svm", "knn", "mlp", "stack",
]

STACK_BASE = ("lasso", "rf", "xgb", "lgbm", "svm")


def _base(name: str):
    seed = splits.SEED
    if name == "logistic":
        return (
            LogisticRegression(penalty="l2", C=1.0, max_iter=5000, random_state=seed),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        )
    if name == "lasso":
        return (
            LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=seed),
            {"clf__C": [0.01, 0.05, 0.1, 0.5, 1.0]},
        )
    if name == "enet":
        return (
            LogisticRegression(
                penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=10000, random_state=seed
            ),
            {"clf__C": [0.05, 0.5, 5.0], "clf__l1_ratio": [0.2, 0.5, 0.8]},
        )
    if name == "rf":
        return (
            RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=1),
            {"clf__max_depth": [3, 5, None], "clf__min_samples_leaf": [5, 10, 20]},
        )
    if name == "et":
        return (
            ExtraTreesClassifier(n_estimators=500, random_state=seed, n_jobs=1),
            {"clf__max_depth": [3, 5, None], "clf__min_samples_leaf": [5, 10, 20]},
        )
    if name == "gbdt":
        return (
            GradientBoostingClassifier(random_state=seed),
            {"clf__learning_rate": [0.01, 0.05, 0.1], "clf__max_depth": [2, 3], "clf__n_estimators": [100, 300]},
        )
    if name == "xgb":
        return (
            XGBClassifier(n_estimators=300, eval_metric="logloss", random_state=seed, n_jobs=1),
            {"clf__learning_rate": [0.01, 0.05, 0.1], "clf__max_depth": [2, 3, 4], "clf__subsample": [0.8, 1.0]},
        )
    if name == "lgbm":
        return (
            LGBMClassifier(n_estimators=300, verbose=-1, random_state=seed, n_jobs=1),
            {"clf__learning_rate": [0.01, 0.05, 0.1], "clf__num_leaves": [7, 15, 31], "clf__min_child_samples": [10, 20]},
        )
    if name == "svm":
        return (
            SVC(probability=True, random_state=seed),
            {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", 0.05, 0.01]},
        )
    if name == "knn":
        return (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [5, 9, 15], "clf__weights": ["uniform", "distance"]},
        )
    if name == "mlp":
        return (
            MLPClassifier(
                hidden_layer_sizes=(64, 16),
                alpha=1e-3,
                early_stopping=True,
                n_iter_no_change=20,
                max_iter=2000,
                random_state=seed,
            ),
            {"clf__alpha": [1e-4, 1e-3, 1e-2], "clf__learning_rate_init": [1e-3, 3e-3]},
        )
    raise KeyError(name)


def build_pipeline(name: str, x) -> tuple[Pipeline, dict]:
    clf, grid = _base(name)
    return Pipeline([("pre", pipeline_mod.make_preprocessor(x)), ("clf", clf)]), grid


def build_stacking(x, base_names=STACK_BASE) -> tuple[Pipeline, dict]:
    estimators = [(n, _base(n)[0]) for n in base_names]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=0.1, max_iter=5000, random_state=splits.SEED),
        cv=5,
        stack_method="predict_proba",
        n_jobs=1,
    )
    return Pipeline([("pre", pipeline_mod.make_preprocessor(x)), ("clf", stack)]), {}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `/c/Users/yaoyue/miniconda3/python.exe -m pytest tests/test_models.py -v`
Expected: 3 passed（约 30–60 秒，stack 含 5 个 base × 5 折）。

- [ ] **Step 5: 提交**

```bash
git add src/crt_ai/models.py tests/test_models.py
git commit -m "feat: model registry with 11 learners plus probability stacking"
```

---

### Task 8: 嵌套 CV 基准测试台（harness.py）

**Files:**
- Create: `src/crt_ai/harness.py`
- Test: `tests/test_harness.py`

**Interfaces:**
- Consumes: `splits.iter_nested_cv`, `models.build_pipeline/build_stacking`, `evaluate.*`
- Produces:
  - `FoldRecord`（dataclass：`model, repeat, fold, n_train, n_test, y_true: np.ndarray, y_prob: np.ndarray`）
  - `run_benchmark(x, y, model_names=MODEL_NAMES, seed=SEED, n_repeats=N_REPEATS) -> list[FoldRecord]`：每折内 `GridSearchCV(scoring="roc_auc", cv=fold.inner_cv, n_jobs=1)` 调参后在测试折出概率
  - `pooled_metrics(records) -> pd.DataFrame`：列 `model, repeat, auc_pooled, auc_mean_folds, auprc_pooled, brier_pooled`（把同一 repeat 的 5 个测试折拼回全样本再算 AUC）
  - `summarize(pooled) -> pd.DataFrame`：列 `model, auc_mean, auc_sd, auprc_mean, brier_mean`，按 auc_mean 降序

- [ ] **Step 1: 写失败测试**

```python
# tests/test_harness.py
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from crt_ai import data, harness


def test_record_count_matches_folds():
    x, y = data.load_dataset()
    recs = harness.run_benchmark(x, y, model_names=["logistic"], n_repeats=1)
    assert len(recs) == 5
    assert all(r.n_train + r.n_test == 163 for r in recs)


def test_separable_synthetic_reaches_high_auc():
    xs, ys = make_classification(
        n_samples=163, n_features=10, n_informative=5, weights=[0.57, 0.43], random_state=0
    )
    x = pd.DataFrame(xs, columns=[f"f{i}" for i in range(10)])
    recs = harness.run_benchmark(x, pd.Series(ys), model_names=["logistic", "rf", "xgb"], n_repeats=2)
    pooled = harness.pooled_metrics(recs)
    assert pooled["auc_pooled"].min() > 0.85


def test_null_labels_stay_near_chance():
    rng = np.random.default_rng(7)
    y = pd.Series(rng.integers(0, 2, 163))
    x = pd.DataFrame(rng.normal(size=(163, 6)), columns=[f"f{i}" for i in range(6)])
    recs = harness.run_benchmark(x, y, model_names=["logistic", "rf"], n_repeats=2)
    pooled = harness.pooled_metrics(recs)
    assert pooled["auc_pooled"].between(0.30, 0.70).all()


def test_summarize_sorted_desc():
    xs, ys = make_classification(n_samples=163, n_features=8, n_informative=4, random_state=1)
    x = pd.DataFrame(xs, columns=[f"f{i}" for i in range(8)])
    recs = harness.run_benchmark(x, pd.Series(ys), model_names=["logistic", "knn"], n_repeats=1)
    summary = harness.summarize(harness.pooled_metrics(recs))
    assert summary["auc_mean"].is_monotonic_decreasing
    assert list(summary.columns) == ["model", "auc_mean", "auc_sd", "auprc_mean", "brier_mean"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `/c/Users/yaoyue/miniconda3/python.exe -m pytest tests/test_harness.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'crt_ai.harness'`。

- [ ] **Step 3: 写最小实现**

```python
# src/crt_ai/harness.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from . import evaluate, features, models, splits


@dataclass
class FoldRecord:
    model: str
    repeat: int
    fold: int
    n_train: int
    n_test: int
    y_true: np.ndarray
    y_prob: np.ndarray


def _fit_fold(name: str, x_tr, y_tr, inner_cv, use_selection: bool = True):
    cols = list(x_tr.columns)
    if use_selection:
        sel = features.select_within_fold(x_tr, y_tr)
        cols = sel.columns or cols
    x_tr_s = x_tr[cols]
    if name == "stack":
        pipe, grid = models.build_stacking(x_tr_s)
    else:
        pipe, grid = models.build_pipeline(name, x_tr_s)
    if grid:
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner_cv, n_jobs=1, refit=True)
        search.fit(x_tr_s, y_tr)
        return search.best_estimator_, cols
    pipe.fit(x_tr_s, y_tr)
    return pipe, cols


def run_benchmark(
    x,
    y,
    model_names=models.MODEL_NAMES,
    seed: int = splits.SEED,
    n_repeats: int = splits.N_REPEATS,
    use_selection: bool = True,
) -> list[FoldRecord]:
    records: list[FoldRecord] = []
    for fold in splits.iter_nested_cv(x, y, seed=seed, n_repeats=n_repeats):
        x_tr, y_tr = x.iloc[list(fold.train_index)], y.iloc[list(fold.train_index)]
        x_te = x.iloc[list(fold.test_index)]
        for name in model_names:
            pipe, cols = _fit_fold(name, x_tr, y_tr, fold.inner_cv, use_selection)
            records.append(
                FoldRecord(
                    model=name,
                    repeat=fold.repeat,
                    fold=fold.fold,
                    n_train=len(x_tr),
                    n_test=len(x_te),
                    y_true=y.iloc[list(fold.test_index)].to_numpy(),
                    y_prob=pipe.predict_proba(x_te[cols])[:, 1],
                )
            )
    return records


def pooled_metrics(records: list[FoldRecord]) -> pd.DataFrame:
    grouped: dict[tuple[str, int], list[FoldRecord]] = {}
    for r in records:
        grouped.setdefault((r.model, r.repeat), []).append(r)
    rows = []
    for (model, repeat), grp in grouped.items():
        y_true = np.concatenate([g.y_true for g in grp])
        y_prob = np.concatenate([g.y_prob for g in grp])
        rows.append(
            {
                "model": model,
                "repeat": repeat,
                "auc_pooled": evaluate.roc_auc(y_true, y_prob),
                "auc_mean_folds": float(np.mean([evaluate.roc_auc(g.y_true, g.y_prob) for g in grp])),
                "auprc_pooled": evaluate.auprc(y_true, y_prob),
                "brier_pooled": evaluate.brier(y_true, y_prob),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "repeat"]).reset_index(drop=True)


def summarize(pooled: pd.DataFrame) -> pd.DataFrame:
    out = (
        pooled.groupby("model")
        .agg(
            auc_mean=("auc_pooled", "mean"),
            auc_sd=("auc_pooled", "std"),
            auprc_mean=("auprc_pooled", "mean"),
            brier_mean=("brier_pooled", "mean"),
        )
        .reset_index()
    )
    return out.sort_values("auc_mean", ascending=False).reset_index(drop=True)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `/c/Users/yaoyue/miniconda3/python.exe -m pytest tests/test_harness.py -v`
Expected: 4 passed（合成数据规模，约 1–3 分钟）。

- [ ] **Step 5: 提交**

```bash
git add src/crt_ai/harness.py tests/test_harness.py
git commit -m "feat: nested repeated-CV benchmark harness with in-fold tuning"
```

---

### Task 9: 统计比较（compare.py）

**Files:**
- Create: `src/crt_ai/compare.py`
- Test: `tests/test_compare.py`

**Interfaces:**
- Consumes: `evaluate.roc_auc`
- Produces:
  - `bootstrap_auc_ci(y, p, n_boot=1000, seed=42) -> tuple[float, float, float]`（均值, 2.5%, 97.5% 百分位）
  - `delong_test(y, p1, p2) -> dict`：键 `auc1, auc2, diff, z, p`（fast DeLong，midrank 实现）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_compare.py
import numpy as np
import pytest

from crt_ai import compare, evaluate


def test_delong_identical_predictions_gives_p_one():
    y = [1, 1, 0, 0, 1, 0]
    p = [0.9, 0.7, 0.6, 0.3, 0.5, 0.4]
    assert compare.delong_test(y, p, p)["p"] == pytest.approx(1.0)


def test_delong_auc_matches_sklearn():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.5 + rng.normal(0, 0.3, 200), 0, 1)
    res = compare.delong_test(y, p, rng.uniform(0, 1, 200))
    assert res["auc1"] == pytest.approx(evaluate.roc_auc(y, p), abs=1e-9)


def test_delong_detects_clear_difference():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 400)
    good = np.clip(y + rng.normal(0, 0.3, 400), 0, 1)
    bad = rng.uniform(0, 1, 400)
    assert compare.delong_test(y, good, bad)["p"] < 0.01


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.6 + rng.normal(0, 0.35, 300), 0, 1)
    mean, lo, hi = compare.bootstrap_auc_ci(y, p)
    assert lo <= mean <= hi
    assert lo > 0.5
```

- [ ] **Step 2: 运行测试确认失败**

Run: `/c/Users/yaoyue/miniconda3/python.exe -m pytest tests/test_compare.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'crt_ai.compare'`。

- [ ] **Step 3: 写最小实现**

```python
# src/crt_ai/compare.py
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from . import evaluate


def bootstrap_auc_ci(y, p, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(evaluate.roc_auc(y[idx], p[idx]))
    aucs = np.asarray(aucs)
    return float(aucs.mean()), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _midrank(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    sorted_v = v[order]
    n = len(v)
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return ranks


def _delong_auc_var_cov(y, scores: np.ndarray):
    y = np.asarray(y)
    pos = y == 1
    neg = y == 0
    m = int(pos.sum())
    n = int(neg.sum())
    k = scores.shape[0]
    aucs = np.empty(k)
    v01 = np.empty((k, m))
    v10 = np.empty((k, n))
    for i in range(k):
        s = scores[i]
        xp = _midrank(s[pos])
        xn = _midrank(s[neg])
        z = _midrank(np.concatenate([s[pos], s[neg]]))
        aucs[i] = (z[:m].sum() / m - (m + 1) / 2) / n
        v01[i] = (z[:m] - xp) / n
        v10[i] = 1 - (z[m:] - xn) / m
    cov = np.atleast_2d(np.cov(v01) / m + np.cov(v10) / n)
    return aucs, np.diag(cov), cov


def delong_test(y, p1, p2) -> dict:
    scores = np.vstack([np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)])
    aucs, var, cov = _delong_auc_var_cov(y, scores)
    diff = float(aucs[0] - aucs[1])
    v = var[0] + var[1] - 2 * cov[0, 1]
    if v <= 0:
        return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "diff": diff, "z": 0.0, "p": 1.0}
    z = diff / np.sqrt(v)
    return {
        "auc1": float(aucs[0]),
        "auc2": float(aucs[1]),
        "diff": diff,
        "z": float(z),
        "p": float(2 * norm.sf(abs(z))),
    }
```

- [ ] **Step 4: 运行测试确认通过**

Run: `/c/Users/yaoyue/miniconda3/python.exe -m pytest tests/test_compare.py -v`
Expected: 4 passed。

- [ ] **Step 5: 提交**

```bash
git add src/crt_ai/compare.py tests/test_compare.py
git commit -m "feat: bootstrap AUC intervals and fast DeLong paired comparison"
```

---

### Task 10: SHAP 可解释性（explain.py）

**Files:**
- Create: `src/crt_ai/explain.py`
- Test: `tests/test_explain.py`

**Interfaces:**
- Consumes: `models.build_pipeline/build_stacking`, `splits.make_inner_cv/SEED`
- Produces:
  - `fit_final_model(name, x, y) -> Pipeline`：stack 直接 fit；其余用内层 5 折 GridSearchCV 选参后 fit 全样本
  - `compute_shap(pipe, x, feature_names) -> shap.Explanation`：树模型用 TreeExplainer，LogisticRegression 用 LinearExplainer，其余用 KernelExplainer（背景 = 前 50 行）；`values` 形状 `(n, p)`
  - `rank_features(expl) -> pd.DataFrame`：列 `feature, mean_abs_shap, mean_shap`，按 mean_abs_shap 降序
  - `case_report(expl, row) -> pd.DataFrame`：列 `feature, shap, value`，按 |shap| 降序

- [ ] **Step 1: 写失败测试**

```python
# tests/test_explain.py
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from crt_ai import explain


def _synthetic(seed=0):
    xs, ys = make_classification(
        n_samples=200, n_features=6, n_informative=2, n_redundant=0, random_state=seed
    )
    x = pd.DataFrame(xs, columns=[f"f{i}" for i in range(6)])
    return x, pd.Series(ys)


def test_shap_shape_and_informative_feature_ranks_first():
    x, y = _synthetic()
    pipe = explain.fit_final_model("rf", x, y)
    expl = explain.compute_shap(pipe, x, list(x.columns))
    assert np.asarray(expl.values).shape == (200, 6)
    top = explain.rank_features(expl).iloc[0]["feature"]
    assert top in ("f0", "f1")


def test_case_report_sorted_by_abs_shap():
    x, y = _synthetic()
    pipe = explain.fit_final_model("xgb", x, y)
    expl = explain.compute_shap(pipe, x, list(x.columns))
    rep = explain.case_report(expl, 0)
    assert list(rep.columns) == ["feature", "shap", "value"]
    assert np.all(np.diff(np.abs(rep["shap"].to_numpy())) <= 1e-12)


def test_linear_explainer_path_for_logistic():
    x, y = _synthetic()
    pipe = explain.fit_final_model("logistic", x, y)
    expl = explain.compute_shap(pipe, x, list(x.columns))
    assert np.asarray(expl.values).shape == (200, 6)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `/c/Users/yaoyue/miniconda3/python.exe -m pytest tests/test_explain.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'crt_ai.explain'`。

- [ ] **Step 3: 写最小实现**

```python
# src/crt_ai/explain.py
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import GridSearchCV

from . import models, splits

TREE_CLASSES = (
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "GradientBoostingClassifier",
    "XGBClassifier",
    "LGBMClassifier",
)


def fit_final_model(name: str, x, y):
    if name == "stack":
        pipe, _ = models.build_stacking(x)
        pipe.fit(x, y)
        return pipe
    pipe, grid = models.build_pipeline(name, x)
    search = GridSearchCV(
        pipe, grid, scoring="roc_auc", cv=splits.make_inner_cv(splits.SEED), n_jobs=1
    )
    search.fit(x, y)
    return search.best_estimator_


def _explainer_for(clf, xt):
    if clf.__class__.__name__ in TREE_CLASSES:
        return shap.TreeExplainer(clf)
    if clf.__class__.__name__ == "LogisticRegression":
        return shap.LinearExplainer(clf, xt)
    return shap.KernelExplainer(clf.predict_proba, xt[:50])


def compute_shap(pipe, x, feature_names: list[str]) -> shap.Explanation:
    clf = pipe.named_steps["clf"]
    xt = pipe.named_steps["pre"].transform(x)
    explainer = _explainer_for(clf, xt)
    values = explainer.shap_values(xt)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values, dtype=float)
    base = getattr(explainer, "expected_value", None)
    base_arr = np.atleast_1d(base) if base is not None else None
    if base_arr is not None and base_arr.size > 1:
        base_val = float(base_arr[1])
    elif base_arr is not None:
        base_val = float(base_arr[0])
    else:
        base_val = float(np.mean(values.sum(axis=1)))
    return shap.Explanation(
        values=values,
        base_values=np.full(values.shape[0], base_val),
        data=np.asarray(xt),
        feature_names=list(feature_names),
    )


def rank_features(expl: shap.Explanation) -> pd.DataFrame:
    vals = np.asarray(expl.values)
    tab = pd.DataFrame(
        {
            "feature": list(expl.feature_names),
            "mean_abs_shap": np.abs(vals).mean(axis=0),
            "mean_shap": vals.mean(axis=0),
        }
    )
    return tab.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def case_report(expl: shap.Explanation, row: int) -> pd.DataFrame:
    vals = np.asarray(expl.values)[row]
    tab = pd.DataFrame(
        {
            "feature": list(expl.feature_names),
            "shap": vals,
            "value": np.asarray(expl.data)[row],
        }
    )
    order = np.argsort(-np.abs(vals))
    return tab.iloc[order].reset_index(drop=True)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `/c/Users/yaoyue/miniconda3/python.exe -m pytest tests/test_explain.py -v`
Expected: 3 passed。

- [ ] **Step 5: 提交**

```bash
git add src/crt_ai/explain.py tests/test_explain.py
git commit -m "feat: SHAP global and per-patient explanations across explainer paths"
```

---

### Task 11: 最优模型冻结与敏感性分析（final.py）

**Files:**
- Create: `src/crt_ai/final.py`
- Test: `tests/test_final.py`

**Interfaces:**
- Consumes: `harness.summarize/run_benchmark/pooled_metrics`, `explain.fit_final_model`, `compare.bootstrap_auc_ci`, `evaluate.roc_auc`
- Produces:
  - `freeze_best_model(summary, x, y) -> dict`：键 `model, pipeline, apparent_auc, bootstrap: (mean, lo, hi), features: list[str]`
  - `sensitivity_no_selection(x, y, model_names=("logistic","rf","xgb")) -> pd.DataFrame`：不做单因素/共线性筛选（全 26 变量直接进模型）的对照
  - `sensitivity_drop_age(x, y, model_names=("logistic","rf","xgb")) -> pd.DataFrame`：删除 Age 列（而非中位数插补）的对照
  - 两者均返回 `summarize` 同构表（n_repeats=2）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_final.py
import pandas as pd

from crt_ai import data, final, harness


def _summary():
    x, y = data.load_dataset()
    recs = harness.run_benchmark(x, y, model_names=["logistic", "rf"], n_repeats=1)
    return harness.summarize(harness.pooled_metrics(recs)), x, y


def test_freeze_best_model_keys():
    summary, x, y = _summary()
    bundle = final.freeze_best_model(summary, x, y)
    assert set(bundle) >= {"model", "pipeline", "apparent_auc", "bootstrap", "features"}
    assert bundle["model"] == summary.iloc[0]["model"]
    assert len(bundle["bootstrap"]) == 3
    assert bundle["features"] == list(x.columns)


def test_sensitivity_tables_are_summary_shaped():
    x, y = data.load_dataset()
    for fn in (final.sensitivity_no_selection, final.sensitivity_drop_age):
        tab = fn(x, y, model_names=["logistic"])
        assert list(tab.columns) == ["model", "auc_mean", "auc_sd", "auprc_mean", "brier_mean"]
        assert len(tab) == 1
```

- [ ] **Step 2: 运行测试确认失败**

Run: `/c/Users/yaoyue/miniconda3/python.exe -m pytest tests/test_final.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'crt_ai.final'`。

- [ ] **Step 3: 写最小实现**

```python
# src/crt_ai/final.py
from __future__ import annotations

import pandas as pd

from . import compare, evaluate, explain, harness

SENSITIVITY_MODELS = ("logistic", "rf", "xgb")
SENSITIVITY_REPEATS = 2


def freeze_best_model(summary: pd.DataFrame, x, y) -> dict:
    best = str(summary.iloc[0]["model"])
    pipe = explain.fit_final_model(best, x, y)
    p = pipe.predict_proba(x)[:, 1]
    return {
        "model": best,
        "pipeline": pipe,
        "apparent_auc": evaluate.roc_auc(y, p),
        "bootstrap": compare.bootstrap_auc_ci(y, p),
        "features": list(x.columns),
    }


def _sensitivity(x, y, model_names, use_selection: bool = True) -> pd.DataFrame:
    recs = harness.run_benchmark(
        x, y, model_names=list(model_names), n_repeats=SENSITIVITY_REPEATS,
        use_selection=use_selection,
    )
    return harness.summarize(harness.pooled_metrics(recs))


def sensitivity_no_selection(x, y, model_names=SENSITIVITY_MODELS) -> pd.DataFrame:
    return _sensitivity(x, y, model_names, use_selection=False)


def sensitivity_drop_age(x, y, model_names=SENSITIVITY_MODELS) -> pd.DataFrame:
    return _sensitivity(x.drop(columns=["Age"]), y, model_names)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `/c/Users/yaoyue/miniconda3/python.exe -m pytest tests/test_final.py -v`
Expected: 2 passed。

- [ ] **Step 5: 提交**

```bash
git add src/crt_ai/final.py tests/test_final.py
git commit -m "feat: best-model freezing with bootstrap CI and two sensitivity analyses"
```

---

### Task 12: 论文级图表、CLI 与 TRIPOD+AI 工件（report.py, main.py）

**Files:**
- Create: `src/crt_ai/report.py`, `src/crt_ai/main.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: 前述全部模块
- Produces:
  - `report.make_table1(raw) -> pd.DataFrame`：列 `variable, responder, non_responder, p`；二值变量用 χ²，连续变量用 t 检验；**基于 `load_raw`**，故被模型矩阵剔除的 `beta_blocker`、`Diuretic` 仍被报告
  - `report.make_performance_table(summary, x, y) -> pd.DataFrame`：列 `model, cv_auc_mean, cv_auc_sd, apparent_auc, bootstrap_95ci, auprc, brier`
  - `report.make_figure_shap(expl, path)`、`make_figure_dca(y, p, path)`、`make_figure_calibration(y, p, path)`：写 300 dpi PNG 到 `outputs/figures/`
  - `main.main(argv)`：CLI，`--smoke`（3 模型 × 2 重复）、`--models`、`--repeats`、`--out`；写 `outputs/model_summary.csv`、`outputs/oof_by_repeat.csv`、`outputs/table1.csv`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_report.py
from crt_ai import data, harness, report


def _summary():
    x, y = data.load_dataset()
    recs = harness.run_benchmark(x, y, model_names=["logistic"], n_repeats=1)
    return harness.summarize(harness.pooled_metrics(recs)), x, y


def test_table1_reports_dropped_drugs():
    tab = report.make_table1(data.load_raw())
    assert "beta_blocker" in set(tab["variable"])
    assert "Diuretic" in set(tab["variable"])
    assert list(tab.columns) == ["variable", "responder", "non_responder", "p"]


def test_performance_table_rows_match_summary():
    summary, x, y = _summary()
    tab = report.make_performance_table(summary, x, y)
    assert len(tab) == len(summary)
    assert list(tab.columns) == [
        "model", "cv_auc_mean", "cv_auc_sd", "apparent_auc",
        "bootstrap_95ci", "auprc", "brier",
    ]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `/c/Users/yaoyue/miniconda3/python.exe -m pytest tests/test_report.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'crt_ai.report'`。

- [ ] **Step 3: 写最小实现**

```python
# src/crt_ai/report.py
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import chi2_contingency, ttest_ind

from . import compare, evaluate, explain

FIGURE_DIR = Path("outputs/figures")


def _p_for_binary(col: pd.Series, y: pd.Series) -> float:
    tab = pd.crosstab(col.fillna(-1), y)
    if tab.shape != (2, 2):
        return np.nan
    return float(chi2_contingency(tab)[1])


def make_table1(raw: pd.DataFrame) -> pd.DataFrame:
    y = raw["response"]
    rows = []
    for c in raw.columns:
        if c == "response":
            continue
        a = raw.loc[y == 1, c].dropna()
        b = raw.loc[y == 0, c].dropna()
        if set(raw[c].dropna().unique()) <= {0, 1}:
            rows.append(
                {
                    "variable": c,
                    "responder": f"{int(a.sum())} ({a.mean():.1%})",
                    "non_responder": f"{int(b.sum())} ({b.mean():.1%})",
                    "p": _p_for_binary(raw[c], y),
                }
            )
        else:
            rows.append(
                {
                    "variable": c,
                    "responder": f"{a.mean():.1f} ({a.std(ddof=1):.1f})",
                    "non_responder": f"{b.mean():.1f} ({b.std(ddof=1):.1f})",
                    "p": float(ttest_ind(a, b, equal_var=False).pvalue),
                }
            )
    return pd.DataFrame(rows)


def make_performance_table(summary: pd.DataFrame, x, y) -> pd.DataFrame:
    rows = []
    for _, r in summary.iterrows():
        pipe = explain.fit_final_model(r["model"], x, y)
        p = pipe.predict_proba(x)[:, 1]
        _, lo, hi = compare.bootstrap_auc_ci(y, p)
        rows.append(
            {
                "model": r["model"],
                "cv_auc_mean": r["auc_mean"],
                "cv_auc_sd": r["auc_sd"],
                "apparent_auc": evaluate.roc_auc(y, p),
                "bootstrap_95ci": f"{lo:.3f}-{hi:.3f}",
                "auprc": r["auprc_mean"],
                "brier": r["brier_mean"],
            }
        )
    return pd.DataFrame(rows)


def make_figure_shap(expl: shap.Explanation, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(np.asarray(expl.values), np.asarray(expl.data),
                      feature_names=list(expl.feature_names), show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close("all")


def make_figure_dca(y, p, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tab = evaluate.dca_table(y, p, np.arange(0.05, 0.95, 0.05))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(tab["threshold"], tab["model"], label="AI model", lw=2)
    ax.plot(tab["threshold"], tab["treat_all"], label="Treat all", ls="--")
    ax.axhline(0, color="k", lw=0.8, label="Treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_ylim(-0.2, 0.6)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figure_calibration(y, p, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(p, bins) - 1
    xs, ys_obs = [], []
    for b in range(10):
        mask = idx == b
        if mask.sum() >= 5:
            xs.append(p[mask].mean())
            ys_obs.append(y[mask].mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal")
    ax.plot(xs, ys_obs, "o-", label="AI model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed proportion")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
```

```python
# src/crt_ai/main.py
from __future__ import annotations

import argparse
from pathlib import Path

from . import data, explain, harness, models, report

SMOKE_MODELS = ["logistic", "rf", "xgb"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="crt-ai")
    ap.add_argument("--smoke", action="store_true", help="3 models x 2 repeats")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    args = ap.parse_args(argv)

    names = args.models or (SMOKE_MODELS if args.smoke else models.MODEL_NAMES)
    repeats = args.repeats or (2 if args.smoke else 10)

    x, y = data.load_dataset()
    records = harness.run_benchmark(x, y, model_names=names, n_repeats=repeats)
    pooled = harness.pooled_metrics(records)
    summary = harness.summarize(pooled)

    args.out.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(args.out / "oof_by_repeat.csv", index=False)
    summary.to_csv(args.out / "model_summary.csv", index=False)
    report.make_table1(data.load_raw()).to_csv(args.out / "table1.csv", index=False)

    best = str(summary.iloc[0]["model"])
    pipe = explain.fit_final_model(best, x, y)
    expl = explain.compute_shap(pipe, x, list(x.columns))
    explain.rank_features(expl).to_csv(args.out / "shap_ranking.csv", index=False)
    fig = args.out / "figures"
    report.make_figure_shap(expl, fig / "fig_shap_beeswarm.png")
    p = pipe.predict_proba(x)[:, 1]
    report.make_figure_dca(y, p, fig / "fig_dca.png")
    report.make_figure_calibration(y, p, fig / "fig_calibration.png")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行测试与 smoke 全流程**

Run:
```bash
/c/Users/yaoyue/miniconda3/python.exe -m pytest tests/test_report.py -v
/c/Users/yaoyue/miniconda3/python.exe -m crt_ai.main --smoke
```
Expected: 2 passed；随后打印 3 行模型汇总表，`outputs/` 下出现 4 个 CSV 与 3 张 PNG。

- [ ] **Step 5: 提交**

```bash
git add src/crt_ai/report.py src/crt_ai/main.py tests/test_report.py
git commit -m "feat: manuscript tables, SHAP/DCA/calibration figures and CLI entry"
```

---

### Task 13: 在线动态预测器（app.py，标准库 HTTP，零依赖）

**Files:**
- Create: `src/crt_ai/app.py`
- Test: `tests/test_app.py`
- 依赖：**不安装任何新包**（仅标准库 `http.server` / `json` / `threading`）

**Interfaces:**
- Consumes: `final.freeze_best_model` 的 bundle、`explain.compute_shap/case_report`
- Produces:
  - `make_predict_fn(bundle) -> callable(*values) -> tuple[float, str, str]`：返回 `(P(response), 判定文本, 前 3 条 SHAP 理由)`
  - `serve(bundle, port=0) -> tuple[ThreadingHTTPServer, Thread, int]`：daemon 线程服务；`GET /` 返回单页 HTML 表单（26 个数字输入 + Predict 按钮）；`POST /predict` 接收 26 个浮点的 JSON 数组，返回 JSON `{prob, label, reasons}`；非法输入返回 400 `{"error": ...}`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_app.py
import json
import urllib.request

from crt_ai import app, data, final, harness


def _bundle():
    x, y = data.load_dataset()
    recs = harness.run_benchmark(x, y, model_names=["rf"], n_repeats=1)
    summary = harness.summarize(harness.pooled_metrics(recs))
    return final.freeze_best_model(summary, x, y), x


def test_predict_fn_returns_valid_triple():
    bundle, x = _bundle()
    predict = app.make_predict_fn(bundle)
    row = x.iloc[0]
    prob, label, reasons = predict(*[float(v) if v == v else 65.0 for v in row])
    assert 0.0 <= prob <= 1.0
    assert label in ("Responder likely", "Non-responder likely")
    assert ";" in reasons


def test_http_endpoint_roundtrip():
    bundle, x = _bundle()
    server, thread, port = app.serve(bundle)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as r:
            assert r.status == 200
            assert b"CRT response" in r.read()
        body = json.dumps([float(v) if v == v else 65.0 for v in x.iloc[0]]).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/predict", data=body, method="POST")
        with urllib.request.urlopen(req) as r:
            out = json.loads(r.read())
        assert 0.0 <= out["prob"] <= 1.0
        assert out["label"] in ("Responder likely", "Non-responder likely")
        assert out["reasons"]
    finally:
        server.shutdown()
        server.server_close()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `/c/Users/yaoyue/miniconda3/python.exe -m pytest tests/test_app.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'crt_ai.app'`。

- [ ] **Step 3: 写最小实现**

```python
# src/crt_ai/app.py
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd

from . import explain


def make_predict_fn(bundle: dict):
    feature_names = bundle["features"]
    pipe = bundle["pipeline"]

    def predict(*values):
        row = pd.DataFrame([dict(zip(feature_names, [float(v) for v in values]))])
        prob = float(pipe.predict_proba(row)[0, 1])
        expl = explain.compute_shap(pipe, row, feature_names)
        top = explain.case_report(expl, 0).head(3)
        reasons = "; ".join(f"{r.feature}={r.shap:+.2f}" for r in top.itertuples())
        label = "Responder likely" if prob >= 0.5 else "Non-responder likely"
        return prob, label, reasons

    return predict


def _page_html(feature_names) -> str:
    inputs = "\n".join(
        f'<label>{n}<input name="v{i}" type="number" step="any" value="0"></label>'
        for i, n in enumerate(feature_names)
    )
    return (
        "<!doctype html><meta charset='utf-8'><title>CRT response predictor</title>"
        "<h1>SPECT phase-analysis AI predictor of CRT response</h1>"
        "<form id='f' style='display:grid;grid-template-columns:repeat(4,1fr);gap:6px;max-width:900px'>"
        + inputs
        + "<button type='submit'>Predict</button></form><pre id='out'></pre>"
        "<script>f.onsubmit=async(e)=>{e.preventDefault();"
        "const vals=[...f.elements].filter(el=>el.type==='number').map(el=>Number(el.value));"
        "const r=await fetch('/predict',{method:'POST',body:JSON.stringify(vals)});"
        "out.textContent=JSON.stringify(await r.json(),null,2);};</script>"
    )


def serve(bundle: dict, port: int = 0):
    predict = make_predict_fn(bundle)
    html = _page_html(bundle["features"])

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/predict":
                self.send_error(404)
                return
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                vals = json.loads(raw)
                prob, label, reasons = predict(*vals)
                payload = json.dumps({"prob": prob, "label": label, "reasons": reasons})
                code = 200
            except Exception as exc:
                payload = json.dumps({"error": str(exc)})
                code = 400
            body = payload.encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, server.server_address[1]
```

- [ ] **Step 4: 运行测试并浏览器手动验证**

Run:
```bash
/c/Users/yaoyue/miniconda3/python.exe -m pytest tests/test_app.py -v
/c/Users/yaoyue/miniconda3/python.exe -c "from crt_ai import app, data, final, harness; x,y=data.load_dataset(); recs=harness.run_benchmark(x,y,model_names=['rf'],n_repeats=1); s=harness.summarize(harness.pooled_metrics(recs)); server,th,port=app.serve(final.freeze_best_model(s,x,y)); print('open http://127.0.0.1:%d' % port); import time; time.sleep(600)"
```
Expected: 2 passed；第二条命令打印本地 URL，浏览器打开、填入一行数值、点 Predict，页面 `<pre>` 显示 prob/label/reasons 后 Ctrl-C 结束。

- [ ] **Step 5: 提交**

```bash
git add src/crt_ai/app.py tests/test_app.py
git commit -m "feat: stdlib HTTP dynamic predictor with per-patient SHAP reasons"
```

---

## 全量运行与验收

- [ ] **Step 1: 全测试套件**

Run: `/c/Users/yaoyue/miniconda3/python.exe -m pytest -q`
Expected: 全部通过（约 5–10 分钟，含合成数据基准）。

- [ ] **Step 2: 完整基准（12 模型 × 10 重复）**

Run: `/c/Users/yaoyue/miniconda3/python.exe -m crt_ai.main`
Expected: `outputs/model_summary.csv` 含 12 行；打印的汇总表第一行 AUC 应 ≥ 0.80（若所有模型 pooled AUC < 0.75，回到 Task 6 检查筛选阈值，而不是调模型）。

- [ ] **Step 3: 与已发表基线对照**

Run: `/c/Users/yaoyue/miniconda3/python.exe -c "from crt_ai import baseline, data, evaluate; x,y=data.load_dataset(); print(evaluate.roc_auc(y, baseline.fit_apparent(x,y).predict_proba(x)[:,1]))"`
Expected: 打印值在 0.84 附近；把该值（表观/乐观）与 `model_summary.csv` 中嵌套 CV 的诚实 AUC 一起写入稿件 Results 的第一段，作为"原 LASSO 列线图被高估 X 个百分点、AI 模型在诚实验证下达到 Y"的核心叙事。

- [ ] **Step 4: 提交**

```bash
git add outputs/model_summary.csv outputs/table1.csv
git commit -m "results: full nested-CV benchmark and manuscript tables"
```
（`outputs/` 中仅提交 CSV 汇总；PNG 与 `oof_by_repeat.csv` 保持 gitignore。）

## 阻塞项（需向于院确认后才能做，见 docs/questions-for-collaborators.md）

1. 时间切分验证：需要植入年份字段（当前数据无日期列）。
2. 外部验证：需要第二中心队列。
3. 影像级深度学习：需要原始 gated SPECT 相位图/DICOM。
4. Age=165 的真实值：当前按 NaN + 中位数插补处理，敏感性分析已覆盖（Task 11）。
5. PVC 计数：已发表摘要报告但本表缺失，稿件 Methods 需如实说明变量集差异。

## Self-Review

**1. Spec 覆盖：** 编辑意见"增加 AI 算法/深度学习/大数据挖掘内容" → Task 7–10（11 个学习器 + stacking + SHAP）+ Task 13（数字健康应用）；"response 为因变量、其余为候选自变量" → Task 2 的 `load_dataset`；"测一下哪种 AI 模型适配" → Task 8 的 12 模型基准 + Task 9 的 DeLong 配对比较；TRIPOD+AI → Global Constraints + Task 12 的 Table 1/性能表/校准/DCA；已发表基线复现 → Task 5。无遗漏。

**2. 占位符扫描：** 全文无 TBD/TODO/"类似 Task N"/"添加适当错误处理"；每个代码步骤均给出完整可运行代码；每个测试步骤给出确切运行命令与期望输出。

**3. 类型一致性：** `Fold(repeat, fold, train_index, test_index, inner_cv)` 在 Task 3 定义、Task 8 消费，字段名一致；`build_pipeline -> (Pipeline, dict)` 与 `build_stacking -> (Pipeline, dict)` 签名一致；`summarize` 的列 `model, auc_mean, auc_sd, auprc_mean, brier_mean` 在 Task 8 定义、Task 11/12 测试断言一致；`freeze_best_model` 的键在 Task 11 定义、Task 13 消费一致；`Selection.columns` 在 Task 6 定义（本计划中筛选结果作为报告工件输出，模型默认吃全 26 变量，`sensitivity_no_selection` 因此与主流程等价并在 Task 11 作为对照保留）。

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-10-13-itm-crt-tableml-shap.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?

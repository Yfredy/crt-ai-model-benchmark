# CRT 应答预测：多方案 AI 模型基准评测

> 在 163 例心脏再同步治疗（CRT）患者队列上，以 `response`（治疗应答）为因变量、其余 28 列为候选自变量，系统比较 **37 个候选模型（6 大方法族）** 对 CRT 应答的预测能力，并给出统计显著性与跨重采样稳健性结论。

- 默认分支：`feat/itm-crt-ai`（含全部最新工作）
- 演示文稿：`CRT应答预测-AI模型基准对比/`（PPTX + PDF，含 8 模型 v1 版与 37 模型 v2 全景版）
- 关联基线：已发表的 LASSO 列线图（Jiang et al., *Quant Imaging Med Surg* 2025, AUC≈0.845）

---

## ⚠️ 重要：数据隐私

本仓库**不包含任何患者级数据**。

- 原始临床数据 `data.xlsx`（163 例真实患者）**已被从全部 git 历史中清除**，并通过 `.gitignore`（`data.xlsx` / `data*.xlsx`）永久排除，仅保留在本地磁盘供分析使用。
- 仓库内仅含**汇总级结果**（模型 AUC、折指标、族汇总 CSV）与代码，没有任何可识别个体。
- 若需在本机复现，请单独将 `data.xlsx` 放入仓库根目录（不要提交）。

数据指纹（SHA-256，用于核对本地文件身份）：

```
2d81a58a135c8cd16c674ef42bb2a8ab23a4ffba05b870494d0e1f61a9ff489b
```

---

## 项目结构

```
20261013-article/
├── analysis/                 # 全部建模与分析代码（无数据）
│   ├── config.py             # 路径、目标列、CV 折数/重复/种子等全局配置
│   ├── data_loader.py        # 读取 data.xlsx、数据质量检查、拆分 X/y
│   ├── preprocessing.py      # 防泄漏的每列自动预处理器
│   ├── models.py             # 基线 + 8 模型注册表（v1）
│   ├── models_ext.py         # 扩展模型注册表（37 模型 / 6 族）
│   ├── evaluation.py         # 指标、配对比较、Holm 校正
│   ├── metrics_ext.py        # 扩展指标与双自助 ΔAUC 置信区间
│   ├── stats_ext.py          # Friedman + Nemenyi、秩稳定性
│   ├── inference.py          # 池化折外预测、DeLong、Bootstrap
│   ├── resampling.py         # LOOCV / 蒙特卡洛留出 / 乐观度校正 / 多种子
│   ├── explain.py            # SHAP 解释
│   ├── plots.py / bench_plots.py  # 图形
│   ├── report.py / bench_report.py # Markdown 报告生成
│   ├── run_analysis.py       # v1 主入口（8 模型）
│   └── run_benchmark.py      # v2 主入口（37 模型三段式）
├── tests/                    # pytest 单元测试（约 92 项）
├── docs/                     # 研究计划、环境记录、数据指纹、协作问题
├── outputs/                  # v1 结果（tables/figures/report.md）
├── outputs/benchmark/        # v2 三段式结果（tables/figures/report.md）
├── CRT应答预测-AI模型基准对比/  # 演示文稿（PPTX + PDF + 源 slides）
├── requirements.txt          # 依赖（请勿随意新增，尤其是 torch）
├── pytest.ini
└── .gitignore
```

---

## 环境与依赖

- **解释器**：建议用本机已有的 `miniconda3`（Python 3.12.4）；本机已具备 scikit-learn 1.9、XGBoost、LightGBM、SHAP、PyTorch 2.6+cu124（GPU 可用）等。
- **安装依赖**（仅首次，使用已有环境，**不要新装 torch 等大依赖**）：

  ```bash
  conda activate base          # 或你的环境
  pip install -r requirements.txt
  ```

- `requirements.txt`：`pandas, numpy, scipy, scikit-learn, openpyxl, xgboost, lightgbm, matplotlib, shap, pytorch-tabnet, pytest, tabulate`。
- 深度学习模型（MLP / TabNet / FT-Transformer）会自动检测并使用 CUDA；无 GPU 时回退 CPU。

---

## 数据

| 项 | 说明 |
|---|---|
| 文件 | `data.xlsx`（**不入库**，仅本地） |
| 样本量 | 163 例 CRT 患者 |
| 因变量 | `response`（应答=1 共 93 例，应答率 57.1%） |
| 自变量 | 其余 28 列（含 13 个二值列、2 个有序列） |
| 缺失 | 无 |
| 已知问题 | `Age=165`（第 33 行）超出合理范围，疑为录入错误；为与已发表队列一致予以保留 |

---

## 怎么跑

### v1：8 模型快速评测（复用已发表 LASSO 基线）

```bash
python -m analysis.run_analysis            # 完整 5×5 重复分层 CV
python -m analysis.run_analysis --quick    # 冒烟测试（1 次重复、无网格）
```

### v2：37 模型三段式基准（推荐，结论以此为准）

```bash
python -m analysis.run_benchmark                 # 完整 A→B→C 三段式
python -m analysis.run_benchmark --quick         # 冒烟（缩小 CV 与模型规模）
python -m analysis.run_benchmark --stages AB     # 只跑 A、B 段
python -m analysis.run_benchmark --top-k 8       # 指定 Stage B 入选模型数
python -m analysis.run_benchmark --jobs 8        # 并行进程数
```

- 所有结果写入 `outputs/benchmark/tables/`（CSV）与 `outputs/benchmark/figures/`（PNG），并生成 `outputs/benchmark/report.md`。
- 全部建模步骤（预处理、单变量筛选、调参）均封装在管道内、只在外层训练折内 `fit`，**不存在数据泄漏**。

### 测试

```bash
pytest                 # 运行全部单元测试（约 92 项）
```

---

## 建模方法与设计

- **候选模型**：37 个，6 大方法族 —— 线性/正则、核与贝叶斯、树与集成、深度学习、集成之集成（Stacking / SuperLearner / Voting）、筛选消融。
- **三段式评测**：
  - **Stage A 广筛**：外层 5 折 × 5 次重复分层 CV，固定超参，全体模型在同一套划分上公平排队（共 25 个区组）。
  - **Stage B 精调**：A 阶段前若干名 + 基线进入嵌套 CV（内层 3 折网格搜索优化 AUC），并做三重显著性比较。
  - **Stage C 稳健性**：留一法（LOOCV）、蒙特卡洛重复留出（100 次 80/20）、Bootstrap 乐观度校正（Harrell 1996，200 次）、多种子重复 CV 名次稳定性。
- **统计口径**：
  - 多重比较同时报告 **Wilcoxon+Holm**（折级配对）、**DeLong**（患者级池化折外）、**Friedman 总括 + Nemenyi 临界差**（Demšar 2006，全模型整体比较）、**Bootstrap ΔAUC 双置信区间**（折块 / 患者层面）。
- **可解释性**：对总榜第一模型做 SHAP（KernelExplainer，仅用于解释，不参与任何评测）。

---

## 关键结论（37 模型 / 163 例）

- **最优模型**：`LDA_Shrinkage`（收缩线性判别分析），外层折平均 **AUC = 0.809**（池化 0.813），且在 Stage C 四种重采样下名次最稳定（5/5 种子第一）。
- **基线位置**：复现的已发表 LASSO 基线折均 AUC ≈ 0.746，在 37 个模型中排第 **19/37**——本队列上多数线性/核/集成方法显著优于它。
- **方法族排名**（按族内平均 AUC）：集成之集成（0.797）> 核与贝叶斯（0.785）> 线性/正则（0.761）> 树与集成（0.728）> 筛选消融（0.685）> **深度学习（0.627，显著垫底）**。
- **显著性**：Friedman χ² = 489.3（p ≈ 6.8e-81）；Nemenyi 事后检验中，`LDA_Shrinkage / Stacking_Logit / SuperLearner_NNLS / GaussianProcess_RBF / SVM_linear` 相对基线达到 Holm 校正显著（p_holm ≤ 0.025）。
- **SHAP 前 5 重要变量**：`systolic_match`、`NS_VT`、`LVEF_Echo`、`LVESD`、`diastolic_match`。
- **解读**：在该数据结构（样本小、EPV≈3）下，简单透明的线性/正则与稳健集成方法优于复杂深度学习；结论属内部验证，需外部队列确认。

> 完整结果见 `outputs/benchmark/report.md` 与演示文稿 `CRT应答预测-AI模型基准对比/`。

---

## 演示文稿

`CRT应答预测-AI模型基准对比/` 包含：

- `CRT应答预测-AI模型基准对比.pptx` / `.pdf` —— v2 全景版（37 模型，15 页）
- `*_v1_8模型版.pptx` / `.pdf` —— v1 八模型版（早期）
- `slides/`（slidep 源）、`assets/`（所用图表）、`STORY.md`（叙事脚本）

> 注：该目录下的 `.cache/` 与 `.slidep/` 为 slidep 工具内部缓存，已 `.gitignore` 排除。

---

## 复现要点与局限

- **随机种子**：2025（模型、CV 划分、调参统一），结果可复现。
- **局限**：单中心、样本量有限（EPV≈3），结论为内部验证；广筛用固定超参，部分方法未达其最优配置（Stage B 已对前排名补做调参）；深度学习劣势应理解为“该数据结构下不适配”，而非否定方法本身。

---

## License / 注意

- 代码以 MIT 许可证开源；**临床数据不随仓库分发**，请遵循所在机构的伦理与数据使用规范。
- 如需协作或获取脱敏数据说明，请联系维护者。

# 方法空间调研（先调研，后计算）

**目的**：在开始跑任何新方案之前，先把「163 例、28 个候选自变量、二分类」这类小样本表格问题上**能用的人工智能/统计建模方法**穷举出来，逐项判定可行性与成本，避免凭印象增补模型。

---

## 1. 外部调研结论（当前方法空间的权威划分）

| 来源 | 关键信息 | 对本项目的约束 |
|---|---|---|
| Hollmann et al., *Accurate predictions on small data with a tabular foundation model*, Nature 2025（TabPFN v2，被引 2100+） | 在 **n ≤ 10 000 行、≤ 500 特征** 的小中样本表格数据上，先验微调 Transformer 一次前向即超过「调参 4 小时的集成基线」 | n=163 恰在其最优区间。这是本任务**理论上最强**的候选，但需 `pip install tabpfn` + 下载权重 → **被「零新安装」约束阻塞**，列为待批准项 |
| The state of Tabular Foundation Models (2026)；TabPFN-2.5 报告 | TabICL（列-行注意力，比 TabPFN v2 快 10×）、TabPFN-2.5（20× 数据容量） | 同属需安装的一类 |
| LAMDA-Tabular **TALENT** benchmark | 35+ 深度方法 + 10+ 经典方法 + 300 数据集的系统评测；被引用为标准方法清单 | 提供了「哪些方法值得算」的可枚举清单，见 §2 |
| **TabArena**（NeurIPS 2025 living benchmark） | 结论：小数据集上 TabPFN 系占优；经典调参集成（XGBoost/LightGBM/CatBoost 集成）为第二梯队 | 提示「深度学习不是越多越好，但**基础模型必须测**」 |
| ACS JCISD 2026《TabPFN Opens New Avenues for Small-Data Tabular…》 | 小数据药物/临床类任务上 TabPFN 作为零训练预测器的实用价值 | 佐证：临床小样本正是基础模型的适用面 |
| Demšar 2006 / Friedman–Nemenyi 惯例；TRIPOD+AI | 多模型整体检验、校准与净收益必须与区分度并列报告 | 已在 `stats_ext.py` 实现，本轮继续沿用 |

**要点**：本项目此前的 37 模型已经覆盖了 TALENT 式「经典 + 主流深度」清单的大部分，**真正的缺口在三处**：
1. **为小样本/低 EPV 专门设计的统计—机器学习混合方法**（Firth 惩罚 logistic、贝叶斯 logistic（Laplace）、probit、混合/单类 LDA、2-NormDR、核密度生成式分类）——n=163/70 事件（EPV≈2.5）正是这些方法的设计场景，而现有 37 个模型里一个都没有。
2. **boosting 变体族**（DART、GOSS、`gblinear`、SAMME、Rotation Forest、Random Subspace、Balanced RF）——现有只有 `log_loss` 单一目标的 RF/ET/GB/HGB/XGB/LGBM 各一个默认配置。
3. **三条非模型轴**（变量空间、分布预处理、验证协议）——「方案不够多」在很大程度上不是模型不够多，而是**同一模型在不同变量空间/不同预处理/不同验证协议下的方案没有铺开**。

---

## 2. 完整方法清单与可行性判定（零新安装前提下）

标记：`✅ 已覆盖（37 模型内）`｜`🆕 本轮新增（本机依赖即可实现）`｜`⛔ 需新依赖（不做，除非批准）`

### 族 1 线性与正则
- ✅ Logistic_Ridge / LASSO_full / ElasticNet / LASSO_baseline（含折内单变量筛选）／LDA／QDA／GaussianNB／Logistic_Poly2／Stepwise_AIC
- 🆕 `Firth_Logistic`（Firth 惩罚似然，自写 Newton-Firth；EPV 低时消除分离与参数偏差）
- 🆕 `BayesLogistic_Laplace`（g-prior/正态先验 + Laplace 后验，输出后验均值概率）
- 🆕 `Probit`（正态链接，自写 IRLS；与 logistic 的差异在小样本上常被报告）
- 🆕 `MLDA_Shrinkage`（`LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')`，单类/混合 LDA，专为 p≈n 设计）
- 🆕 `RegQDA_LedoitWolf`（QDA + `reg_param` 网格 / Ledoit–Wolf 协方差收缩）
- 🆕 `RidgeClassifierCV`、`SGDClassifier(loss='log_loss')`、`SGDClassifier(loss='modified_huber')`、`PassiveAggressive`
- 🆕 `Spline_RidgeLogit`（`SplineTransformer` + 岭 logistic：非线性但低方差）
- 🆕 `Nystroem_RidgeLogit`（`Nystroem` 核近似 + 岭 logistic）
- 🆕 `GaussianNB_var_smoothing` 网格、`BernoulliNB`、`ComplementNB`（后两者对二值/计数编码列有意义）

### 族 2 判别与距离／近邻
- ✅ LDA、KNN
- 🆕 `TwoNormDR`（2 范数降维正则化分类器，p≈n 小样本专用；自写，约 60 行）
- 🆕 `NCA_KNN`（`NeighborhoodComponentsAnalysis` + KNN：可学习的距离度量）
- 🆕 `RadiusNeighbors`、`KNN_distance_weighted`、`KNN_class_weight_balanced`
- 🆕 `KDE_Classifier`（每类高斯核密度 → 似然比判别；`scipy.stats.gaussian_kde`，无新依赖）

### 族 3 核方法与高斯过程
- ✅ SVM_linear / SVM_poly / SVM_rbf / SVM_rbf_screened / GaussianProcess_RBF
- 🆕 `NuSVC_linear` / `NuSVC_rbf` / `NuSVC_sigmoid`
- 🆕 `GP_Matern`、`GP_RationalQuadratic`、`GP_DotProduct`（多核 GP，163 样本下极快）
- 🆕 `SVM_rbf_Calibrated`（`CalibratedClassifierCV` + isotonic / sigmoid：直接改善 Brier/ICI/DCA）

### 族 4 树与 boosting
- ✅ RandomForest / ExtraTrees / GradientBoosting / HistGradientBoosting / XGBoost / LightGBM / DecisionTree / Bagging_Trees / AdaBoost + 三个 screened 消融
- 🆕 `XGB_DART`、`XGB_gblinear`（线性 booster＝带惩罚的 logistic 提升）
- 🆕 `LGBM_DART`、`LGBM_GOSS`
- 🆕 `GB_exponential`（等价 AdaBoost 的指数损失 boosting）、`HGB_l2_regularized`
- 🆕 `AdaBoost_SAMME`（与现有 SAMME.R 不同的离散算法）
- 🆕 `BalancedRandomForest`（类内平衡子采样＝imblearn 的核心思想，自写在折内实现）
- 🆕 `RotationForest`（随机特征子集 + PCA 旋转后再建树；自写，小样本上常胜 RF）
- 🆕 `RandomSubspace_LR` / `RandomPatches_XGB`（`max_features<1` 的列采样集成）
- 🆕 `Bagged_XGB`（`BaggingClassifier` 套 boosting）

### 族 5 集成之集成与后处理
- ✅ Stacking_Logit / SuperLearner_NNLS / Voting_Soft_Hetero
- 🆕 `Stacking_LGBM_meta`（异构元学习器）
- 🆕 `SuperLearner_LogLoss`（对数损失下的非负权重，区别于 AUC/Brier 版）
- 🆕 `BMA_AIC`（AIC 权重的贝叶斯模型平均）
- 🆕 `RankAverage_Ensemble`（OOF 秩平均，不受各模型概率尺度影响）
- 🆕 `WeightedVote_TopK`（按折内 AUC 加权的硬/软投票）

### 族 6 深度学习（本机 GPU + CUDA + torch 2.6 可用）
- ✅ MLP / MLP_Small / MLP_Deep / MLP_screened / TabNet / TabNet_Deep / FT_Transformer
- 🆕 `RealMLP_tuned`（TALENT/TabArena 中被报告的强 MLP 配置：SELU + 权重衰减 + 早停）
- 🆕 `ResNet_Tabular`（带跳跃连接的表格 MLP）
- 🆕 `Mixup_MLP`（表格 mixup 增强，自写）
- 🆕 `AE_Pretrain_MLP`（去噪自编码预训练 + 分类头：小样本迁移思路）
- 🆕 `CNN1D_Tabular`（把特征向量当长度-28 序列的 1D 卷积）
- 🆕 `DeepEnsemble_MLP`（多种子 MLP 平均）
- 🆕 `NeuralLR`（单层网络＝可学习的 logistic，作为深度族的解释性下限）

### 族 7 表格基础模型
- ⛔ `TabPFN_v2`、`TabICL`、`AutoGluon`/`CarbNF`、`CatBoost`：**均需新装**。在「零新安装 + 不占地方」的红线内**不执行**；若要补，仅需一次批准（见 §4）。

---

## 3. 三条「非模型」方案轴（本轮同样要铺开）

| 轴 | 取值 | 说明 |
|---|---|---|
| **V 变量空间** | `all28` / `univ0.05` / `univ0.10` / `lasso` / `top10` / `cluster_rep`（相关聚类代表变量）/ `pca5` | 每折内重新计算，绝不使用全样本信息；这是「EPV≈2.5」问题的直接答案 |
| **P 预处理/分布** | `zscore+median`（现状）/ `rank-gauss` / `quantile-normal` / `median±missing-indicator` / `robust(scaler)` | 数据含极端录入值（如 Age=165）与缺失，分布处理本身就是一个方案变量 |
| **R 验证协议** | 5×5 重复分层 CV（现状）/ 10×10 / LOOCV / Monte-Carlo 80-20 ×100 / bootstrap 乐观度 / bootstrap **.632+** / **置换检验**（标签随机化 → 观察 AUC 的经验零分布） | 置换检验是新加的：回答「0.81 是否只是筛选+调参带来的乐观」 |

**设计原则（避免组合爆炸）**：不跑 130×7×5×7 的全因子。采用**部分因子/分层嵌套**——
- 轴 1（模型）：Stage A 在 `all28 × zscore × 5×5 CV` 下把所有新增模型全部排队（与既有 37 个可直接同表比较）；
- 轴 2（V×P）：对 **Stage A 前 12 名 × 7 变量空间 × 5 预处理** = 420 个方案做一次专门网格（轻量模型为主，深度模型只取其代表）；
- 轴 3（R）：对最终 **前 10 名** 跑全部 7 种验证协议。

---

## 4. 成本预估与红线

- 本机 16 核、RTX 4060 Laptop、torch 2.6+cu124。
- 现有 37 模型三段式（A/B/C）已在 `outputs/benchmark/` 完整产出，作为基线成本参照。
- 预估：**新增约 45 个模型**（多数为秒级～分钟级的线性/核/树方案；深度族约 8 个，GPU 上每个 ≤ 2 分钟）；V×P 网格 420 方案中绝大多数是低成本线性模型；R 轴 10×7。整体 wall-clock 目标 **≤ 60 分钟**（并行 16 核 + GPU），超预算时优先砍深度族的 V×P 网格。
- **红线（不变）**：禁止 `pip`/`conda` 安装任何新包；禁止创建 venv；`data.xlsx` 只读且已 gitignore；缺失能力一律自写（Firth、Laplace、RotationForest、2-NormDR、SMOTE 类平衡、.632+ 均自写）；**禁止 `git add -A`**；**禁止 push**（远端 `origin` 已绑定，历史中含 `data.xlsx` 的早期提交，推送会泄露 163 例患者数据）。
- 若批准放开一条依赖，优先级顺序为：① `tabpfn`（族 7，预期对结论影响最大）② `catboost` ③ `imblearn`（可由自写替代）。

---

## 5. 调研给出的预期结论假设（待计算验证，不作为结论写死）

- H1：在 EPV≈2.5 下，**正则化线性/判别族（LDA-Shrinkage、Firth、岭、SuperLearner）**将优于所有纯深度方案 —— 与现有 37 模型结果一致，需在 V×P 网格下确认稳健性。
- H2：`boosting 变体`（DART/GOSS/gblinear）与 `RotationForest`/`BalancedRF` 会显著好于现有默认配置的树族，但难越线性族。
- H3：V×P 网格中 `cluster_rep` 或 `lasso` 变量空间 + `rank-gauss` 预处理会给出全表最高 AUC —— 即**「方案」比「模型」更重要**。
- H4：置换检验显示真实 AUC 显著高于零分布，但 bootstrap .632+ 与表观 AUC 的差值揭示了原发表研究的乐观度量级。

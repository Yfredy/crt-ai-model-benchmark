# ITM 投稿 v3：极宽方法空间基准（qwen-max 交付）+ 演示文稿

**执行方式**：superpowers:subagent-driven-development（每个 Task 一个新 implementer + 一个 task reviewer，5 轮修复上限，最后一次全分支审查）。

**Goal:** 在 v2/37 模型的基础上，把「163 例 CRT 患者、`response` 为因变量、其余 28 列为候选自变量」这一小样本表格二分类问题的**可用方法尽可能全部算出来**，产出 `qwen-max/` 下的完整结果工件，并最终生成可演示的 **PPTX + PDF**。

**Spec / 调研依据:** `qwen-max/01-method-space-research.md`（先调研后计算的产物；本计划的每个 Task 都可追溯到该文件的方法清单与三条非模型轴）。

**Architecture:** 继续在既有 `analysis/` 包内**增量扩展**，不新建平行包、不改写已通过的模块。新增四个模块：
`models_ultra.py`（新方法注册表，族 1–6）→ `scheme_axes.py`（变量空间轴 V × 预处理轴 P 的折内 transformer）→ `protocols_ext.py`（验证协议轴 R：.632+、置换检验等）→ `run_ultra.py`（编排 + 全部产物落 `qwen-max/`）。所有统计推断复用 `analysis.inference` / `analysis.stats_ext` / `analysis.metrics_ext`，禁止另写一份 DeLong/IRLS。

## Global Constraints

- **零新安装**：解释器固定 `C:/Users/yaoyue/miniconda3/python.exe`（miniconda base）。禁止 `pip`/`conda` 安装、禁止 venv。可用：scikit-learn 1.9.0、xgboost 3.4.1、lightgbm 4.7.0、shap 0.53.0、scipy 1.18.0、numpy 2.4.6、pandas 2.3.3、matplotlib 3.11.1、joblib、torch 2.6.0+cu124（GPU RTX 4060 可用）、pytorch-tabnet 4.1.0、onnxruntime、networkx。缺失能力一律自写（Firth、Laplace 贝叶斯 logistic、probit、2-NormDR、核密度判别、Rotation Forest、Balanced RF、SMOTE 类平衡、.632+、置换检验、BMA、秩平均集成）。
- **不测**需要新依赖的方法（TabPFN v2 / TabICL / CatBoost / AutoGluon / imblearn）；在报告「局限」中如实说明未纳入表格基础模型的原因 = 用户禁止新增依赖。
- **防泄漏**：任何筛选、降维、重采样、标准化、调参只能在**外层训练折内** fit。新增的 V/P 轴组件必须是 sklearn 兼容 transformer 并由 `Pipeline` 承载。
- **sklearn 1.9 API**：`SVC(probability=True)` 已弃用（1.11 移除），一律用 `CalibratedClassifierCV(SVC(...), cv=3)` 折内校准。
- **只读数据**：`data.xlsx` 权限只读且已 gitignore，任何写操作禁止。
- **种子**：统一 `analysis.config.RANDOM_SEED = 2025`；同命令两次运行逐位一致（深度模型固定 torch 种子 + `torch.use_deterministic_algorithms` 尽力而为，并在报告注明 GPU 非确定性）。
- **产物目录**：本轮全部新结果写 `qwen-max/`（`tables/`、`figures/`、`deck/`），**不覆盖** `outputs/`（保留作 37 模型基线对照）。
- **git**：每个 Task 结束一次 commit，前缀 `feat:`/`test:`/`fix:`/`docs:`；**禁止 `git add -A` / `git add .`**，只 add 明确列出的文件；**禁止 `git push`**（远端已绑定，历史含患者数据，推送即泄露）。
- 沟通中文，稿件正文英文。
- **时间预算**：Stage A ultra（≈82 方案）目标 ≤ 25 分钟；V×P 网格（420 方案）≤ 25 分钟；R 轴（10×7 协议）≤ 15 分钟；总 wall-clock ≤ 90 分钟。测试一律 `--quick`/合成数据规模。

---

### Task 1: 新方法注册表 `analysis/models_ultra.py`

**Files:** 新建 `analysis/models_ultra.py`；新建 `tests/test_models_ultra.py`；不改任何既有文件（`analysis/models_ext.py`、`analysis/models.py` 保持原样）。

**Interfaces**
- Produces：`ULTRA_FAMILY: dict[str, str]`、`get_ultra_models(seed=RANDOM_SEED, light=False, device="cpu") -> dict[str, Pipeline]`、`get_ultra_grids() -> dict[str, list[dict]]`、以及自写估计器类 `FirthLogistic`、`BayesLogisticLaplace`、`ProbitRegression`、`TwoNormDR`、`KDEClassifier`、`RotationForestClassifier`、`BalancedForestClassifier`、`RankAverageClassifier`、`BMAWeights`、`SMOTEBalancer`（后者为 sampler transformer）。
- Consumes：`analysis.preprocessing.AutoPreprocessor`、`analysis.features`/`models_ext.UnivariateLRFilter`、`analysis.models.make_baseline_pipeline`、`analysis.config`。先读 `analysis/models_ext.py` 与 `analysis/models.py` 以对齐 `_pipe()`/`_screen_pipe()` 注册风格。

**要求（逐项实现，命名即注册键）**
1. 族 1 线性与正则：`Firth_Logistic`、`BayesLogistic_Laplace`、`Probit`、`MLDA_Shrinkage`、`RegQDA_LedoitWolf`、`RidgeClassifierCV`、`SGD_LogLoss`、`SGD_ModifiedHuber`、`PassiveAggressive`、`Spline_RidgeLogit`、`Nystroem_RidgeLogit`、`BernoulliNB`、`ComplementNB`、`GaussianNB_VarSmooth`
2. 族 2 判别/近邻：`TwoNormDR`、`NCA_KNN`、`RadiusNeighbors`、`KNN_Distance`、`KNN_Balanced`、`KDE_Classifier`
3. 族 3 核/GP：`NuSVC_linear`、`NuSVC_rbf`、`NuSVC_sigmoid`、`GP_Matern`、`GP_RationalQuadratic`、`GP_DotProduct`、`SVM_rbf_Calibrated`
4. 族 4 树/boosting：`XGB_DART`、`XGB_gblinear`、`LGBM_DART`、`LGBM_GOSS`、`GB_Exponential`、`HGB_L2`、`AdaBoost_SAMME`、`Balanced_RandomForest`、`RotationForest`、`RandomSubspace_LR`、`RandomPatches_XGB`、`Bagged_XGB`
5. 族 5 集成之集成：`Stacking_LGBM_meta`、`SuperLearner_LogLoss`、`BMA_AIC`、`RankAverage_TopK`、`WeightedVote_TopK`
6. 族 6 深度（torch，`device` 参数生效）：`RealMLP_SELU`、`ResNet_Tabular`、`Mixup_MLP`、`AE_Pretrain_MLP`、`CNN1D_Tabular`、`DeepEnsemble_MLP`、`NeuralLR`
7. 族 5/6 之外的类不平衡采样对照：`SMOTE_Logistic`、`SMOTE_XGB`（`SMOTEBalancer` 只在训练折内过采样少数类，折内实现，禁止触及测试折）。

**每个类都必须**：`BaseEstimator, ClassifierMixin`、实现 `fit`/`predict_proba`（或 `decision_function`）、`clone`-safe（所有超参为 `__init__` 关键字参数、不在 `__init__` 内拟合）、`get_params/set_params` 不被破坏。

**测试（≥18 个，全部合成数据或小规模，禁止跑真实全量基准）**
- `test_ultra_keys_disjoint_from_ext_and_core`：`set(get_ultra_models())` 与 `get_models()`/`get_ext_models()` 的键**无交集**，且每键都有 `ULTRA_FAMILY` 标签。
- `test_every_ultra_model_fits_and_predicts`：参数化遍历全部键，在 120×8 合成数据（含缺失值、含二值列、含一个近常数列）上 `fit`+`predict_proba`，断言输出形状 `(120,2)`、行和≈1、无 NaN。
- `test_clone_safe`：`clone(est)` 后 `get_params()` 与原对象一致。
- 数学正确性单测（关键）：`Firth_Logistic` 在**完全分离**数据上必须收敛且系数绝对值显著小于未惩罚 MLE（给出可比较的参考实现或断言符号/量级）；`Probit` 与 `Firth_Logistic` 系数比值落在 0.6–0.7（probit≈logit/1.7）；`BayesLogistic_Laplace` 的先验强度→∞ 时系数→0；`TwoNormDR`/`KDE_Classifier`/`RotationForest`/`Balanced_RandomForest`/`SMOTEBalancer` 各断言其核心机制（如 SMOTE 后少数类样本数增加、Balanced RF 每类子样本量相等）。
- `test_no_leakage_train_only`：`SMOTEBalancer`/`UnivariateLRFilter` 类 transformer 在 `Pipeline` 内 `fit` 时只能看到传入的 X，断言其 `fit` 不接受 `y` 之外的全局信息且 `transform` 对测试集形状正确。
- `test_get_ultra_grids_covers_tunable`：`get_ultra_grids()` 的键 ⊆ `get_ultra_models()` 的键，且网格参数名在对应 pipeline 中真实存在（用 `pipeline.get_params()` 校验，**这一条能捕获拼写错误**）。

**DoD**：`python -m pytest -p no:cacheprovider -q` 全绿（既有 92 + 新增 ≥18）；commit。

---

### Task 2: 变量空间轴 × 预处理轴 `analysis/scheme_axes.py`

**Files:** 新建 `analysis/scheme_axes.py`、`tests/test_scheme_axes.py`。

**Interfaces**
- Produces：`make_variable_space(name) -> transformer`，`name ∈ {all28, univ0.05, univ0.10, lasso, top10, cluster_rep, pca5}`；`make_preprocessor(name) -> transformer`，`name ∈ {zscore_median, rank_gauss, quantile_normal, missing_indicator, robust_scaler}`；`variable_space_names()`、`preprocessor_names()`；`grid_products(models: dict, V: list, P: list) -> list[dict]`。
- Consumes：`analysis.inference`（不重复造轮子）、`analysis.preprocessing.AutoPreprocessor`、`analysis.models_ext.UnivariateLRFilter`（若存在）、自写相关聚类代表变量选择。

**要点**
- 全部 V/P 组件是**只依赖训练折的 sklearn transformer**（`fit(X, y)` / `transform(X)`）；`cluster_rep` 用 `scipy.cluster.hierarchy` 在 1−|Spearman| 距离上做相关聚类，每簇取与簇内平均相关最高的变量为代表；`lasso` 用折内 L1 logistic 的非零系数选列；`pca5` 保 95% 方差且上限 5 维。
- `missing_indicator` 必须保留缺失信息（生成 `X_missing` 指示列）而不是抹掉它——这是数据质量问题（Age=165、缺失列）的直接对策。
- `rank_gauss`（高斯秩变换）与 `quantile_normal` 需在 transform 时对新数据做插值映射（保存训练折的分位数/经验 CDF）。

**测试（≥12）**
- 每个 V/P 组合都能进 `Pipeline` 并在合成数据上跑通（参数化 7×5=35 例，规模小）。
- **泄漏专项**：构造「测试集的 y 与 X 关系被反转」的数据，断言折内 fit 的 `univ0.05`/`lasso`/`cluster_rep` 的选列结果**不受测试集影响**（用同一训练折、两个不同测试折得到的 `get_support()/columns_` 完全一致）。
- `pca5`、`rank_gauss`、`quantile_normal` 输出无 NaN、维度符合承诺；`missing_indicator` 生成的列数 = 原列数 + 含缺失列数。
- `grid_products` 生成的方案数 = |models|×|V|×|P|，且键名唯一、命名格式 `<model>__<V>__<P>`。

---

### Task 3: 验证协议轴 `analysis/protocols_ext.py`

**Files:** 新建 `analysis/protocols_ext.py`、`tests/test_protocols_ext.py`；可**追加**（不修改）`analysis/resampling.py` 的导出。

**Interfaces**
- Produces：`bootstrap_0632(y, estimator_factory, X, n_boot=200, seed=..., plus=True) -> dict(auc_apparent, auc_boot, optimism, auc_0632)`；`permutation_auc_null(y, estimator_factory, X, n_perm=500, splits=..., seed=...) -> dict(observed_auc, null_mean, null_q95, p_value, n_perm)`；`repeated_cv_protocol(n_folds, n_repeats, seed)`；`protocol_names() -> list[str]`（含 `cv5x5`、`cv10x10`、`loocv`、`mc8020_100`、`bootstrap_optimism`、`bootstrap_0632plus`、`permutation_test`）；`run_protocol(name, estimator, X, y, seed) -> dict` 统一返回 `{"protocol","auc","ci_lo","ci_hi","n_evals","extra"}`。
- Consumes：`analysis.inference.delong_ci/bootstrap_auc_ci/pooled_oof`、`analysis.resampling`（已有 LOOCV / MC holdout / bootstrap 乐观度，复用勿重写）。

**要点**
- `.632+` 严格按 Efron–Tibshirani 定义实现（含 R̂ 与 D，独立样本假设下的 `d_i` 用留一预测概率计算）；在小样本、类不平衡下的退化分支（R̂=0）必须有处理并被测试。
- 置换检验：对**完整建模流程**（含折内筛选与调参）做标签随机化，得到「本流程在零假设下能达到的 AUC 分布」，从而量化「0.81 里有多少是选择带来的乐观」。这是 v2 完全没有的、审稿最可能问的证据。
- 两者都必须能被 `run_ultra.py` 并行调用（无全局状态，纯函数 + 工厂）。

**测试（≥10，合成小规模）**
- `.632+` 在一个**人工纯噪声**数据集上估计出的 AUC 应接近 0.5（宽松阈值 0.35–0.65），在一个**强信号**数据集上接近但低于表观 AUC（`auc_0632 < auc_apparent` 必成立）。
- 置换检验：强信号数据 → `p_value` 极小；纯噪声 + 含折内筛选的流程 → `p_value` 不显著（>0.05），证明该方法能识破「筛选制造的假高分」。
- `run_protocol` 对 7 个协议名都能返回含全部键的 dict；未知协议名 → `ValueError`。
- 种子固定时两次调用结果逐位相同。

---

### Task 4: 编排器 `analysis/run_ultra.py`（全部产物落 `qwen-max/`）

**Files:** 新建 `analysis/run_ultra.py`；`tests/test_run_ultra.py`；`.gitignore` 追加 `qwen-max/` 之外的中间缓存（**不要** gitignore `qwen-max/` 本身，结果需入库）。

**Interfaces**
- Consumes：`get_ultra_models`/`get_ultra_grids`（Task 1）、`get_ext_models`（既有 37）、`scheme_axes`（Task 2）、`protocols_ext`（Task 3）、`analysis.run_benchmark` 的 `evaluate_model_cv`/`run_parallel`/`summarize_folds`/`family_summary`（复用，不改其行为）、`analysis.stats_ext`（Friedman/Nemenyi/Holm）、`analysis.bench_plots`、`analysis.bench_report`、`analysis.explain`。
- Produces：`main()`，CLI `python -m analysis.run_ultra [--quick] [--stage a|vp|r|b|all] [--jobs N] [--top-k K] [--device cpu|cuda] [--out-dir qwen-max]`。

**必须产出的文件**（路径以 `qwen-max/` 为根）
- `tables/stage_a_all_models.csv` —— **既有 37 + 全部新增**在同一 5×5 CV、同一划分下的合并总榜（列：model, family, n_folds, auc_mean, auc_sd, auc_min, auc_max, ap, brier, ici, mcc, sens, spec, ppv, npv, calib_slope, calib_intercept, mean_net_benefit）
- `tables/stage_a_friedman_nemenyi.csv`、`tables/stage_a_paired_vs_baseline.csv`（vs `Logistic_LASSO_baseline`：折级配对 Wilcoxon + DeLong + Holm）
- `tables/stage_vp_grid.csv` —— V×P 全网格（`model__V__P` 长表），`tables/stage_vp_top50.csv`、`tables/stage_vp_axis_effects.csv`（V 轴与 P 轴的主效应：对每个 V 取所有模型 AUC 的均值/中位数，并做 Friedman）
- `tables/stage_r_protocols.csv` —— 前 10 名 × 7 验证协议
- `tables/stage_b_nested_tuning.csv` —— 前 K（默认 12）名做嵌套调参（内层 3 折）后的结果
- `tables/final_leaderboard.csv` —— **最终单一结论表**：每个方案一行，含 AUC 及各协议一致性、是否显著优于基线、家族
- `figures/*.png` —— 家族箱线、总榜 Top-20 条图（带 DeLong/bootstrap CI）、Nemenyi CD 图、V 轴效应图、P 轴效应图、ROC、校准、DCA、置换检验零分布直方图、SHAP（对最终冠军模型）
- `report.md` —— 中文长报告（设计、方法清单、结果表、结论、局限含「未纳入表格基础模型因禁止新增依赖」）+ 英文 `## Methods` 与 `## Results` 段落可直接进稿件
- `report.json` —— 机器可读的完整结果（含运行时间、种子、包版本）

**编排要求**
- Stage A ultra：`run_parallel` 跑 37+~60 方案；深度族 `device` 可选 `cuda`；每个方案的异常必须被捕获并记录为 `note`（**一个模型崩溃不能毁掉整轮**），崩溃清单进 `report.md` 的「失败方案」小节。
- V×P 网格：仅对 Stage A 前 12 名（深度族只保留 `RealMLP_SELU`、`FT_Transformer`、`TabNet` 三个代表）跑 7×5；其余只跑 `all28 × zscore_median`。
- 全程写 `qwen-max/tables/*.csv` 与 `qwen-max/report.md`，运行日志写 `qwen-max/run.log`。
- `--quick`：3 个超便宜方案 × V 只取 2 个 × P 只取 2 个 × 协议只取 2 个，总时长 ≤ 2 分钟，且必须真实产出全部文件（用于测试与 CI）。

**测试（≥8，用 `--quick` 或 monkeypatch 缩小规模）**
- `test_quick_run_produces_all_artifacts`：跑 `main(["--quick"])`，断言上面每张表/每幅图/`report.md`/`report.json` 都被创建且非空，列名符合约定。
- `test_leaderboard_is_sorted_and_complete`：`final_leaderboard.csv` 按 AUC 降序、无重复模型名、行数 == 参与方案数。
- `test_crashed_model_does_not_abort_run`：注入一个 `fit` 抛异常的假估计器，断言整体继续且该方案出现在失败清单。
- `test_no_writes_to_outputs_dir`：断言本轮运行不修改 `outputs/` 下任何文件（用 mtime/哈希比对）。
- `test_reproducible_same_seed`：两次 `--quick` 同种子 → `stage_a_all_models.csv` 数值逐位一致。

---

### Task 5: 结果固化与冠军解释（`qwen-max/02-conclusions.md`）

**Files:** 新建 `analysis/ultra_report.py` 或复用 `analysis/bench_report.py`（择优，不得重复实现）；新建 `qwen-max/02-conclusions.md`；`tests/test_ultra_report.py`。

**要求**
- 从 `qwen-max/report.json` + 各表生成**结论级**文档：① 最优模型/最优方案（含它的 V、P、协议）；② 方法族排序与统计显著性；③ 「方案 vs 模型」效应大小对比（V/P 主效应是否超过换模型的增益——这是本研究最有价值的发现）；④ 深度学习在本数据上的定位；⑤ 稳健性矩阵（前 10 名 × 7 协议的一致性）；⑥ 显著性（vs 已发表 LASSO 基线）；⑦ 与原发表数值的差距说明（不得声称复现）。
- 对冠军模型出 SHAP（线性/核路径自动选择）+ 一个典型病例解释。
- 中文正文；英文 `## Methods`/`## Results` 段落可直接粘贴到稿件。**红线**：不写「外部验证」「临床常规采用」等超出单中心内部验证的表述；不写「复现了 Jiang et al.」。

**测试（≥4）**：文档生成函数在 quick 结果上产出全部 7 个小节标题；关键数字（冠军名、冠军 AUC、最优 V、最优 P）与 CSV 一致；缺数据时给出明确 `NA` 而非崩溃。

---

### Task 6: 演示交付物 PPTX + PDF（`qwen-max/deck/`）

**Files:** `qwen-max/deck/`（`slides/`、`assets/`、`STORY.md`、`*.pptx`、`*.pdf`）。

**要求**
- 用 **ppt-master** 技能生成面向「演示给别人看」的成片：叙事顺序 = 问题（编辑要求补 AI 层）→ 数据与防泄漏设计 → 方法空间（族 × 轴全景图）→ 结果（总榜、CD 图、V/P 效应、稳健性）→ 结论（哪个模型最适配、为什么深度不行、方案比模型更重要）→ 局限与下一步（TabPFN 等待批准项）。
- 图表一律引用 `qwen-max/figures/` 的真实产物；关键数字来自 `final_leaderboard.csv`，**禁止**在幻灯片上写与 CSV 不一致的数值。
- 同时导出 **PPTX 与 PDF** 两种格式，文件名含日期与「37→N 模型」规模标识。
- 中文为主、术语保留英文。

**DoD**：`qwen-max/deck/` 下 PPTX 与 PDF 均存在且可打开（PDF 页数 == 幻灯片数），commit。

---

### Task 7: 全量重跑 + 验收 + 最终审查

- 以完整规模跑 `python -m analysis.run_ultra`（非 quick），确认 ≤ 90 分钟预算并记录实际耗时。
- 校验：`sha256sum data.xlsx` == `docs/data-fingerprint.txt`；`git status` 中 `data.xlsx` 仍未被跟踪；全量 pytest 绿；`qwen-max/` 全部工件存在；报告中每个数字可追溯到某张 CSV。
- 用 Task 6 的最新数字重生成 PPTX/PDF。
- 最后一次全分支审查（最强模型）+ 修复轮 + 清理工作区。

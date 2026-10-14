# CRT 应答预测：多方案机器学习基准评测报告

**任务**：在 `data.xlsx`（163 例 CRT 患者）上，以 `response` 为因变量、其余 28 列为候选自变量，系统比较**传统统计模型与 AI 模型**的预测能力。

## 1. 设计总览

- **候选模型数**：37 个，覆盖 6 个方法族（线性/正则、核与贝叶斯、树与集成、深度学习、集成之集成、筛选消融）
- **Stage A（广筛）**：外层 5 折 × 5 次重复分层 CV，固定超参，全部模型在同一套划分上公平排队
- **Stage B（精调）**：A 阶段前若干名进入嵌套 CV，内层 3 折网格搜索优化 AUC
- **Stage C（稳健性）**：留一法（LOOCV）、蒙特卡洛重复留出（100 次 80/20）、Bootstrap 乐观度校正（200 次）、多种子重复 CV
- **队列**：163 例，应答 93 例（应答率 57.1%）
- **随机种子**：2025（所有模型、CV 划分与调参统一）
- **防泄漏**：预处理、单变量筛选、超参选择全部封装在管道内，只在外层训练折内 fit

## 2. Stage A：全模型广筛结果

### 2.1 总榜（按外层折平均 AUC 降序，前 15）

| model               | family                |   auc_mean |   auc_sd |   auc_min |   auc_max |   ap_mean |   brier_mean |   mean_net_benefit_mean |   calibration_slope_mean |
|:--------------------|:----------------------|-----------:|---------:|----------:|----------:|----------:|-------------:|------------------------:|-------------------------:|
| LDA_Shrinkage       | Linear & Regularized  |      0.809 |    0.047 |     0.714 |     0.906 |     0.860 |        0.184 |                   0.291 |                    0.664 |
| Stacking_Logit      | Ensemble of Ensembles |      0.802 |    0.046 |     0.711 |     0.901 |     0.851 |        0.184 |                   0.284 |                    1.346 |
| GaussianProcess_RBF | Kernel & Bayes        |      0.801 |    0.046 |     0.692 |     0.865 |     0.848 |        0.183 |                   0.284 |                    1.394 |
| SuperLearner_NNLS   | Ensemble of Ensembles |      0.800 |    0.045 |     0.718 |     0.872 |     0.853 |        0.184 |                   0.287 |                    1.025 |
| SVM_linear          | Kernel & Bayes        |      0.800 |    0.050 |     0.688 |     0.883 |     0.848 |        0.184 |                   0.282 |                    1.567 |
| SVM_poly            | Kernel & Bayes        |      0.797 |    0.059 |     0.639 |     0.898 |     0.843 |        0.194 |                   0.268 |                    2.008 |
| ExtraTrees          | Trees & Ensembles     |      0.795 |    0.061 |     0.688 |     0.948 |     0.839 |        0.187 |                   0.277 |                    1.440 |
| Logistic_Ridge      | Linear & Regularized  |      0.793 |    0.054 |     0.688 |     0.877 |     0.843 |        0.185 |                   0.278 |                    0.752 |
| Logistic_ElasticNet | Linear & Regularized  |      0.790 |    0.051 |     0.684 |     0.877 |     0.838 |        0.187 |                   0.275 |                    0.758 |
| SVM_rbf             | Kernel & Bayes        |      0.790 |    0.053 |     0.650 |     0.887 |     0.845 |        0.189 |                   0.276 |                    1.430 |
| Voting_Soft_Hetero  | Ensemble of Ensembles |      0.788 |    0.051 |     0.695 |     0.877 |     0.838 |        0.185 |                   0.284 |                    1.101 |
| Logistic_LASSO_full | Linear & Regularized  |      0.784 |    0.053 |     0.673 |     0.877 |     0.831 |        0.192 |                   0.272 |                    0.750 |
| LDA                 | Linear & Regularized  |      0.778 |    0.064 |     0.654 |     0.902 |     0.819 |        0.202 |                   0.234 |                    0.484 |
| XGBoost             | Trees & Ensembles     |      0.755 |    0.065 |     0.639 |     0.883 |     0.803 |        0.217 |                   0.229 |                    0.505 |
| QDA                 | Linear & Regularized  |      0.752 |    0.085 |     0.564 |     0.877 |     0.806 |        0.240 |                   0.177 |                    0.334 |

完整表格见 `tables/stage_a_summary.csv` （共 37 个模型）。

### 2.2 按方法族汇总

| family                |   n_models |   auc_mean |   auc_sd_across_folds |   best_model_auc | best_model          |
|:----------------------|-----------:|-----------:|----------------------:|-----------------:|:--------------------|
| Ensemble of Ensembles |          3 |      0.797 |                 0.047 |            0.802 | Stacking_Logit      |
| Kernel & Bayes        |          5 |      0.785 |                 0.058 |            0.801 | GaussianProcess_RBF |
| Linear & Regularized  |         10 |      0.761 |                 0.071 |            0.809 | LDA_Shrinkage       |
| Trees & Ensembles     |          9 |      0.728 |                 0.078 |            0.795 | ExtraTrees          |
| Ablation (screening)  |          4 |      0.685 |                 0.089 |            0.739 | SVM_rbf_screened    |
| Deep Learning         |          6 |      0.627 |                 0.104 |            0.676 | FT_Transformer      |

### 2.3 池化折外预测性能（含 DeLong 与自助双置信区间，前 10）

| model               |   auc |   auc_ci_delong_lo |   auc_ci_delong_hi |   auc_ci_boot_lo |   auc_ci_boot_hi |   calib_slope |   calib_intercept |   brier |
|:--------------------|------:|-------------------:|-------------------:|-----------------:|-----------------:|--------------:|------------------:|--------:|
| LDA_Shrinkage       | 0.813 |              0.747 |              0.879 |            0.744 |            0.876 |         0.634 |             0.056 |   0.179 |
| SVM_linear          | 0.808 |              0.741 |              0.876 |            0.738 |            0.872 |         1.409 |            -0.115 |   0.182 |
| Logistic_Ridge      | 0.805 |              0.737 |              0.873 |            0.736 |            0.871 |         0.726 |             0.041 |   0.180 |
| SVM_poly            | 0.805 |              0.736 |              0.873 |            0.731 |            0.867 |         1.447 |            -0.054 |   0.191 |
| SuperLearner_NNLS   | 0.804 |              0.737 |              0.871 |            0.734 |            0.869 |         1.002 |             0.012 |   0.178 |
| GaussianProcess_RBF | 0.803 |              0.735 |              0.871 |            0.733 |            0.866 |         1.319 |            -0.019 |   0.180 |
| Stacking_Logit      | 0.802 |              0.733 |              0.870 |            0.731 |            0.867 |         1.303 |            -0.060 |   0.180 |
| Logistic_ElasticNet | 0.801 |              0.733 |              0.870 |            0.729 |            0.868 |         0.734 |             0.038 |   0.182 |
| Voting_Soft_Hetero  | 0.797 |              0.728 |              0.866 |            0.726 |            0.861 |         1.108 |            -0.029 |   0.180 |
| SVM_rbf             | 0.795 |              0.727 |              0.863 |            0.724 |            0.856 |         1.328 |             0.019 |   0.186 |

### 2.4 多模型整体检验：Friedman + Nemenyi

- Friedman 总括检验：χ² = **489.306**，p = **6.82e-81**（37 个模型 × 25 个重采样区组）
- Nemenyi 临界差 CD = **16.67**（α=0.05）：平均秩落在基线 ±CD 区间内的模型，与基线无统计学显著差异

与已发表 LASSO 基线的 Nemenyi 事后检验（Holm 校正）：

| model                   |   mean_rank |   rank_diff_vs_control |   z_nemenyi |   p_nemenyi |   p_holm | significant_vs_control   | direction   |
|:------------------------|------------:|-----------------------:|------------:|------------:|---------:|:-------------------------|:------------|
| LDA_Shrinkage           |       6.760 |                -11.900 |      -3.887 |       0.000 |    0.004 | True                     | better      |
| Stacking_Logit          |       7.540 |                -11.120 |      -3.632 |       0.000 |    0.009 | True                     | better      |
| SuperLearner_NNLS       |       8.120 |                -10.540 |      -3.443 |       0.001 |    0.017 | True                     | better      |
| GaussianProcess_RBF     |       8.220 |                -10.440 |      -3.410 |       0.001 |    0.019 | True                     | better      |
| SVM_linear              |       8.520 |                -10.140 |      -3.312 |       0.001 |    0.025 | True                     | better      |
| SVM_poly                |       9.760 |                 -8.900 |      -2.907 |       0.004 |    0.084 | False                    | better      |
| Logistic_Ridge          |      10.100 |                 -8.560 |      -2.796 |       0.005 |    0.114 | False                    | better      |
| Voting_Soft_Hetero      |      10.160 |                 -8.500 |      -2.776 |       0.005 |    0.115 | False                    | better      |
| ExtraTrees              |      10.300 |                 -8.360 |      -2.731 |       0.006 |    0.126 | False                    | better      |
| SVM_rbf                 |      10.980 |                 -7.680 |      -2.509 |       0.012 |    0.230 | False                    | better      |
| Logistic_ElasticNet     |      11.180 |                 -7.480 |      -2.443 |       0.015 |    0.262 | False                    | better      |
| Logistic_LASSO_full     |      12.580 |                 -6.080 |      -1.986 |       0.047 |    0.706 | False                    | better      |
| LDA                     |      13.600 |                 -5.060 |      -1.653 |       0.098 |    1.000 | False                    | better      |
| XGBoost                 |      16.320 |                 -2.340 |      -0.764 |       0.445 |    1.000 | False                    | better      |
| QDA                     |      16.760 |                 -1.900 |      -0.621 |       0.535 |    1.000 | False                    | better      |
| Logistic_LASSO_baseline |      18.660 |                  0.000 |       0.000 |       1.000 |    1.000 | False                    | control     |
| LightGBM                |      19.060 |                  0.400 |       0.131 |       0.896 |    1.000 | False                    | worse       |
| RandomForest            |      19.220 |                  0.560 |       0.183 |       0.855 |    1.000 | False                    | worse       |
| SVM_rbf_screened        |      19.460 |                  0.800 |       0.261 |       0.794 |    1.000 | False                    | worse       |
| HistGradientBoosting    |      19.740 |                  1.080 |       0.353 |       0.724 |    1.000 | False                    | worse       |
| KNN                     |      19.860 |                  1.200 |       0.392 |       0.695 |    1.000 | False                    | worse       |
| GradientBoosting        |      21.520 |                  2.860 |       0.934 |       0.350 |    1.000 | False                    | worse       |
| GaussianNB              |      21.660 |                  3.000 |       0.980 |       0.327 |    1.000 | False                    | worse       |
| Logistic_Poly2          |      22.580 |                  3.920 |       1.280 |       0.200 |    1.000 | False                    | worse       |
| Stepwise_Logistic_AIC   |      23.120 |                  4.460 |       1.457 |       0.145 |    1.000 | False                    | worse       |
| Bagging_Trees           |      23.480 |                  4.820 |       1.574 |       0.115 |    1.000 | False                    | worse       |
| AdaBoost                |      24.380 |                  5.720 |       1.868 |       0.062 |    0.864 | False                    | worse       |
| RandomForest_screened   |      24.860 |                  6.200 |       2.025 |       0.043 |    0.686 | False                    | worse       |
| XGBoost_screened        |      26.060 |                  7.400 |       2.417 |       0.016 |    0.266 | False                    | worse       |
| MLP_Deep                |      27.900 |                  9.240 |       3.018 |       0.003 |    0.061 | False                    | worse       |
| FT_Transformer          |      28.040 |                  9.380 |       3.064 |       0.002 |    0.055 | False                    | worse       |
| TabNet                  |      28.120 |                  9.460 |       3.090 |       0.002 |    0.052 | False                    | worse       |
| DecisionTree            |      28.880 |                 10.220 |       3.338 |       0.001 |    0.024 | True                     | worse       |
| MLP_screened            |      30.300 |                 11.640 |       3.802 |       0.000 |    0.005 | True                     | worse       |
| MLP_Small               |      30.400 |                 11.740 |       3.835 |       0.000 |    0.004 | True                     | worse       |
| TabNet_Deep             |      31.340 |                 12.680 |       4.142 |       0.000 |    0.001 | True                     | worse       |
| MLP                     |      33.460 |                 14.800 |       4.834 |       0.000 |    0.000 | True                     | worse       |

![Nemenyi 秩图](../figures/nemenyi_ranks.png)

## 3. Stage B：嵌套调参后的前排名

| model                   | family                |   auc_mean |   auc_sd |   auc_min |   auc_max |   mean_net_benefit_mean |   calibration_slope_mean |
|:------------------------|:----------------------|-----------:|---------:|----------:|----------:|------------------------:|-------------------------:|
| Stacking_Logit          | Ensemble of Ensembles |      0.802 |    0.046 |     0.711 |     0.901 |                   0.284 |                    1.346 |
| SuperLearner_NNLS       | Ensemble of Ensembles |      0.800 |    0.045 |     0.718 |     0.872 |                   0.287 |                    1.025 |
| GaussianProcess_RBF     | Kernel & Bayes        |      0.800 |    0.048 |     0.692 |     0.865 |                   0.282 |                    1.365 |
| LDA_Shrinkage           | Linear & Regularized  |      0.795 |    0.046 |     0.706 |     0.893 |                   0.277 |                    0.615 |
| ExtraTrees              | Trees & Ensembles     |      0.792 |    0.062 |     0.671 |     0.937 |                   0.273 |                    1.665 |
| SVM_poly                | Kernel & Bayes        |      0.786 |    0.060 |     0.639 |     0.898 |                   0.264 |                    1.980 |
| SVM_linear              | Kernel & Bayes        |      0.786 |    0.053 |     0.673 |     0.877 |                   0.277 |                    1.426 |
| Logistic_Ridge          | Linear & Regularized  |      0.783 |    0.054 |     0.665 |     0.877 |                   0.268 |                    0.936 |
| Logistic_LASSO_baseline | Linear & Regularized  |      0.746 |    0.074 |     0.568 |     0.906 |                   0.252 |                    0.854 |

### 3.1 与 LASSO 基线的三重显著性比较

**口径说明**

- `p_wilcoxon_holm`：以每个外层折 AUC 为配对单位的 Wilcoxon 符号秩检验 + Holm 校正（配对说法正确的是这一种）
- `p_delong_holm`：在池化折外预测上做 DeLong 检验 + Holm 校正（患者层面、阈值无关）
- `auc_diff_ci_*`：ΔAUC 的两种自助置信区间，block=折块自助，patient=患者层面配对自助

| model               |   auc_fold_mean |   auc_diff_fold |   auc_diff_ci_block_lo |   auc_diff_ci_block_hi |   auc_diff_ci_patient_lo |   auc_diff_ci_patient_hi |   p_wilcoxon |   auc_pooled |   auc_diff_pooled |   z_delong |   p_delong |   p_wilcoxon_holm |   p_delong_holm |
|:--------------------|----------------:|----------------:|-----------------------:|-----------------------:|-------------------------:|-------------------------:|-------------:|-------------:|------------------:|-----------:|-----------:|------------------:|----------------:|
| Stacking_Logit      |           0.802 |           0.057 |                  0.033 |                  0.080 |                    0.008 |                    0.100 |        0.001 |        0.802 |             0.052 |      2.197 |      0.028 |             0.005 |           0.168 |
| SuperLearner_NNLS   |           0.800 |           0.055 |                  0.032 |                  0.077 |                    0.013 |                    0.100 |        0.000 |        0.804 |             0.054 |      2.427 |      0.015 |             0.004 |           0.122 |
| GaussianProcess_RBF |           0.800 |           0.054 |                  0.030 |                  0.078 |                    0.008 |                    0.104 |        0.001 |        0.804 |             0.053 |      2.163 |      0.030 |             0.005 |           0.168 |
| LDA_Shrinkage       |           0.795 |           0.049 |                  0.020 |                  0.074 |                    0.012 |                    0.103 |        0.004 |        0.804 |             0.054 |      2.317 |      0.021 |             0.014 |           0.144 |
| ExtraTrees          |           0.792 |           0.047 |                  0.027 |                  0.065 |                   -0.002 |                    0.085 |        0.000 |        0.791 |             0.041 |      1.819 |      0.069 |             0.004 |           0.258 |
| SVM_poly            |           0.786 |           0.040 |                  0.009 |                  0.070 |                   -0.008 |                    0.103 |        0.016 |        0.796 |             0.045 |      1.563 |      0.118 |             0.049 |           0.258 |
| SVM_linear          |           0.786 |           0.040 |                  0.009 |                  0.071 |                   -0.000 |                    0.102 |        0.020 |        0.798 |             0.048 |      1.849 |      0.065 |             0.049 |           0.258 |
| Logistic_Ridge      |           0.783 |           0.037 |                  0.006 |                  0.066 |                   -0.004 |                    0.098 |        0.018 |        0.794 |             0.044 |      1.719 |      0.086 |             0.049 |           0.258 |

![ROC](../figures/roc_stage_b.png)

![Calibration](../figures/calibration_stage_b.png)

![DCA](../figures/dca_stage_b.png)

## 4. Stage C：结论在多种重采样下是否站得住

| model               | family                |   auc_loo |   auc_montecarlo |   auc_apparent |   optimism |   auc_corrected |   auc_corrected_ci_lo |   auc_corrected_ci_hi |   auc_632 |   n_boot |
|:--------------------|:----------------------|----------:|-----------------:|---------------:|-----------:|----------------:|----------------------:|----------------------:|----------:|---------:|
| LDA_Shrinkage       | Linear & Regularized  |     0.810 |            0.812 |          0.891 |      0.059 |           0.832 |                 0.790 |                 0.875 |     0.875 |      200 |
| Stacking_Logit      | Ensemble of Ensembles |     0.792 |            0.810 |          0.980 |      0.047 |           0.933 |                 0.905 |                 0.955 |     0.963 |      200 |
| GaussianProcess_RBF | Kernel & Bayes        |     0.803 |            0.806 |          0.977 |      0.030 |           0.947 |                 0.921 |                 0.965 |     0.972 |      200 |
| SuperLearner_NNLS   | Ensemble of Ensembles |     0.796 |            0.809 |          0.980 |      0.043 |           0.937 |                 0.905 |                 0.959 |     0.965 |      200 |

`auc_apparent`→`optimism`→`auc_corrected` 是 Bootstrap 乐观度校正链条：模型在全数据上的表观性能减去自抽样估计的乐观度，得到对外部数据的近似期望性能。

![Robustness](../figures/stage_c_robustness.png)

### 4.1 多种子下的名次稳定性

| model               |   mean_rank |   sd_rank |   times_best |   times_top3 |
|:--------------------|------------:|----------:|-------------:|-------------:|
| LDA_Shrinkage       |        1.00 |      0.00 |            5 |            5 |
| Stacking_Logit      |        2.60 |      0.89 |            0 |            4 |
| SuperLearner_NNLS   |        2.80 |      0.45 |            0 |            5 |
| GaussianProcess_RBF |        3.60 |      0.89 |            0 |            1 |

`times_best`/`times_top3` 表示在多个随机种子下该模型获得第一 / 进入前三的次数——用于区分真实方法优势与随机种子运气。

## 5. 最优模型（LDA_Shrinkage）的可解释性

SHAP 方法：**KernelExplainer**（在全队列上拟合，仅用于解释，不影响上述任何评估结果）

|   mean_abs_shap |
|----------------:|
|          0.0815 |
|          0.0691 |
|          0.0536 |
|          0.0491 |
|          0.0399 |
|          0.0331 |
|          0.0283 |
|          0.0282 |
|          0.0278 |
|          0.0214 |

![SHAP bar](../figures/shap_bar.png)

## 6. 数据质量备注

- Age=165 at row 33 outside plausible range [18, 120] (suspected entry error; kept as-is to match the published cohort)

## 7. 可复用的 Methods 文本

Candidate predictors comprised all 163 patients' 28 baseline variables. We benchmarked 37 modelling approaches spanning penalised regression, linear/quadratic discriminant analysis, naive Bayes, kernel machines, Gaussian processes, k-nearest neighbours, tree ensembles, gradient boosting, feed-forward networks, TabNet, an FT-Transformer, soft voting, stacked generalisation and a non-negative super learner. Every learner was embedded in a leakage-free pipeline in which imputation, scaling, optional univariate screening and hyper-parameter tuning were fitted inside the outer training fold only. Stage A used stratified 5-fold cross-validation repeated 5 times with fixed default hyper-parameters; Stage B re-evaluated the leading candidates with nested cross-validation (inner 3-fold grid search optimising AUC). Discrimination was quantified by AUC and average precision; calibration by Brier score, calibration slope and intercept; clinical utility by decision- curve net benefit. Differences against the published LASSO reference model were tested with paired Wilcoxon signed-rank tests over folds (Holm correction), DeLong tests on pooled out-of-fold predictions, and bootstrap confidence intervals. A Friedman omnibus test with Nemenyi post-hoc analysis (Demšar 2006) was used for the overall multi-model comparison. Leading models were further re-validated with leave-one-out cross-validation, 100 replicate Monte-Carlo holdouts, bootstrap optimism correction (Harrell 1996) and multi-seed repetition.

## 8. 局限

- 单中心、样本量有限（163 例、28 个候选变量，EPV 约 3），所有结论本质上是内部验证，尚需外部队列确认。
- 地真手写数据与超参网格有限：广筛阶段使用固定超参，因此某方法族的代表模型可能未达到其最优配置（Stage B 已对前排名补做调参）。
- 深度学习方案的样本量远低于其常规需求，其劣势应理解为「该数据结构下不适配」，而非否定方法本身。
- 净收益区间依赖阈概率范围假设，临床解读时应结合具体决策场景。

## 9. 复现

```bash
python -m analysis.run_benchmark            # 完整三段式评测
python -m analysis.run_benchmark --quick    # 冒烟（缩小 CV 与模型规模）
```
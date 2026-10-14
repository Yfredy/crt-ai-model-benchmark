"""扩展模型库：把候选模型从 8 个扩到 ~38 个，覆盖六大方法族。

设计原则：
- 所有模型都包进 Pipeline，预处理只在训练折 fit → 无数据泄漏；
- 使用 sklearn 1.9 的现行 API（不再依赖 1.10/1.11 将移除的 penalty= / probability=）；
- 每个模型带 FAMILY 标签，便于按族汇报；
- get_ext_models() 返回 {name: pipeline}，get_ext_grids() 返回 {name: 参数网格}，
  网格中的键名与 pipeline 步骤名一致。
"""
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from analysis.baseline import make_baseline_pipeline, UnivariateLRFilter
from analysis.config import RANDOM_SEED
from analysis.models import TabNetWrapper
from analysis.preprocessing import AutoPreprocessor

# 家族标签：用于按族汇总与叙事
FAMILY = {}


def _pipe(clf) -> Pipeline:
    return Pipeline([("pre", AutoPreprocessor()), ("clf", clf)])


def _screen_pipe(clf) -> Pipeline:
    """带单变量筛选的管道：pre → univar(p<0.05) → clf（论文同类做法）。"""
    return Pipeline([("pre", AutoPreprocessor()),
                     ("univar", UnivariateLRFilter(p_threshold=0.05)),
                     ("clf", clf)])


# --------------------------------------------------------------------------
# 自研封装 1：逐步 Logistic（正向选择，AIC 准则）——临床统计的经典做法
# --------------------------------------------------------------------------

class StepwiseLogistic(BaseEstimator, ClassifierMixin):
    """正向逐步 logistic 回归：每轮加入使 AIC 下降最多的变量，至 max_k 或无改进。

    max_k 由 EPV(>]限制：候选变量 28、事件数 ~93，取 6 留出安全边际。
    """

    def __init__(self, max_k=6, p_enter=0.05, seed=RANDOM_SEED):
        self.max_k = max_k
        self.p_enter = p_enter
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        selected = []
        candidates = list(range(X.shape[1]))
        while len(selected) < self.max_k and candidates:
            best_j, best_aic = None, np.inf
            for j in candidates:
                cols = selected + [j]
                if np.std(X[:, cols], axis=0).min() < 1e-12:
                    continue
                aic = self._aic(X[:, cols], y)
                if aic < best_aic:
                    best_j, best_aic = j, aic
            if best_j is None:
                break
            trial_aic = self._aic(X[:, selected + [best_j]], y)
            cur_aic = self._aic(X[:, selected], y) if selected else np.inf
            if selected and trial_aic >= cur_aic:
                break
            selected.append(best_j)
            candidates.remove(best_j)
        if not selected:
            selected = list(range(min(3, X.shape[1])))
        self.selected_ = selected
        self.model_ = LogisticRegression(C=1e6, max_iter=5000).fit(X[:, selected], y)
        self.classes_ = np.unique(y)
        return self

    @staticmethod
    def _aic(Xc, y):
        if Xc.shape[1] == 0:
            p = y.mean()
            ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
            return 2.0 - 2 * ll
        try:
            lr = LogisticRegression(C=1e6, max_iter=2000).fit(Xc, y)
        except Exception:
            return np.inf
        pr = np.clip(lr.predict_proba(Xc)[:, 1], 1e-9, 1 - 1e-9)
        ll = float(np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr)))
        k = Xc.shape[1] + 1
        return 2 * k - 2 * ll

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self.model_.predict_proba(X[:, self.selected_])


# --------------------------------------------------------------------------
# 自研封装 2：FT-Transformer（Gorishniy 2021）——表格深度学习的代表架构
# --------------------------------------------------------------------------

class FTTransformerClassifier(BaseEstimator, ClassifierMixin):
    """连续特征的 Feature Tokenizer + Transformer + CLS 头。

    全部 CHAPTER 维度的()部件用 CPU 训练（163 例极小数据，GPU 无收益且不可复现）。
    """

    def __init__(self, dim=32, depth=3, heads=4, attn_dropout=0.1,
                 ff_dropout=0.1, max_epochs=150, patience=20, lr=1e-3,
                 batch_size=32, seed=RANDOM_SEED):
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, X, y):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        # 内部留出集用于早停（20%），训练过小则退化为无早停
        from sklearn.model_selection import train_test_split
        n_tr_int = max(int(0.8 * len(X)), 10)
        use_es = len(X) - n_tr_int >= 10
        if use_es:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=self.seed)
        else:
            X_tr, y_tr = X, y
            X_val = y_val = None

        # 先把超参落到局部变量，避免nn.Module 内部的 self 与估计器 self 混淆
        n_features = X.shape[1]
        dim, depth, heads = self.dim, self.depth, self.heads
        attn_dropout, ff_dropout = self.attn_dropout, self.ff_dropout

        class FTNet(nn.Module):
            """Feature Tokenizer + Transformer + CLS head（CPU 训练）。"""

            def __init__(self):
                super().__init__()
                self.cls_token = nn.Parameter(torch.zeros(dim))
                self.weight = nn.Parameter(torch.ones(n_features, dim) * 0.1)
                self.bias = nn.Parameter(torch.zeros(n_features, dim))
                layer = nn.TransformerEncoderLayer(
                    d_model=dim, nhead=heads, dim_feedforward=dim * 4,
                    dropout=attn_dropout, activation="gelu",
                    batch_first=True, norm_first=True)
                self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
                self.head = nn.Sequential(nn.LayerNorm(dim),
                                          nn.Dropout(ff_dropout),
                                          nn.Linear(dim, 2))

            def forward(self, x):
                tok = x.unsqueeze(-1) * self.weight + self.bias
                cls = self.cls_token.expand(x.shape[0], 1, -1)
                h = torch.cat([cls, tok], dim=1)
                h = self.encoder(h)
                return self.head(h[:, 0])

        model = FTNet()
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss()

        Xt = torch.tensor(X_tr); yt = torch.tensor(y_tr)
        if use_es:
            Xv = torch.tensor(X_val); yv = torch.tensor(y_val)
        bs = min(self.batch_size, len(Xt))
        best_state, best_loss, bad = None, np.inf, 0

        for epoch in range(self.max_epochs):
            model.train()
            perm = torch.randperm(len(Xt), generator=torch.Generator().manual_seed(self.seed + epoch))
            epoch_loss = 0.0
            for i in range(0, len(Xt), bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                logits = model(Xt[idx])
                loss = lossf(logits, yt[idx])
                loss.backward()
                opt.step()
                epoch_loss += float(loss) * len(idx)
            epoch_loss /= len(Xt)

            if use_es:
                model.eval()
                with torch.no_grad():
                    vloss = float(lossf(model(Xv), yv))
                if vloss < best_loss - 1e-6:
                    best_loss, bad = vloss, 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            # 训练损失发散（极小数据上罕见但需兜底）
            if not np.isfinite(epoch_loss):
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.model_ = model
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        import torch
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            logits = self.model_(torch.tensor(X))
            proba = torch.softmax(logits, dim=1).numpy()
        return proba



# --------------------------------------------------------------------------
# 自研封装 3：Super Learner（折内预测训练元模型，天生无泄漏）
# --------------------------------------------------------------------------

class NegativeEnsembleMeta(BaseEstimator, ClassifierMixin):
    """带单纯形约束的非负最小二乘元模型（Super Learner 的经典权重估计）。"""

    def __init__(self, constrain=True):
        self.constrain = constrain

    def fit(self, Z, y):
        from scipy.optimize import nnls
        Z = np.asarray(Z, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        zmax = max(np.abs(Z).max(), 1.0) * 1e3
        if self.constrain:
            Za = np.vstack([Z, np.ones((1, Z.shape[1])) * zmax])
            ya = np.concatenate([y, [zmax]])
            w, _ = nnls(Za, ya)
            if w.sum() > 0:
                w = w / w.sum()
        else:
            w, _ = nnls(Z, y)
            if w.sum() > 0:
                w = w / w.sum()
        if w.sum() <= 0:                      # 兜底：等权
            w = np.ones(Z.shape[1]) / Z.shape[1]
        self.coef_ = w
        self.classes_ = np.unique(np.asarray(y).astype(int))
        return self

    def predict_proba(self, Z):
        Z = np.asarray(Z, dtype=np.float64)
        lin = Z @ self.coef_
        p = np.clip(lin, 1e-6, 1 - 1e-6)     # NNLS 的输出本身就是概率尺度
        return np.column_stack([1 - p, p])


class LeakSafeStacking(BaseEstimator, ClassifierMixin):
    """层级堆叠：基模型用内层 cross_val_predict 产生元特征，再训练元模型。

    整个 fit 只发生在当前的外层训练折内部，因此对外部评估不存在泄漏。
    """

    def __init__(self, base_models, meta_model=None, n_inner_folds=5,
                 seed=RANDOM_SEED):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_inner_folds = n_inner_folds
        self.seed = seed

    def fit(self, X, y):
        cv = StratifiedKFold(self.n_inner_folds, shuffle=True,
                             random_state=self.seed)
        Z = np.zeros((len(X), len(self.base_models)), dtype=np.float64)
        self.fitted_bases_ = []
        for j, key in enumerate(self.base_models):
            est = clone(BASE_MODEL_FACTORY[key]())
            proba = cross_val_predict(est, X, y, cv=cv, method="predict_proba",
                                      n_jobs=1)[:, 1]
            Z[:, j] = proba
            self.fitted_bases_.append(clone(BASE_MODEL_FACTORY[key]()).fit(X, y))
        meta = self.meta_model or NegativeEnsembleMeta()
        self.meta_ = clone(meta).fit(Z, y)
        self.classes_ = np.unique(np.asarray(y).astype(int))
        return self

    def predict_proba(self, X):
        Zt = np.column_stack([b.predict_proba(X)[:, 1] for b in self.fitted_bases_])
        return self.meta_.predict_proba(Zt)


BASE_MODEL_FACTORY = {
    "lr": lambda: LogisticRegression(C=1.0, max_iter=5000),
    "lda": lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    # 参与投票/堆叠的 SVM 必须输出概率：用折内校准而非 probability=True
    # （后者在 sklearn 1.9 已弃用、1.11 将移除）
    "svm": lambda: CalibratedClassifierCV(
        SVC(C=1.0, gamma="scale"), cv=3),
    # 堆叠/投票的基模型刻意取中等规模：它们会在每个外层折内被多次拟合
    # （含内层 cross_val_predict），规模过大时 Stage C 的 LOOCV 会被拖垮。
    "rf": lambda: RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED,
                                         n_jobs=1),
    "lgbm": lambda: LGBMClassifier(n_estimators=150, num_leaves=7,
                                   min_child_samples=10, random_state=RANDOM_SEED,
                                   n_jobs=1, verbose=-1),
    "xgb": lambda: XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.1,
                                 subsample=0.8, random_state=RANDOM_SEED, n_jobs=1,
                                 eval_metric="logloss", verbosity=0),
}

STACK_BASES = ["lr", "lda", "svm", "rf", "lgbm"]
VOTING_BASES = ["lr", "svm", "rf", "lgbm"]


# --------------------------------------------------------------------------
# 模型库主体
# --------------------------------------------------------------------------

def get_ext_models(seed=RANDOM_SEED, light=False, device="cpu") -> dict:
    """返回扩展后的模型字典。light=True 时缩小规模用于冒烟测试。"""
    n_tree = 30 if light else 500
    mlp_iter = 50 if light else 3000

    models = {}

    # ---- 族 1：线性 / 正则化 / 统计判别（临床建模主力）----
    models["Logistic_LASSO_baseline"] = make_baseline_pipeline(seed)
    models["Logistic_LASSO_full"] = _pipe(LogisticRegression(
        C=1.0, solver="saga", l1_ratio=1.0, max_iter=20000, tol=1e-4,
        random_state=seed))
    models["Logistic_Ridge"] = _pipe(LogisticRegression(
        C=1.0, max_iter=5000, random_state=seed))
    models["Logistic_ElasticNet"] = _pipe(LogisticRegression(
        C=1.0, solver="saga", l1_ratio=0.5, max_iter=20000, tol=1e-4,
        random_state=seed))
    models["Stepwise_Logistic_AIC"] = _pipe(StepwiseLogistic(max_k=6, seed=seed))
    models["LDA"] = _pipe(LinearDiscriminantAnalysis())
    models["LDA_Shrinkage"] = _pipe(
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    models["QDA"] = _pipe(QuadraticDiscriminantAnalysis(reg_param=0.1))
    models["GaussianNB"] = _pipe(GaussianNB())
    models["Logistic_Poly2"] = Pipeline([
        ("pre", AutoPreprocessor()),
        ("univar", UnivariateLRFilter(p_threshold=0.05)),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("clf", LogisticRegression(C=0.1, max_iter=5000, random_state=seed)),
    ])

    # ---- 族 2：核方法 / 贝叶斯 / 惰性学习 ----
    models["SVM_linear"] = _pipe(CalibratedClassifierCV(
        LinearSVC(C=0.1, max_iter=20000), cv=3))
    models["SVM_rbf"] = _pipe(CalibratedClassifierCV(
        SVC(C=1.0, gamma="scale", random_state=seed), cv=3))
    models["SVM_poly"] = _pipe(CalibratedClassifierCV(
        SVC(kernel="poly", degree=3, C=1.0, random_state=seed), cv=3))
    models["GaussianProcess_RBF"] = _pipe(GaussianProcessClassifier(
        kernel=1.0 * RBF(1.0), n_restarts_optimizer=2, random_state=seed))
    models["KNN"] = _pipe(KNeighborsClassifier(n_neighbors=15))

    # ---- 族 3：树模型与集成 ----
    models["DecisionTree"] = _pipe(DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=5, random_state=seed))
    models["RandomForest"] = _pipe(RandomForestClassifier(
        n_estimators=n_tree, random_state=seed, n_jobs=1))
    models["ExtraTrees"] = _pipe(ExtraTreesClassifier(
        n_estimators=n_tree, random_state=seed, n_jobs=1))
    models["GradientBoosting"] = _pipe(GradientBoostingClassifier(
        n_estimators=100 if not light else 20, max_depth=2, learning_rate=0.1,
        random_state=seed))
    models["HistGradientBoosting"] = _pipe(HistGradientBoostingClassifier(
        max_iter=100 if not light else 20, learning_rate=0.1,
        max_leaf_nodes=7, random_state=seed))
    models["XGBoost"] = _pipe(XGBClassifier(
        n_estimators=200 if not light else 20, max_depth=2, learning_rate=0.1,
        subsample=0.8, reg_lambda=1.0, eval_metric="logloss",
        random_state=seed, n_jobs=1, verbosity=0))
    models["LightGBM"] = _pipe(LGBMClassifier(
        n_estimators=200 if not light else 20, num_leaves=7,
        min_child_samples=10, random_state=seed, n_jobs=1, verbose=-1))
    models["AdaBoost"] = _pipe(AdaBoostClassifier(
        n_estimators=100 if not light else 20, learning_rate=0.5,
        random_state=seed))
    models["Bagging_Trees"] = _pipe(BaggingClassifier(
        n_estimators=100 if not light else 20, random_state=seed, n_jobs=1))

    # ---- 族 4：深度学习 ----
    models["MLP_Small"] = _pipe(MLPClassifier(
        hidden_layer_sizes=(16, 8), alpha=1.0, max_iter=mlp_iter,
        early_stopping=not light, validation_fraction=0.15,
        random_state=seed))
    models["MLP"] = _pipe(MLPClassifier(
        hidden_layer_sizes=(32, 16), alpha=0.5, max_iter=mlp_iter,
        early_stopping=not light, validation_fraction=0.15,
        random_state=seed))
    models["MLP_Deep"] = _pipe(MLPClassifier(
        hidden_layer_sizes=(64, 32, 16), alpha=1.0, max_iter=mlp_iter,
        early_stopping=not light, validation_fraction=0.15,
        random_state=seed))
    models["TabNet"] = _pipe(TabNetWrapper(
        seed=seed, max_epochs=5 if light else 200, patience=2 if light else 25))
    models["TabNet_Deep"] = _pipe(TabNetWrapper(
        seed=seed, n_d=16, n_a=16, n_steps=5,
        max_epochs=5 if light else 200, patience=2 if light else 25))
    models["FT_Transformer"] = _pipe(FTTransformerClassifier(
        seed=seed, max_epochs=10 if light else 150,
        patience=5 if light else 20))

    # ---- 族 5：集成之集成 ----
    models["Voting_Soft_Hetero"] = VotingClassifier(
        estimators=[(k, _pipe(BASE_MODEL_FACTORY[k]())) for k in VOTING_BASES],
        voting="soft")
    models["Stacking_Logit"] = _pipe(LeakSafeStacking(
        base_models=STACK_BASES,
        meta_model=LogisticRegression(C=1.0, max_iter=5000),
        n_inner_folds=3, seed=seed))
    models["SuperLearner_NNLS"] = _pipe(LeakSafeStacking(
        base_models=STACK_BASES, meta_model=NegativeEnsembleMeta(constrain=True),
        n_inner_folds=3, seed=seed))

    # ---- 族 6：消融 —— 单变量筛选对部分模型的影响 ----
    models["SVM_rbf_screened"] = _screen_pipe(CalibratedClassifierCV(
        SVC(C=1.0, gamma="scale", random_state=seed), cv=3))
    models["XGBoost_screened"] = _screen_pipe(XGBClassifier(
        n_estimators=200 if not light else 20, max_depth=2, learning_rate=0.1,
        subsample=0.8, eval_metric="logloss", random_state=seed, n_jobs=1,
        verbosity=0))
    models["MLP_screened"] = _screen_pipe(MLPClassifier(
        hidden_layer_sizes=(32, 16), alpha=0.5, max_iter=mlp_iter,
        early_stopping=not light, validation_fraction=0.15, random_state=seed))
    models["RandomForest_screened"] = _screen_pipe(RandomForestClassifier(
        n_estimators=n_tree, random_state=seed, n_jobs=1))

    # 家族标签
    families = {
        "Linear & Regularized": ["Logistic_LASSO_baseline", "Logistic_LASSO_full",
                                 "Logistic_Ridge", "Logistic_ElasticNet",
                                 "Stepwise_Logistic_AIC", "LDA", "LDA_Shrinkage",
                                 "QDA", "GaussianNB", "Logistic_Poly2"],
        "Kernel & Bayes": ["SVM_linear", "SVM_rbf", "SVM_poly",
                           "GaussianProcess_RBF", "KNN"],
        "Trees & Ensembles": ["DecisionTree", "RandomForest", "ExtraTrees",
                              "GradientBoosting", "HistGradientBoosting",
                              "XGBoost", "LightGBM", "AdaBoost", "Bagging_Trees"],
        "Deep Learning": ["MLP_Small", "MLP", "MLP_Deep", "TabNet", "TabNet_Deep",
                          "FT_Transformer"],
        "Ensemble of Ensembles": ["Voting_Soft_Hetero", "Stacking_Logit",
                                  "SuperLearner_NNLS"],
        "Ablation (screening)": ["SVM_rbf_screened", "XGBoost_screened",
                                 "MLP_screened", "RandomForest_screened"],
    }
    FAMILY.clear()
    for fam, names in families.items():
        for n in names:
            if n in models:
                FAMILY[n] = fam
    for n in models:
        FAMILY.setdefault(n, "Other")
    return models


def get_ext_grids() -> dict:
    """Stage B 调参网格；键名与 pipeline 步骤名一致。"""
    return {
        "Logistic_Ridge": [{"clf__C": [0.1, 0.5, 1.0, 5.0]}],
        "Logistic_ElasticNet": [{"clf__C": [0.1, 0.5, 1.0],
                                 "clf__l1_ratio": [0.15, 0.5, 0.9]}],
        "Logistic_LASSO_full": [{"clf__C": [0.05, 0.1, 0.5, 1.0]}],
        "LDA_Shrinkage": [{"clf__solver": ["lsqr"],
                           "clf__shrinkage": ["auto", 0.1, 0.5, 0.9]}],
        "Stepwise_Logistic_AIC": [{"clf__max_k": [3, 4, 6]}],
        "SVM_rbf": [{"clf__estimator__C": [0.25, 0.5, 1.0, 2.0, 5.0],
                     "clf__estimator__gamma": ["scale", 0.05, 0.01]}],
        "SVM_linear": [{"clf__estimator__C": [0.01, 0.05, 0.1, 0.5]}],
        "SVM_poly": [{"clf__estimator__C": [0.1, 0.5, 1.0],
                      "clf__estimator__degree": [2, 3]}],
        "KNN": [{"clf__n_neighbors": [5, 10, 15, 25, 35]}],
        "RandomForest": [{"clf__max_features": ["sqrt", "log2"],
                          "clf__min_samples_leaf": [1, 3, 5]}],
        "ExtraTrees": [{"clf__max_features": ["sqrt", "log2"],
                        "clf__min_samples_leaf": [1, 3, 5]}],
        "GradientBoosting": [{"clf__n_estimators": [50, 100, 200],
                              "clf__learning_rate": [0.05, 0.1]}],
        "HistGradientBoosting": [{"clf__max_iter": [50, 100, 200],
                                  "clf__max_leaf_nodes": [3, 7, 15]}],
        "XGBoost": [{"clf__max_depth": [2, 3],
                     "clf__n_estimators": [100, 200, 300],
                     "clf__learning_rate": [0.05, 0.1]}],
        "LightGBM": [{"clf__num_leaves": [3, 7, 15],
                      "clf__min_child_samples": [5, 10, 20],
                      "clf__learning_rate": [0.05, 0.1]}],
        "AdaBoost": [{"clf__n_estimators": [50, 100, 200],
                      "clf__learning_rate": [0.1, 0.5, 1.0]}],
        "Bagging_Trees": [{"clf__n_estimators": [50, 100, 200],
                           "clf__max_features": [0.5, 0.8, 1.0]}],
        "DecisionTree": [{"clf__max_depth": [2, 3, 4, 5],
                          "clf__min_samples_leaf": [3, 5, 10]}],
        "MLP": [{"clf__alpha": [0.1, 0.5, 1.0, 5.0],
                 "clf__hidden_layer_sizes": [(16,), (32, 16), (64, 32)]}],
        "MLP_Small": [{"clf__alpha": [0.5, 1.0, 5.0],
                       "clf__hidden_layer_sizes": [(8,), (16, 8)]}],
        "MLP_Deep": [{"clf__alpha": [0.5, 1.0, 5.0],
                      "clf__hidden_layer_sizes": [(32, 16, 8), (64, 32, 16)]}],
        # 深度学习网格刻意压缩：嵌套 CV 下每个外层折都要跑一遍完整网格，
        # 且它们在本数据上通常不是前排名，投入更多组合收益极低。
        "TabNet": [{"clf__n_steps": [3, 4], "clf__n_d": [8, 16]}],
        "TabNet_Deep": [{"clf__n_steps": [4, 5]}],
        "FT_Transformer": [{"clf__dim": [16, 32], "clf__depth": [2, 3]}],
        "GaussianProcess_RBF": [{"clf__kernel": [1.0 * RBF(1.0), 1.0 * RBF(0.5),
                                                 1.0 * RBF(2.0)]}],
    }

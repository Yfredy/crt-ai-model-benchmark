import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class AutoPreprocessor(BaseEstimator, TransformerMixin):
    """逐列预处理：二值列(取值⊆{0,1})中位数填补后原样保留；其余列中位数填补+标准化。

    fit 只发生在训练折上，输出形状与输入一致、列顺序不变。
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

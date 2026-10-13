import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, norm

from analysis.config import TARGET_COL
from analysis.data_loader import column_kinds


def irls_logistic(X, y, max_iter=100, tol=1e-10):
    """带截距的无惩罚 logistic 回归（Newton–Raphson / IRLS）。

    返回 dict(coef=[截距, *斜率], se, z, p, ci_lo, ci_hi)，同一尺度。
    Fisher 信息奇异或权重退化（完全分离）时抛 ValueError。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, k = X.shape
    A = np.column_stack([np.ones(n), X])
    ybar = float(np.clip(y.mean(), 1e-9, 1 - 1e-9))
    beta = np.zeros(k + 1)
    beta[0] = np.log(ybar / (1 - ybar))
    converged = False
    for _ in range(max_iter):
        mu = 1.0 / (1.0 + np.exp(-np.clip(A @ beta, -30, 30)))
        W = mu * (1 - mu)
        if np.any(W <= 1e-12):
            raise ValueError("degenerate weights (perfect separation?)")
        grad = A.T @ (y - mu)
        H = A.T @ (A * W[:, None]) + 1e-10 * np.eye(k + 1)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"singular Fisher information: {e}") from e
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    if not converged:
        raise ValueError("IRLS did not converge")
    mu = 1.0 / (1.0 + np.exp(-np.clip(A @ beta, -30, 30)))
    W = mu * (1 - mu)
    cov = np.linalg.inv(A.T @ (A * W[:, None]) + 1e-10 * np.eye(k + 1))
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    z = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    return {"coef": beta, "se": se, "z": z, "p": 2 * norm.sf(np.abs(z)),
            "ci_lo": beta - norm.ppf(0.975) * se,
            "ci_hi": beta + norm.ppf(0.975) * se}


def univariate_lr_table(df, feature_names=None, p_threshold=0.05) -> pd.DataFrame:
    """复现已发表论文的单变量 logistic 筛选（Table 2）。

    连续变量按 1 个标准差报告 OR，二值变量直接报告 OR。
    不可估计的变量记 NaN 并在 note 列说明，不中断整表。
    """
    df = df.reset_index(drop=True)
    names = feature_names or [c for c in df.columns if c != TARGET_COL]
    y = df[TARGET_COL].to_numpy(dtype=float)
    kinds = column_kinds(df)
    rows = []
    for c in names:
        x = df[c].to_numpy(dtype=float)
        sd = float(np.nanstd(x))
        if sd == 0 or np.isnan(sd):
            rows.append({"variable": c, "n": int(len(x)), "or_per_sd": np.nan,
                         "ci_lo": np.nan, "ci_hi": np.nan, "p_wald": np.nan,
                         "selected": False, "note": "constant column"})
            continue
        is_binary = c in kinds["binary"]
        xs = x if is_binary else (x - float(np.nanmean(x))) / sd
        xs = np.where(np.isnan(xs), np.nanmedian(xs), xs)
        try:
            fit = irls_logistic(xs.reshape(-1, 1), y)
            rows.append({"variable": c, "n": int(len(x)),
                         "or_per_sd": float(np.exp(fit["coef"][1])),
                         "ci_lo": float(np.exp(fit["ci_lo"][1])),
                         "ci_hi": float(np.exp(fit["ci_hi"][1])),
                         "p_wald": float(fit["p"][1]),
                         "selected": bool(fit["p"][1] < p_threshold),
                         "note": "binary (OR)" if is_binary else "OR per 1 SD"})
        except ValueError as e:
            rows.append({"variable": c, "n": int(len(x)), "or_per_sd": np.nan,
                         "ci_lo": np.nan, "ci_hi": np.nan, "p_wald": np.nan,
                         "selected": False, "note": f"not estimable: {e}"})
    return (pd.DataFrame(rows).sort_values("p_wald", na_position="last")
            .reset_index(drop=True))


def _median_iqr(s) -> str:
    s = pd.Series(s).dropna()
    if s.empty:
        return "NA"
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    return f"{med:.2f} ({q1:.2f}-{q3:.2f})"


def _smd(a, b) -> float:
    a = pd.Series(a).dropna().to_numpy(dtype=float)
    b = pd.Series(b).dropna().to_numpy(dtype=float)
    if a.size < 2 or b.size < 2:
        return np.nan
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def _n_pct(s) -> str:
    s = pd.Series(s).dropna()
    if s.empty:
        return "NA"
    return f"{int(s.sum())}/{len(s)} ({s.mean() * 100:.1f}%)"


def table1_by_group(df, group_col=TARGET_COL) -> pd.DataFrame:
    """稿件 Table 1：应答 vs 非应答的基线特征比较。

    连续变量：中位数 (IQR) + Mann–Whitney U；二值变量：n (%) + Fisher 精确检验。
    另附标准化均差 SMD（|SMD| > 0.1 视为有意义的组间差异）。
    """
    df = df.reset_index(drop=True)
    if not df[group_col].isin([0, 1]).all():
        raise ValueError(f"group column '{group_col}' must be binary 0/1")
    kinds = column_kinds(df)
    g1, g0 = df[df[group_col] == 1], df[df[group_col] == 0]
    rows = []
    for c in [x for x in df.columns if x != group_col]:
        v = df[c]
        if c in kinds["binary"]:
            b1, b0 = g1[c].dropna(), g0[c].dropna()
            n1, n0 = int(b1.sum()), int(b0.sum())
            try:
                _, p = fisher_exact([[n1, len(b1) - n1], [n0, len(b0) - n0]])
            except ValueError:
                p = np.nan
            rows.append({"variable": c, "type": "binary", "overall": _n_pct(v),
                         "responder": _n_pct(b1), "non_responder": _n_pct(b0),
                         "p_value": float(p), "smd": _smd(g1[c], g0[c])})
        else:
            a, b = g1[c].dropna(), g0[c].dropna()
            try:
                _, p = mannwhitneyu(a, b, alternative="two-sided") if a.size and b.size \
                    else (np.nan, np.nan)
            except ValueError:  # 退化列（如两组取值完全相同）
                p = np.nan
            rows.append({"variable": c, "type": "continuous", "overall": _median_iqr(v),
                         "responder": _median_iqr(g1[c]),
                         "non_responder": _median_iqr(g0[c]),
                         "p_value": float(p), "smd": _smd(g1[c], g0[c])})
    return pd.DataFrame(rows)

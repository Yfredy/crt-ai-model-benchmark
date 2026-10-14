"""多模型整体比较的统计检验。

文献依据（Demšar 2006；Hothorn et al. 2005）：当比较对象超过两个模型时，
逐对做 Wilcoxon 会放大一类错误，规范做法是先做 Friedman 总括检验（把所有
重采样块视为区组、模型视为处理），若拒绝原假设再做 Nemenyi 事后检验，用
临界差 CD 判断两两差异是否跨过显著性门槛。

本模块同时提供多种子秩稳定性分析，用来区分"真实性能差异"与"随机种子噪声"。
"""
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, norm

from analysis.evaluation import holm_adjust


def average_ranks(block_df: pd.DataFrame, higher_is_better=True):
    """block_df: 行=重采样块（如 repeat×fold），列=模型，值=指标。

    在每个块内独立排名（1=最好），返回平均秩 Series。
    """
    ranks = block_df.rank(axis=1, ascending=not higher_is_better, method="average")
    return ranks.mean(axis=0).sort_values()


def friedman_test(block_df: pd.DataFrame, higher_is_better=True):
    """Friedman 总括检验。

    返回 dict(statistic, p_value, n_blocks, n_models, mean_ranks)。
    要求：无缺失值，且块数 ≥ 2、模型数 ≥ 3。
    """
    mat = block_df.dropna(axis=0, how="any").to_numpy(dtype=np.float64)
    n_blocks, n_models = mat.shape
    if n_blocks < 2 or n_models < 3:
        raise ValueError("Friedman test needs >= 2 blocks and >= 3 models")
    ranks = block_df.dropna(axis=0, how="any").rank(
        axis=1, ascending=not higher_is_better, method="average").to_numpy()
    # 用秩作为 Friedman 的输入（等价于以秩代替原值）
    stat, p = friedmanchisquare(*[ranks[:, j] for j in range(n_models)])
    return {
        "statistic": float(stat), "p_value": float(p),
        "n_blocks": int(n_blocks), "n_models": int(n_models),
        "mean_ranks": pd.Series(ranks.mean(axis=0), index=block_df.columns)
                        .sort_values(),
    }


def _q_critical(k, alpha=0.05):
    """Nemenyi 临界值 q_alpha(k, ∞)：studentized range 的分位数。

    优先用 SciPy 精确实现，不可用时退到 Harter-table 的常用近似。
    """
    try:
        from scipy.stats import studentized_range
        return float(studentized_range.ppf(1 - alpha, k, np.inf))
    except Exception:  # pragma: no cover - 旧版 SciPy
        table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
                 8: 3.031, 9: 3.102, 10: 3.164, 15: 3.394, 20: 3.542,
                 30: 3.779, 40: 3.940}
        keys = sorted(table)
        if k in table:
            return table[k]
        if k < keys[0]:
            return table[keys[0]]
        if k > keys[-1]:
            return table[keys[-1]]
        for a, b in zip(keys, keys[1:]):
            if a < k < b:
                w = (k - a) / (b - a)
                return table[a] * (1 - w) + table[b] * w
        return table[keys[-1]]


def nemenyi_critical_difference(n_models, n_blocks, alpha=0.05):
    """临界差 CD = q_alpha · sqrt(k(k+1)/(6N))。"""
    q = _q_critical(n_models, alpha=alpha)
    return float(q * np.sqrt(n_models * (n_models + 1) / (6.0 * n_blocks)))


def nemenyi_vs_control(block_df: pd.DataFrame, control, higher_is_better=True,
                       alpha=0.05):
    """以指定模型（如已发表基线）为对照的 Nemenyi 事后检验。

    返回 DataFrame(model, mean_rank, rank_diff, p_nemenyi, significant)。
    """
    res = friedman_test(block_df, higher_is_better=higher_is_better)
    k, n = res["n_models"], res["n_blocks"]
    ranks = res["mean_ranks"]
    se = np.sqrt(k * (k + 1) / (6.0 * n))
    cd = nemenyi_critical_difference(k, n, alpha=alpha)
    rows = []
    ctrl_rank = float(ranks.get(control, np.nan))
    for model, rank in ranks.items():
        if model == control:
            continue
        z = (rank - ctrl_rank) / se if se > 0 else 0.0
        p = float(2 * (1 - norm.cdf(abs(z))))
        rows.append({"model": model, "mean_rank": float(rank),
                     "rank_diff_vs_control": float(rank - ctrl_rank),
                     "z_nemenyi": z, "p_nemenyi": min(p, 1.0),
                     "p_holm": np.nan, "significant_vs_control": np.nan,
                     "direction": np.nan})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm_adjust(out["p_nemenyi"].to_numpy())
        out["significant_vs_control"] = out["p_holm"] < alpha
        out["direction"] = np.where(out["rank_diff_vs_control"] < 0,
                                    "better", "worse")
    # 附对照行：读者需要知道基线自己的平均秩
    ctrl_row = pd.DataFrame([{
        "model": control, "mean_rank": ctrl_rank, "rank_diff_vs_control": 0.0,
        "z_nemenyi": 0.0, "p_nemenyi": 1.0, "p_holm": 1.0,
        "significant_vs_control": False, "direction": "control"}])
    out = pd.concat([ctrl_row, out], ignore_index=True)
    return out.sort_values("mean_rank").reset_index(drop=True), cd


def merge_tests(block_df, control, higher_is_better=True, alpha=0.05):
    """一次给出总括 + 事后检验，并把 CD 挂在结果上便于绘图。"""
    post, cd = nemenyi_vs_control(block_df, control, higher_is_better, alpha)
    fried = friedman_test(block_df, higher_is_better)
    post.attrs["friedman"] = {k: v for k, v in fried.items()
                              if k != "mean_ranks"}
    post.attrs["critical_difference"] = cd
    post.attrs["n_models"] = fried["n_models"]
    post.attrs["n_blocks"] = fried["n_blocks"]
    return post, fried


def rank_stability(seed_rank_df: pd.DataFrame):
    """多种子稳定性：输入 行=种子、列=模型 的秩（或性能值）矩阵。

    返回 DataFrame(model, mean_rank, sd_rank, best_count, ever_top3)，
    用于判断"最优模型"是稳定结论还是种子运气。
    """
    mean_rank = seed_rank_df.mean(axis=0)
    sd_rank = seed_rank_df.std(axis=0)
    top3 = seed_rank_df.apply(
        lambda row: [c for c in seed_rank_df.columns if row[c] <= 3], axis=1)
    ever_top3 = {}
    for lst in top3:
        for c in lst:
            ever_top3[c] = ever_top3.get(c, 0) + 1
    best = seed_rank_df.idxmin(axis=1).value_counts()
    out = pd.DataFrame({
        "model": mean_rank.index,
        "mean_rank": mean_rank.to_numpy(),
        "sd_rank": sd_rank.to_numpy(),
        "times_best": [int(best.get(m, 0)) for m in mean_rank.index],
        "times_top3": [int(ever_top3.get(m, 0)) for m in mean_rank.index],
    }).sort_values("mean_rank").reset_index(drop=True)
    return out

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

"""单步分类指标（设计文档 §5.5.1），只用 numpy。y=1 表示 BLOCK（正类），s 是风险分，越大越像 BLOCK。"""
import numpy as np


def _rankdata(a: np.ndarray) -> np.ndarray:
    """平均秩（并列取平均），等价于 scipy.stats.rankdata。"""
    ranks = np.empty(len(a))
    ranks[a.argsort(kind="mergesort")] = np.arange(1, len(a) + 1)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    return (np.bincount(inv, weights=ranks) / counts)[inv]


def auroc(y, s) -> float:
    y, s = np.asarray(y), np.asarray(s, float)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(y, s) -> float:
    """平均精度（average precision）：按分数从高到低，在每个正样本处取精度再平均。"""
    y, s = np.asarray(y), np.asarray(s, float)
    if y.sum() == 0:
        return float("nan")
    y_sorted = y[np.argsort(-s, kind="mergesort")]
    precision = np.cumsum(y_sorted) / np.arange(1, len(y) + 1)
    return float(precision[y_sorted == 1].mean())


def threshold_at_fpr(y, s, fpr: float) -> float:
    """按目标误报率选阈值（§4.2）：s >= 阈值判 BLOCK，负样本里判成 BLOCK 的不超过 fpr，在此前提下阈值尽量低。"""
    y, s = np.asarray(y), np.asarray(s, float)
    neg = np.sort(s[y == 0])[::-1]
    k = int(np.floor(fpr * len(neg)))  # 最多允许 k 个负样本判成 BLOCK
    return float(np.nextafter(neg[k], np.inf)) if k < len(neg) else -np.inf  # 刚好高过第 k+1 高的负样本


def recall_at_fpr(y, s, fpr: float) -> float:
    """误报率不超过 fpr 时能拦住的正样本比例。"""
    y, s = np.asarray(y), np.asarray(s, float)
    if (y == 0).sum() == 0 or (y == 1).sum() == 0:
        return float("nan")
    return float((s[y == 1] >= threshold_at_fpr(y, s, fpr)).mean())


def at_threshold(y, s, t: float) -> dict:
    y, pred = np.asarray(y), np.asarray(s, float) >= t
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else float("nan")
    return {"precision": precision, "recall": recall, "fpr": fp / (fp + tn) if fp + tn else float("nan"),
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def bootstrap_ci(fn, y, s, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """对样本有放回重抽 n 次，返回 fn 的 95% 置信区间（2.5 / 97.5 分位）。"""
    y, s = np.asarray(y), np.asarray(s, float)
    rng = np.random.default_rng(seed)
    vals = [fn(y[idx], s[idx]) for idx in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    vals = np.array([v for v in vals if not np.isnan(v)])
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) else (float("nan"),) * 2


def report(y, s, threshold: float = 0.5, n_boot: int = 1000) -> dict:
    """§5.5.1 的整套指标，每项带 bootstrap 95% 置信区间。"""
    fns = {"auroc": auroc, "auprc": auprc,
           "recall@fpr1%": lambda y, s: recall_at_fpr(y, s, 0.01),
           "recall@fpr5%": lambda y, s: recall_at_fpr(y, s, 0.05),
           f"f1@{threshold}": lambda y, s: at_threshold(y, s, threshold)["f1"],
           f"fpr@{threshold}": lambda y, s: at_threshold(y, s, threshold)["fpr"],
           f"recall@{threshold}": lambda y, s: at_threshold(y, s, threshold)["recall"]}
    y, s = np.asarray(y), np.asarray(s, float)
    out = {"n": len(y), "n_block": int(y.sum()), "n_allow": int((1 - y).sum())}
    for name, fn in fns.items():
        out[name] = {"value": fn(y, s), "ci95": bootstrap_ci(fn, y, s, n_boot)}
    return out

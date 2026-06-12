"""Mention-calibration metric (Brier) and a leakage-free date split.

G1' replaces the mis-specified next-token G1: for a vocabulary of market terms we
predict P(mention) for a held-out event and score calibration against the realized
YES/NO outcomes. Calibration (not raw accuracy) is what makes Kelly sizing sane.
"""
from __future__ import annotations

import math


def brier_score(predictions: list[float], outcomes: list[int]) -> float:
    if not predictions:
        return 0.0
    return sum((p - o) ** 2 for p, o in zip(predictions, outcomes)) / len(predictions)


def calibration_report(pred_by_term: dict[str, float], outcome_by_term: dict[str, int]) -> dict:
    terms = sorted(pred_by_term)
    preds = [pred_by_term[t] for t in terms]
    outs = [outcome_by_term[t] for t in terms]
    return {
        "n": len(terms),
        "brier": brier_score(preds, outs),
        "terms": terms,
        "predictions": preds,
        "outcomes": outs,
    }


def split_before(docs: list[dict], cutoff_date: str) -> tuple[list[dict], list[dict]]:
    """train = docs strictly before cutoff_date; test = docs on/after. ISO dates sort lexically."""
    train = [d for d in docs if d.get("date", "") < cutoff_date]
    test = [d for d in docs if d.get("date", "") >= cutoff_date]
    return train, test


def log_loss(predictions: list[float], outcomes: list[int], eps: float = 1e-15) -> float:
    """Mean binary cross-entropy. Predictions clamped to [eps, 1-eps] to avoid log(0)."""
    if not predictions:
        return 0.0
    total = 0.0
    for p, o in zip(predictions, outcomes):
        p = min(max(p, eps), 1.0 - eps)
        total += -(o * math.log(p) + (1 - o) * math.log(1.0 - p))
    return total / len(predictions)


def auc(predictions: list[float], outcomes: list[int]) -> float | None:
    """Rank-based ROC AUC (Mann-Whitney). Returns None if only one outcome class is present."""
    n_pos = sum(1 for o in outcomes if o == 1)
    n_neg = len(outcomes) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    paired = sorted(zip(predictions, outcomes), key=lambda x: x[0])
    ranks = [0.0] * len(paired)
    i = 0
    while i < len(paired):
        j = i
        while j < len(paired) and paired[j][0] == paired[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0  # 1-based average rank for ties
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    sum_pos_ranks = sum(r for r, (_, o) in zip(ranks, paired) if o == 1)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def reliability_diagram(predictions: list[float], outcomes: list[int], n_bins: int = 10) -> list[dict]:
    """Bin predictions into n_bins equal-width buckets; report mean prediction vs observed frequency."""
    bins = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [
            i for i, p in enumerate(predictions)
            if (lo <= p < hi) or (b == n_bins - 1 and p == hi)  # last bin includes 1.0
        ]
        if idx:
            mean_pred = sum(predictions[i] for i in idx) / len(idx)
            observed = sum(outcomes[i] for i in idx) / len(idx)
        else:
            mean_pred = observed = None
        bins.append({"bin": b, "lo": lo, "hi": hi, "count": len(idx),
                     "mean_pred": mean_pred, "observed": observed})
    return bins


def calibration_summary(pool: list[tuple[float, int]]) -> dict:
    """Reduce a flat pool of (pred, outcome) pairs to all Level-1 metrics."""
    preds = [p for p, _ in pool]
    outs = [o for _, o in pool]
    n = len(pool)
    return {
        "n": n,
        "base_rate": (sum(outs) / n) if n else 0.0,
        "brier": brier_score(preds, outs),
        "log_loss": log_loss(preds, outs),
        "auc": auc(preds, outs),
        "reliability": reliability_diagram(preds, outs),
    }

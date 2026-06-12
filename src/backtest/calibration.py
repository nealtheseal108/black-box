"""Mention-calibration metric (Brier) and a leakage-free date split.

G1' replaces the mis-specified next-token G1: for a vocabulary of market terms we
predict P(mention) for a held-out event and score calibration against the realized
YES/NO outcomes. Calibration (not raw accuracy) is what makes Kelly sizing sane.
"""
from __future__ import annotations


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

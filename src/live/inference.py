"""Bayesian inference core for the live predictor (C6).

Inference-only — no weight updates. Posterior recomputed from fixed prior +
cumulative diction log-likelihood each chunk (Appendix A.3).
"""
import math

_EPS = 1e-6
_OPPOSITE = {"hawkish": "dovish", "dovish": "hawkish"}


def to_logodds(p: float) -> float:
    p = min(1 - _EPS, max(_EPS, p))      # clamp away from 0/1 so log-odds is finite
    return math.log(p / (1 - p))


def from_logodds(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def bayesian_update(prior_prob: float, loglik_delta: float) -> float:
    """Inference-only posterior: prior log-odds shifted by accumulated diction evidence."""
    return from_logodds(to_logodds(prior_prob) + loglik_delta)


def diction_loglikelihood(phrase_signals: list[dict], market_axis: str, scale: float = 1.0) -> float:
    """Sum signal weights: same-axis adds, opposite-axis subtracts, unrelated axes neutral."""
    total = 0.0
    opp = _OPPOSITE.get(market_axis)
    for s in phrase_signals:
        if s["axis"] == market_axis:
            total += s["weight"]
        elif opp is not None and s["axis"] == opp:
            total -= s["weight"]
    return total * scale


def chunk_words(words, every: int = 5):
    buf = []
    for w in words:
        buf.append(w)
        if len(buf) == every:
            yield buf
            buf = []
    if buf:
        yield buf

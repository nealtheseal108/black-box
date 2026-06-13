"""Frozen corpus co-occurrence weights for live Bayesian propagation.

w(A->B) = clamp(log(P(B|A) / P(B)), -w_max, +w_max): the log-odds shift in term B's
probability when evidence term A appears, estimated at the document level. Frozen
(computed once from the corpus) — inference-only, never updated at runtime (A.3).
The clamp bounds a flukey pairing (two terms sharing a few of 29 docs look perfectly
correlated) from yanking a live position.
"""
from __future__ import annotations

import math

from src.mentions.terms import MarketTerm

_EPS = 1e-9


def cooccurrence_weights(
    docs: list[dict],
    evidence_terms: list[MarketTerm],
    target_terms: list[MarketTerm],
    w_max: float = 2.0,
) -> dict[tuple[str, str], float]:
    n = len(docs)
    appears: dict[str, set[int]] = {}
    for t in {*evidence_terms, *target_terms}:
        appears[t.canonical] = {i for i, d in enumerate(docs) if t.mentioned_in(d["text"])}

    weights: dict[tuple[str, str], float] = {}
    for a in evidence_terms:
        a_docs = appears[a.canonical]
        if not a_docs:
            continue
        for b in target_terms:
            if b.canonical == a.canonical:
                continue
            p_b = (len(appears[b.canonical]) + _EPS) / (n + _EPS)
            p_b_given_a = (len(appears[b.canonical] & a_docs) + _EPS) / (len(a_docs) + _EPS)
            raw = math.log(p_b_given_a / p_b)
            weights[(a.canonical, b.canonical)] = max(-w_max, min(w_max, raw))
    return weights

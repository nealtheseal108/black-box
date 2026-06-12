"""Phase-conditioned mention base rates from the Warsh corpus.

Prepared-statement diction (scripted monologue) and Q&A diction (unscripted
dialogue) behave differently, so we estimate each separately and combine with a
noisy-OR — a term resolves YES if it is said in EITHER phase:

    P_event = 1 - (1 - P_prep) * (1 - P_qa)

Add-k (Jeffreys, k=0.5) smoothing gives every term a nonzero floor, so a term
Warsh has never said on record is not trapped at 0 (the bounded news multiplier
can then lift it).
"""
from __future__ import annotations

from src.mentions.terms import MarketTerm

# context_type -> phase. Scripted monologue vs. unscripted dialogue.
PREP_TYPES = frozenset({"speech", "essay", "op_ed", "lecture", "testimony"})
QA_TYPES = frozenset({"interview", "hearing"})


def _phase_docs(docs: list[dict], phase_types: frozenset[str]) -> list[dict]:
    return [d for d in docs if d.get("context_type") in phase_types]


def phase_base_rate(docs: list[dict], term: MarketTerm, phase_types: frozenset[str], k: float = 0.5) -> float:
    """Add-k smoothed fraction of `phase_types` docs that mention `term`."""
    pool = _phase_docs(docs, phase_types)
    n = len(pool)
    hits = sum(1 for d in pool if term.mentioned_in(d["text"]))
    return (hits + k) / (n + 2 * k)


def combine_phases(p_prep: float, p_qa: float) -> float:
    """Noisy-OR: probability the term appears in EITHER phase."""
    return 1.0 - (1.0 - p_prep) * (1.0 - p_qa)


def event_base_rate(docs: list[dict], term: MarketTerm, k: float = 0.5) -> dict:
    p_prep = phase_base_rate(docs, term, PREP_TYPES, k)
    p_qa = phase_base_rate(docs, term, QA_TYPES, k)
    return {"p_prep": p_prep, "p_qa": p_qa, "p_event": combine_phases(p_prep, p_qa)}

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

from datetime import date

from src.mentions.terms import MarketTerm

# context_type -> phase. Scripted monologue vs. unscripted dialogue.
PREP_TYPES = frozenset({"speech", "essay", "op_ed", "lecture", "testimony"})
QA_TYPES = frozenset({"interview", "hearing"})

DEFAULT_HALF_LIFE_YEARS = 4.0


def recency_weight(doc_date: str, as_of: str, half_life_years: float = DEFAULT_HALF_LIFE_YEARS) -> float:
    """Exponential time-decay weight: halves every `half_life_years` of age.

    Age is measured from `as_of` (the event date). Docs dated on/after the event
    are clamped to weight 1.0. This is how we stop crisis-era (2008-09) vocabulary
    from dominating a base rate meant to reflect the speaker's *current* diction.
    """
    if not doc_date:
        return 1.0
    age_years = max(0.0, (date.fromisoformat(as_of) - date.fromisoformat(doc_date)).days / 365.25)
    return 0.5 ** (age_years / half_life_years)


def phase_base_rate(
    docs: list[dict],
    term: MarketTerm,
    phase_types: frozenset[str],
    k: float = 0.5,
    as_of: str | None = None,
    half_life_years: float = DEFAULT_HALF_LIFE_YEARS,
) -> float:
    """Add-k smoothed (optionally recency-weighted) fraction of `phase_types` docs that mention `term`.

    With `as_of=None` this is the plain document fraction. With `as_of` set, each
    doc contributes its recency weight to both the numerator and denominator, so
    recent docs count more — the weighted add-k estimate.
    """
    pool = _phase_docs(docs, phase_types)
    if as_of is None:
        weights = [1.0] * len(pool)
    else:
        weights = [recency_weight(d.get("date", ""), as_of, half_life_years) for d in pool]
    weighted_n = sum(weights)
    weighted_hits = sum(w for d, w in zip(pool, weights) if term.mentioned_in(d["text"]))
    return (weighted_hits + k) / (weighted_n + 2 * k)


def _phase_docs(docs: list[dict], phase_types: frozenset[str]) -> list[dict]:
    return [d for d in docs if d.get("context_type") in phase_types]


def combine_phases(p_prep: float, p_qa: float) -> float:
    """Noisy-OR: probability the term appears in EITHER phase."""
    return 1.0 - (1.0 - p_prep) * (1.0 - p_qa)


def event_base_rate(
    docs: list[dict],
    term: MarketTerm,
    k: float = 0.5,
    as_of: str | None = None,
    half_life_years: float = DEFAULT_HALF_LIFE_YEARS,
) -> dict:
    p_prep = phase_base_rate(docs, term, PREP_TYPES, k, as_of, half_life_years)
    p_qa = phase_base_rate(docs, term, QA_TYPES, k, as_of, half_life_years)
    return {"p_prep": p_prep, "p_qa": p_qa, "p_event": combine_phases(p_prep, p_qa)}

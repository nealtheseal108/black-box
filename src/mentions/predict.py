"""Mode-1 (pre-speech) prior assembly.

For each market term: phase-decomposed corpus base rate -> bounded news multiplier
-> P_prior(mention). This is the number traded BEFORE the speech, and the input the
calibration gate scores.
"""
from __future__ import annotations

from src.mentions.terms import MarketTerm
from src.mentions.base_rate import event_base_rate, DEFAULT_HALF_LIFE_YEARS
from src.mentions.news_adjuster import NewsAdjuster, apply_multipliers, clamp_multiplier


def predict_event_priors(
    docs: list[dict],
    terms: list[MarketTerm],
    news_context: str,
    adjuster: NewsAdjuster,
    k: float = 0.5,
    as_of: str | None = None,
    half_life_years: float = DEFAULT_HALF_LIFE_YEARS,
) -> dict[str, dict]:
    base = {t.canonical: event_base_rate(docs, t, k, as_of, half_life_years) for t in terms}
    base_event = {c: v["p_event"] for c, v in base.items()}

    raw_mults = adjuster.multipliers([t.canonical for t in terms], news_context)
    adjusted = apply_multipliers(base_event, raw_mults)

    out: dict[str, dict] = {}
    for t in terms:
        c = t.canonical
        out[c] = {
            "p_prep": base[c]["p_prep"],
            "p_qa": base[c]["p_qa"],
            "p_event_base": base[c]["p_event"],
            "multiplier": clamp_multiplier(raw_mults.get(c, 1.0)),
            "p_prior": adjusted[c],
        }
    return out

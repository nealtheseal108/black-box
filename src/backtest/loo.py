"""Leave-one-event-out engine: reduce a speaker's dated corpus to a flat
(prediction, outcome) pool for Level-1 calibration scoring.

For each held-out doc, the Mode-1 predictor is built from ONLY that speaker's
strictly-earlier docs (temporal split, no leakage), recency-weighted to the event
date. Events with too few prior docs are skipped (cold-start guard).
"""
from __future__ import annotations

from src.mentions.news_adjuster import NullAdjuster
from src.mentions.predict import predict_event_priors
from src.backtest.events import Event


def leave_one_out(
    docs: list[dict],
    speaker: str,
    vocab,
    min_prior_docs: int = 3,
    half_life_years: float = 4.0,
) -> dict:
    docs_sorted = sorted(docs, key=lambda d: d.get("date", ""))
    pool: list[tuple[float, int]] = []
    per_event: list[dict] = []
    n_scored = n_skipped = 0

    for held in docs_sorted:
        date = held.get("date", "")
        text = held.get("text", "")
        if not date or not text:
            n_skipped += 1
            continue
        prior = [d for d in docs_sorted if d.get("date", "") < date]
        if len(prior) < min_prior_docs:
            n_skipped += 1
            continue
        event = Event(speaker=speaker, date=date, text=text,
                      context_type=held.get("context_type", ""))
        terms = vocab.terms_for(event)
        if not terms:
            n_skipped += 1
            continue
        priors = predict_event_priors(
            prior, terms, news_context="", adjuster=NullAdjuster(),
            as_of=date, half_life_years=half_life_years,
        )
        for t in terms:
            pred = priors[t.canonical]["p_prior"]
            outcome = int(t.mentioned_in(event.text))
            pool.append((pred, outcome))
        n_scored += 1
        per_event.append({"date": date, "context_type": event.context_type, "n_terms": len(terms)})

    return {"pool": pool, "n_scored": n_scored, "n_skipped": n_skipped, "per_event": per_event}

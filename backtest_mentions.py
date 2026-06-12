"""G1' calibration backtest — Mode-1 mention predictor vs. the resolved April-21 hearing.

The confirmation hearing is a RESOLVED mentions market and we hold Warsh's cleaned
transcript. We train base rates strictly on pre-hearing docs (no leakage), predict
P(mention) for the market terms, resolve the actual outcomes from the hearing text,
and score Brier calibration + the G1' gate.

Usage: python backtest_mentions.py
"""
from __future__ import annotations

import json
from pathlib import Path

from src.mentions.terms import MarketTerm, load_terms
from src.mentions.news_adjuster import NullAdjuster
from src.mentions.predict import predict_event_priors
from src.backtest.calibration import calibration_report, split_before
from src.backtest.gates import evaluate_mention_gate

CORPUS = Path("corpus/warsh_corpus.jsonl")
TERMS_FIXTURE = Path("corpus/market_terms/hearing_2026-04-21.json")
HEARING_DATE = "2026-04-21"


def _load_corpus() -> list[dict]:
    return [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]


def resolve_outcomes(terms: list[MarketTerm], event_text: str) -> dict[str, int]:
    """Ground truth: 1 if the term's pattern matches the event transcript, else 0."""
    return {t.canonical: int(t.mentioned_in(event_text)) for t in terms}


def run_calibration(as_of: str | None = HEARING_DATE) -> dict:
    """Backtest the Mode-1 predictor against the resolved hearing.

    `as_of` recency-weights the base rate to the event date (default: the hearing
    date). Pass `as_of=None` for the unweighted estimate (used for A/B comparison).
    """
    docs = _load_corpus()
    terms = load_terms(TERMS_FIXTURE)
    train, test = split_before(docs, HEARING_DATE)
    event_text = " ".join(d["text"] for d in test)

    priors = predict_event_priors(train, terms, news_context="", adjuster=NullAdjuster(), as_of=as_of)
    pred_by_term = {c: row["p_prior"] for c, row in priors.items()}
    outcome_by_term = resolve_outcomes(terms, event_text)

    report = calibration_report(pred_by_term, outcome_by_term)
    gate = evaluate_mention_gate(report["brier"])
    return {
        "report": report,
        "gate": gate,
        "priors": priors,
        "outcomes": outcome_by_term,
        "train_docs": len(train),
        "test_event_words": len(event_text.split()),
    }


def main() -> None:
    r = run_calibration()
    print(f"train docs (pre-{HEARING_DATE}): {r['train_docs']}   test event words: {r['test_event_words']}")
    print(f"{'term':<22} {'P_prior':>8} {'outcome':>8}")
    for t in r["report"]["terms"]:
        print(f"{t:<22} {r['priors'][t]['p_prior']:>8.3f} {r['outcomes'][t]:>8d}")
    print(f"\nBrier: {r['report']['brier']:.4f}   G1' gate (< 0.25): "
          f"{'PASS' if r['gate']['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()

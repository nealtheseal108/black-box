"""Live Mode-2 runner.

replay_hearing(): offline proof — streams the resolved April-21 hearing transcript
through the full tracker and returns final per-term probabilities. The live entry
point (Fed StreamText feed) reuses the same tracker with StreamTextClient instead
of ReplayStream; paper trading + episode logging wire in here.

Usage: python live_run.py   # runs the hearing replay and prints resolutions
"""
from __future__ import annotations

import json
from pathlib import Path

from src.mentions.terms import load_terms
from src.mentions.news_adjuster import NullAdjuster
from src.mentions.predict import predict_event_priors
from src.live.cooccurrence import cooccurrence_weights
from src.live.tracker import LiveMentionTracker
from src.live.streamtext import ReplayStream

CORPUS = Path("corpus/warsh_corpus.jsonl")
LEXICON = Path("corpus/lexicon/fed_macro_terms.json")
HEARING = Path("corpus/manual/warsh-confirmation-hearing-2026-04-21.txt")
HEARING_DATE = "2026-04-21"


def _load_corpus() -> list[dict]:
    return [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]


def _hearing_text() -> str:
    raw = HEARING.read_text()
    if "\n---\n" in raw:
        raw = raw.split("\n---\n", 1)[1]
    return raw


def replay_hearing(chunk_words: int = 25) -> dict:
    docs = _load_corpus()
    terms = load_terms(LEXICON)
    prior_docs = [d for d in docs if d.get("date", "") < HEARING_DATE]

    priors_full = predict_event_priors(prior_docs, terms, news_context="",
                                       adjuster=NullAdjuster(), as_of=HEARING_DATE)
    priors = {c: row["p_prior"] for c, row in priors_full.items()}
    weights = cooccurrence_weights(prior_docs, terms, terms, w_max=2.0)

    tracker = LiveMentionTracker(terms=terms, priors=priors, weights=weights, evidence_terms=terms)
    n_chunks = 0
    for delta in ReplayStream(_hearing_text(), chunk_words=chunk_words).stream():
        tracker.consume(delta)
        n_chunks += 1
    return {"final_probabilities": tracker.probabilities(), "n_chunks": n_chunks}


def main() -> None:
    r = replay_hearing()
    probs = r["final_probabilities"]
    resolved = sorted(c for c, p in probs.items() if p == 1.0)
    print(f"chunks streamed: {r['n_chunks']}   terms tracked: {len(probs)}")
    print(f"resolved (spoken): {resolved}")
    print(f"{'term':<22} {'live P':>7}")
    for c in sorted(probs, key=lambda c: -probs[c]):
        print(f"{c:<22} {probs[c]:>7.3f}")


if __name__ == "__main__":
    main()

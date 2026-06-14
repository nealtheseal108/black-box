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
from src.trading.executor import TradeExecutor
from src.trading.client import PaperKalshiClient
from src.live.episode_log import EpisodeLogger

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


def replay_hearing_with_trading(quotes, mapping, paper_log, episode_log,
                                chunk_words: int = 25, threshold: float = 0.08, fee: float = 0.02) -> dict:
    """End-to-end: pre-speech bets on Mode-1 priors, live bets as the hearing streams,
    then settle every position against what was actually said. Returns a trade summary.
    `quotes` is any object with quotes_for(canonical) -> {venue: price}.
    """
    import json
    from pathlib import Path

    docs = _load_corpus()
    terms = load_terms(LEXICON)
    prior_docs = [d for d in docs if d.get("date", "") < HEARING_DATE]

    priors_full = predict_event_priors(prior_docs, terms, news_context="",
                                       adjuster=NullAdjuster(), as_of=HEARING_DATE)
    priors = {c: row["p_prior"] for c, row in priors_full.items()}
    weights = cooccurrence_weights(prior_docs, terms, terms, w_max=2.0)
    tracker = LiveMentionTracker(terms=terms, priors=priors, weights=weights, evidence_terms=terms)

    executor = TradeExecutor(mapping=mapping, quotes=quotes,
                             client=PaperKalshiClient(log_path=paper_log),
                             episode_log=EpisodeLogger(episode_log),
                             bankroll=None, threshold=threshold, fee=fee, phase="pre")

    # pre-speech: bet on Mode-1 priors
    n_trades = len(executor.evaluate(priors))

    # live: stream the hearing, bet on tracker updates
    executor.set_phase("live")
    for delta in ReplayStream(_hearing_text(), chunk_words=chunk_words).stream():
        tracker.consume(delta)
        n_trades += len(executor.evaluate(tracker.probabilities()))

    # settle against realized outcomes (resolved terms => 1, else 0)
    final = tracker.probabilities()
    outcomes = {c: (1 if p == 1.0 else 0) for c, p in final.items()}
    executor.settle(outcomes)

    ep_path = Path(episode_log)
    lines = ep_path.read_text().splitlines() if ep_path.exists() else []
    total_pnl = sum(json.loads(l)["reward"] for l in lines if l.strip())
    return {"n_trades": n_trades, "n_episodes": len(lines), "total_pnl": total_pnl}


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

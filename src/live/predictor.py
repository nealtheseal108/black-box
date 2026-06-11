"""LivePredictor — Bayesian posterior per speech chunk (C6 inference loop).

Inference-only: prior is fixed; cumulative diction log-likelihood is recomputed
from scratch each chunk (idempotent — a stuttered word stream cannot double-count).
"""
from __future__ import annotations

from typing import Iterator

from src.live.inference import bayesian_update, diction_loglikelihood, chunk_words
from src.live.types import MarketState, Transcriber
from src.trading.types import Signal


class LivePredictor:
    def __init__(self, model, markets: list[MarketState], every: int = 5, scale: float = 1.0):
        self.model = model
        self.markets = markets
        self.every = every
        self.scale = scale

    def run(self, transcriber: Transcriber) -> Iterator[Signal]:
        """Stream words; every `every` words recompute each market's posterior from its
        fixed prior + the cumulative diction log-likelihood over the full transcript so far,
        and emit a Signal. Recompute-from-prior (not incremental) keeps it idempotent."""
        transcript: list[str] = []
        for chunk in chunk_words(transcriber.words(), every=self.every):
            transcript.extend(chunk)
            text = " ".join(transcript)
            signals = self.model.phrase_signals(text)
            for m in self.markets:
                delta = diction_loglikelihood(signals, m.signal_axis, self.scale)
                posterior = bayesian_update(m.prior_prob, delta)
                yield Signal.from_quote(
                    ticker=m.ticker, model_prob=posterior, market_price=m.yes_price,
                    timestamp=f"chunk-{len(transcript)}",
                )

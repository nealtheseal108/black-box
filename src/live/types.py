"""Live predictor types (C6) — INTERFACES.md §3–4.

MarketState: per-market state for the live Bayesian update loop.
Transcriber: protocol for injectable STT (Deepgram default, Whisper alternate, ListTranscriber for tests).
"""
from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable


@dataclass
class MarketState:
    ticker: str
    yes_price: float        # current market YES price (from Kalshi feed / C7 quotes)
    prior_prob: float       # Mode-1 prior P(YES) from C4 (output/priors/...)
    signal_axis: str        # diction axis that drives this market: hawkish|dovish|independence|qt


@runtime_checkable
class Transcriber(Protocol):
    def words(self) -> Iterable[str]:
        """Yield transcript words as they are recognized (live STT)."""
        ...


class ListTranscriber:
    """Test/replay transcriber over a fixed word list (no audio, no network)."""
    def __init__(self, words: list[str]): self._words = words
    def words(self):
        yield from self._words

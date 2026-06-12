"""VocabularySource seam: which candidate terms to score for a given event.

LexiconVocabulary returns a fixed curated pool — correct for Warsh, whose past
speeches predate mention markets. A future MarketVocabulary will return the real
historical Kalshi/Polymarket term list for events that had markets (Powell, SOTU).
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.mentions.terms import MarketTerm, load_terms
from src.backtest.events import Event


class VocabularySource(Protocol):
    def terms_for(self, event: Event) -> list[MarketTerm]:
        """Return the candidate terms to score for this event."""
        ...


class LexiconVocabulary:
    """Fixed curated lexicon, independent of the event."""

    def __init__(self, path: str | Path) -> None:
        self._terms = load_terms(path)

    def terms_for(self, event: Event) -> list[MarketTerm]:
        return self._terms

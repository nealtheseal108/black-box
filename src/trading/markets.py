"""Curated term -> venue market mapping.

A MarketLink ties one canonical model term to its Kalshi ticker and Polymarket id.
The executor trades ONLY terms present here — an unmapped term is skipped, never
fuzzy-matched, so a bet can never land on the wrong contract.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MarketLink:
    canonical: str
    kalshi_ticker: str
    polymarket_id: str
    kalshi_question: str = ""
    polymarket_question: str = ""

    def ticker_for(self, venue: str) -> str:
        return self.kalshi_ticker if venue == "kalshi" else self.polymarket_id


def load_market_map(path: str | Path) -> dict[str, MarketLink]:
    raw = json.loads(Path(path).read_text())
    return {e["canonical"]: MarketLink(**e) for e in raw}

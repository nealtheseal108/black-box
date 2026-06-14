"""Cross-venue market quotes for the mention markets.

Each venue adapter turns an injected fetch(market_id) -> dict|None into a normalized
YES price in [0,1]. `requests` is imported lazily so the test suite needs no network
or dependency. CrossVenueQuotes aggregates per-term prices across venues, using the
curated mapping to look up each venue's market id.
"""
from __future__ import annotations

from typing import Callable, Protocol

from src.trading.markets import MarketLink


class VenueQuotes(Protocol):
    venue: str
    def price(self, market_id: str) -> float | None:
        """Return the current YES price (0..1) for this venue's market, or None."""
        ...


def _kalshi_fetch(market_id: str) -> dict | None:
    import requests  # lazy
    r = requests.get(f"https://api.elections.kalshi.com/trade-api/v2/markets/{market_id}", timeout=15)
    if r.status_code != 200:
        return None
    m = r.json().get("market", {})
    cents = m.get("yes_bid")
    return {"yes_price": cents / 100.0} if cents is not None else None


def _polymarket_fetch(market_id: str) -> dict | None:
    import requests  # lazy
    r = requests.get(f"https://gamma-api.polymarket.com/markets/{market_id}", timeout=15)
    if r.status_code != 200:
        return None
    price = r.json().get("outcomePrices", [None])[0]
    return {"yes_price": float(price)} if price is not None else None


class _VenueQuotesBase:
    def __init__(self, fetch: Callable[[str], dict | None]) -> None:
        self._fetch = fetch

    def price(self, market_id: str) -> float | None:
        data = self._fetch(market_id)
        if not data:
            return None
        return data.get("yes_price")


class KalshiQuotes(_VenueQuotesBase):
    venue = "kalshi"
    def __init__(self, fetch: Callable[[str], dict | None] | None = None) -> None:
        super().__init__(fetch or _kalshi_fetch)


class PolymarketQuotes(_VenueQuotesBase):
    venue = "polymarket"
    def __init__(self, fetch: Callable[[str], dict | None] | None = None) -> None:
        super().__init__(fetch or _polymarket_fetch)


class CrossVenueQuotes:
    def __init__(self, venues: list, mapping: dict[str, MarketLink]) -> None:
        self._venues = venues
        self._mapping = mapping

    def quotes_for(self, canonical: str) -> dict[str, float]:
        link = self._mapping.get(canonical)
        if link is None:
            return {}
        out: dict[str, float] = {}
        for v in self._venues:
            p = v.price(link.ticker_for(v.venue))
            if p is not None:
                out[v.venue] = p
        return out

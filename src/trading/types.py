"""Trading data types — INTERFACES.md §3."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class KalshiMarket:
    ticker: str          # Kalshi market id
    title: str
    yes_price: float     # last/mid, dollars 0.00–1.00
    no_price: float
    question: str        # what the contract resolves on


@dataclass
class Signal:
    ticker: str
    model_prob: float    # model P(YES)
    market_price: float  # current YES price
    edge: float          # model_prob - market_price (gross)
    side: str            # "yes" | "no"
    timestamp: str       # ISO 8601

    @classmethod
    def from_quote(cls, ticker: str, model_prob: float, market_price: float,
                   timestamp: str) -> Signal:
        edge = model_prob - market_price
        side = "yes" if edge > 0 else "no"
        return cls(
            ticker=ticker,
            model_prob=model_prob,
            market_price=market_price,
            edge=edge,
            side=side,
            timestamp=timestamp,
        )


@dataclass
class Order:
    ticker: str
    side: str            # "yes" | "no"
    count: int           # number of contracts
    limit_price: float   # limit price per contract (0.00–1.00)
    mode: str            # "paper" | "live"


@dataclass
class Fill:
    ticker: str
    side: str
    count: int
    fill_price: float
    simulated: bool      # True for paper fills
    timestamp: str


@dataclass
class Position:
    ticker: str
    side: str
    count: int
    avg_price: float
    mode: str

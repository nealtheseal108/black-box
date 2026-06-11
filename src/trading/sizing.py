"""Capped Kelly position sizer.

Safety invariants:
- Kelly fraction capped at KELLY_CAP (0.25) — never all-in.
- Kelly fraction is never negative (no shorting via this path).
- size_order returns None when mode=="live" and bankroll is None (E1 not provided).
- Paper mode uses PAPER_NOTIONAL as a simulated bankroll when bankroll is unset.
"""
from __future__ import annotations
from src.trading.types import Order

KELLY_CAP = 0.25            # quarter-Kelly ceiling — size with evidence, never ahead of it
PAPER_NOTIONAL = 1000.0     # simulated bankroll for paper mode when BANKROLL unset


def kelly_fraction(prob: float, price: float) -> float:
    """Compute Kelly fraction for a binary YES contract.

    Binary contract priced 0..1; payoff if YES = (1-price)/price, prob of win = prob.
    """
    if not (0 < price < 1):
        return 0.0
    b = (1 - price) / price
    f = (b * prob - (1 - prob)) / b      # Kelly criterion
    f = max(0.0, f)                       # never negative (no shorting via this path)
    return min(f, KELLY_CAP)              # cap


def size_order(signal, bankroll, mode: str, cap: float = KELLY_CAP) -> Order | None:
    """Size an order for the given signal.

    Returns None (hard stop) when mode=="live" and bankroll is None —
    we refuse to place live orders without an explicit bankroll (E1 not provided).
    """
    if mode == "live" and bankroll is None:
        return None                       # E1 not provided — refuse to size live. Hard stop.
    bank = bankroll if bankroll is not None else PAPER_NOTIONAL
    prob = signal.model_prob if signal.side == "yes" else 1 - signal.model_prob
    price = signal.market_price if signal.side == "yes" else 1 - signal.market_price
    frac = kelly_fraction(prob, price)
    if frac <= 0:
        return None
    stake = bank * frac
    count = max(1, int(stake / max(price, 0.01)))
    return Order(ticker=signal.ticker, side=signal.side, count=count,
                 limit_price=round(price, 2), mode=mode)

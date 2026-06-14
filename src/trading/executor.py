"""TradeExecutor — one-shot, cross-venue, paper.

For each MAPPED, not-yet-traded term: build a C7 Signal against each venue's price,
route to the venue with the highest after-fee net edge (best execution), gate, size
(quarter-Kelly), and place on the paper client. One position per term (no re-trading);
unmapped terms are refused. settle() computes paper P&L at resolution and logs the
RL episode (reward = realized P&L).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from src.trading.types import Signal, Order, Fill
from src.trading.gate import net_edge, passes_gate
from src.trading.sizing import size_order
from src.trading.markets import MarketLink


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class OpenPosition:
    canonical: str
    venue: str
    order: Order
    fill: Fill
    model_prob: float
    phase: str


class TradeExecutor:
    def __init__(self, mapping: dict[str, MarketLink], quotes, client, episode_log,
                 bankroll=None, threshold: float = 0.08, fee: float = 0.02,
                 phase: str = "pre") -> None:
        self._mapping = mapping
        self._quotes = quotes
        self._client = client
        self._episode_log = episode_log
        self._bankroll = bankroll
        self._threshold = threshold
        self._fee = fee
        self._phase = phase
        self._traded: dict[str, OpenPosition] = {}

    def set_phase(self, phase: str) -> None:
        self._phase = phase

    def evaluate(self, probs: dict[str, float]) -> list[OpenPosition]:
        opened: list[OpenPosition] = []
        for canonical, p in probs.items():
            if canonical in self._traded:
                continue
            link = self._mapping.get(canonical)
            if link is None:
                continue  # refuse unmapped — never fuzzy-match a bet
            venue_prices = self._quotes.quotes_for(canonical)
            if not venue_prices:
                continue
            best = None
            for venue, price in venue_prices.items():
                sig = Signal.from_quote(ticker=link.ticker_for(venue), model_prob=p,
                                        market_price=price, timestamp=_now())
                ne = net_edge(sig, self._fee)
                if best is None or ne > best[2]:
                    best = (venue, sig, ne)
            venue, sig, _ = best
            if not passes_gate(sig, self._threshold, self._fee):
                continue
            order = size_order(sig, self._bankroll, mode="paper")
            if order is None:
                continue
            fill = self._client.place(order)
            pos = OpenPosition(canonical=canonical, venue=venue, order=order, fill=fill,
                               model_prob=p, phase=self._phase)
            self._traded[canonical] = pos
            opened.append(pos)
        return opened

    def settle(self, outcomes: dict[str, int]) -> None:
        for canonical, pos in self._traded.items():
            outcome = outcomes.get(canonical, 0)
            pnl = self._pnl(pos, outcome)
            self._episode_log.log(
                state={"canonical": canonical, "model_prob": pos.model_prob, "phase": pos.phase},
                action={"venue": pos.venue, "side": pos.order.side,
                        "count": pos.order.count, "price": pos.fill.fill_price},
                reward=pnl,
            )

    @staticmethod
    def _pnl(pos: OpenPosition, outcome: int) -> float:
        price = pos.fill.fill_price
        count = pos.order.count
        won = (pos.order.side == "yes" and outcome == 1) or (pos.order.side == "no" and outcome == 0)
        return (1.0 - price) * count if won else -price * count

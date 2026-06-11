"""Position manager — deduplication and aggregate exposure cap.

Safety invariants:
- No double-fire: each (ticker, side) key may only be accepted once.
- Aggregate cap: sum of all open notional (count * limit_price) cannot exceed max_aggregate.
"""
from __future__ import annotations
from src.trading.types import Order


class PositionManager:
    """Tracks open positions; enforces dedupe and aggregate exposure limits."""

    def __init__(self, max_aggregate: float) -> None:
        self.max_aggregate = max_aggregate
        self._open_keys: set[tuple[str, str]] = set()   # (ticker, side)
        self._aggregate_notional: float = 0.0

    def accept(self, order: Order) -> bool:
        """Return True and record the order if it passes both guards; False otherwise."""
        key = (order.ticker, order.side)
        # Guard 1: dedupe — no double-fire on the same (ticker, side)
        if key in self._open_keys:
            return False
        # Guard 2: aggregate exposure cap
        notional = order.count * order.limit_price
        if self._aggregate_notional + notional > self.max_aggregate:
            return False
        # Passed both guards — record
        self._open_keys.add(key)
        self._aggregate_notional += notional
        return True

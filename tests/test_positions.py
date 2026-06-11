from src.trading.types import Order
from src.trading.positions import PositionManager


def order(tkr, n): return Order(ticker=tkr, side="yes", count=n, limit_price=0.55, mode="paper")


def test_dedupe_rejects_second_order_same_ticker_side():
    pm = PositionManager(max_aggregate=10_000)
    assert pm.accept(order("A", 1))
    assert not pm.accept(order("A", 1))   # already have A/yes — no double-fire


def test_aggregate_cap_blocks_overexposure():
    pm = PositionManager(max_aggregate=100)
    assert pm.accept(order("A", 100))     # 100 * 0.55 = 55 ok
    assert not pm.accept(order("B", 1000))  # would blow the aggregate cap

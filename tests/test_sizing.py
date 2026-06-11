from src.trading.types import Signal
from src.trading.sizing import kelly_fraction, size_order


def sig(prob, price): return Signal.from_quote("X", prob, price, "t")


def test_kelly_fraction_capped_and_nonnegative():
    f = kelly_fraction(prob=0.65, price=0.55)
    assert 0 < f <= 0.25   # capped at quarter-Kelly per default cap


def test_kelly_fraction_zero_when_no_edge():
    assert kelly_fraction(prob=0.50, price=0.50) == 0.0


def test_size_order_refuses_live_without_bankroll():
    assert size_order(sig(0.65, 0.55), bankroll=None, mode="live") is None


def test_size_order_sizes_in_paper_with_notional_bankroll():
    o = size_order(sig(0.65, 0.55), bankroll=None, mode="paper")
    assert o is not None and o.mode == "paper" and o.count >= 1


def test_size_order_sizes_no_side_correctly():
    # model_prob 0.30 < market_price 0.50 → side "no"; exercises the 1-price branch.
    s = sig(0.30, 0.50)
    assert s.side == "no"
    o = size_order(s, bankroll=1000.0, mode="paper")
    assert o is not None and o.side == "no"
    # NO side: prob_no=0.70, price_no=0.50 → Kelly (0.70-0.50)/0.50 = 0.40, capped to 0.25.
    # stake = 1000*0.25 = 250; count = int(250/0.50) = 500.
    assert o.count == 500
    assert o.limit_price == 0.50

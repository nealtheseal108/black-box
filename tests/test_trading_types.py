from src.trading.types import Signal, KalshiMarket, Order


def test_signal_computes_edge_and_side_from_prob_vs_price():
    s = Signal.from_quote(ticker="FED-HOLD-JUN", model_prob=0.62, market_price=0.55,
                          timestamp="2026-06-16T18:00:00Z")
    assert abs(s.edge - 0.07) < 1e-9
    assert s.side == "yes"   # model thinks YES underpriced


def test_signal_side_is_no_when_model_below_price():
    s = Signal.from_quote("X", model_prob=0.30, market_price=0.50, timestamp="t")
    assert s.side == "no"
    assert s.edge < 0   # gross edge sign carries direction

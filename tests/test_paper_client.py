import pytest
from src.trading.types import Order
from src.trading.client import PaperKalshiClient, make_client


def test_paper_client_simulates_fill_without_network():
    c = PaperKalshiClient()
    fill = c.place(Order(ticker="X", side="yes", count=3, limit_price=0.55, mode="paper"))
    assert fill.ticker == "X" and fill.count == 3 and fill.simulated is True


def test_make_client_defaults_to_paper():
    assert isinstance(make_client(mode="paper"), PaperKalshiClient)


def test_make_client_refuses_live_without_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("SPEECHEDGE_ALLOW_LIVE", raising=False)
    with pytest.raises(RuntimeError, match="live"):
        make_client(mode="live")   # live requires explicit, loud opt-in (E4)

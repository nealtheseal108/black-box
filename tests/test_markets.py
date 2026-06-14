from pathlib import Path
from src.trading.markets import MarketLink, load_market_map

MAP = Path("corpus/markets/june_presser.json")


def test_market_link_exposes_per_venue_ticker():
    link = MarketLink(canonical="rate cut", kalshi_ticker="K-RC", polymarket_id="p-rc",
                      kalshi_question="q1", polymarket_question="q2")
    assert link.ticker_for("kalshi") == "K-RC"
    assert link.ticker_for("polymarket") == "p-rc"


def test_load_market_map_keys_by_canonical():
    m = load_market_map(MAP)
    assert "rate cut" in m and "trump" in m
    assert m["rate cut"].kalshi_ticker == "KXWARSHJUNE-RATECUT"
    assert "soft landing" not in m

from src.trading.markets import MarketLink
from src.trading.quotes import KalshiQuotes, PolymarketQuotes, CrossVenueQuotes


def test_kalshi_quotes_parses_yes_price():
    def fake_fetch(market_id):
        return {"yes_price": 0.61}
    assert KalshiQuotes(fetch=fake_fetch).price("K-RC") == 0.61


def test_quote_returns_none_when_fetch_returns_none():
    def fake_fetch(market_id):
        return None
    assert KalshiQuotes(fetch=fake_fetch).price("missing") is None


def test_cross_venue_returns_price_per_available_venue():
    link = MarketLink(canonical="rate cut", kalshi_ticker="K-RC", polymarket_id="p-rc")
    mapping = {"rate cut": link}
    kalshi = KalshiQuotes(fetch=lambda mid: {"yes_price": 0.60})
    poly = PolymarketQuotes(fetch=lambda mid: {"yes_price": 0.70})
    xq = CrossVenueQuotes([kalshi, poly], mapping)
    assert xq.quotes_for("rate cut") == {"kalshi": 0.60, "polymarket": 0.70}


def test_cross_venue_omits_venue_with_no_quote_and_unmapped_term():
    link = MarketLink(canonical="rate cut", kalshi_ticker="K-RC", polymarket_id="p-rc")
    mapping = {"rate cut": link}
    kalshi = KalshiQuotes(fetch=lambda mid: {"yes_price": 0.60})
    poly = PolymarketQuotes(fetch=lambda mid: None)
    xq = CrossVenueQuotes([kalshi, poly], mapping)
    assert xq.quotes_for("rate cut") == {"kalshi": 0.60}
    assert xq.quotes_for("not in mapping") == {}

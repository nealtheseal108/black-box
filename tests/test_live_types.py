from src.live.types import MarketState, ListTranscriber

def test_market_state_holds_prior_price_and_axis():
    m = MarketState(ticker="FED-HOLD-JUN", yes_price=0.55, prior_prob=0.62, signal_axis="hawkish")
    assert m.ticker == "FED-HOLD-JUN" and m.signal_axis == "hawkish"

def test_list_transcriber_yields_words_in_order():
    t = ListTranscriber(["inflation", "is", "a", "choice"])
    assert list(t.words()) == ["inflation", "is", "a", "choice"]

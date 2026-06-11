from src.live.types import MarketState, ListTranscriber
from src.live.predictor import LivePredictor

class FakeModel:
    """phrase_signals returns hawkish signal once the transcript contains the trigger."""
    def phrase_signals(self, text: str):
        return [{"phrase": "inflation is a choice", "axis": "hawkish", "weight": 1.0}] \
            if "inflation is a choice" in text else []

def test_predictor_raises_posterior_on_hawkish_speech_for_hawkish_market():
    markets = [MarketState("FED-HOLD-JUN", yes_price=0.55, prior_prob=0.55, signal_axis="hawkish")]
    pred = LivePredictor(FakeModel(), markets, every=3)
    sigs = list(pred.run(ListTranscriber("inflation is a choice without excuse".split())))
    # at least one signal emitted; final posterior for the market should exceed the 0.55 prior
    assert sigs
    last = [s for s in sigs if s.ticker == "FED-HOLD-JUN"][-1]
    assert last.model_prob > 0.55
    assert last.market_price == 0.55

def test_predictor_neutral_speech_leaves_posterior_near_prior():
    markets = [MarketState("FED-HOLD-JUN", 0.55, 0.55, "hawkish")]
    pred = LivePredictor(FakeModel(), markets, every=2)
    sigs = list(pred.run(ListTranscriber("the weather is nice today".split())))
    if sigs:
        assert abs(sigs[-1].model_prob - 0.55) < 1e-9   # no diction evidence → prior unchanged

from pathlib import Path
import json
from live_predictor import load_market_states

def test_load_market_states_merges_priors_and_prices(tmp_path: Path):
    priors = [{"ticker": "FED-HOLD-JUN", "prior_prob": 0.62, "rationale": "r",
               "as_of": "t", "sources": ["s"]}]
    pf = tmp_path / "priors.json"; pf.write_text(json.dumps(priors))
    # price + axis map supplied alongside (from Kalshi feed + phrase_market_map)
    prices = {"FED-HOLD-JUN": 0.55}
    axes = {"FED-HOLD-JUN": "hawkish"}
    states = load_market_states(pf, prices, axes)
    assert len(states) == 1
    assert states[0].prior_prob == 0.62 and states[0].yes_price == 0.55
    assert states[0].signal_axis == "hawkish"

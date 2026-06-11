from src.dashboard.state import build_state


def test_build_state_assembles_all_panels():
    transcript = "inflation is a choice"
    tone = {"hawkish": 0.9, "dovish": 0.0, "independence": 0.0, "qt": 0.0}
    markets = [{"ticker": "FED-HOLD-JUN", "prior_prob": 0.55, "model_prob": 0.66,
                "yes_price": 0.55, "edge": 0.11, "side": "yes"}]
    fills = [{"ticker": "FED-HOLD-JUN", "side": "yes", "count": 10, "fill_price": 0.55,
              "simulated": True, "timestamp": "t"}]
    st = build_state(transcript, tone, markets, fills, resolutions={})
    d = st.model_dump()
    assert d["transcript"] == transcript
    assert d["tone"]["hawkish"] == 0.9
    assert d["markets"][0]["ticker"] == "FED-HOLD-JUN" and d["markets"][0]["edge"] == 0.11
    assert d["bets"][0]["count"] == 10 and d["bets"][0]["simulated"] is True
    assert d["pnl"]["open_positions"] == 1

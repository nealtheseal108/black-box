from src.agents.context_types import MacroSnapshot, MarketContext, Prior
from src.agents.synthesize import build_prompt, synthesize_priors

SNAP = MacroSnapshot(as_of="2026-06-14T00:00:00Z",
    data_prints=["April CPI 3.4%"], futures=["55% hold priced"],
    news=["energy shock"], speaker_recent=["'inflation is a choice'"])
MKTS = [MarketContext(ticker="FED-HOLD-JUN", title="June hold", question="Will the Fed hold?", yes_price=0.55)]


def test_build_prompt_includes_context_and_market():
    p = build_prompt(SNAP, MKTS[0])
    assert "inflation is a choice" in p
    assert "FED-HOLD-JUN" in p and "0.55" in p


def test_synthesize_priors_uses_injected_call_and_returns_priors():
    # Fake Claude call returns a validated Prior per market — no network.
    def fake_call(prompt: str, market: MarketContext) -> Prior:
        return Prior(ticker=market.ticker, prior_prob=0.62, rationale="hawkish lean",
                     as_of=SNAP.as_of, sources=["fmp", "speaker"])
    priors = synthesize_priors(SNAP, MKTS, call_fn=fake_call)
    assert len(priors) == 1
    assert priors[0].ticker == "FED-HOLD-JUN"
    assert 0 <= priors[0].prior_prob <= 1

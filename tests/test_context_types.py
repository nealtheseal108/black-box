from src.agents.context_types import Prior
import json


def test_prior_matches_interface_shape():
    p = Prior(ticker="FED-RATE-CUT-JUN", prior_prob=0.62,
              rationale="April CPI 3yr high + energy shock argues for hold",
              as_of="2026-06-14T00:00:00Z", sources=["fmp:cpi", "speaker:hearing-2026-04-21"])
    d = json.loads(p.model_dump_json())
    assert set(d) == {"ticker", "prior_prob", "rationale", "as_of", "sources"}
    assert 0.0 <= d["prior_prob"] <= 1.0


def test_prior_prob_must_be_a_probability():
    import pytest, pydantic
    with pytest.raises(pydantic.ValidationError):
        Prior(ticker="X", prior_prob=1.4, rationale="r", as_of="t", sources=[])

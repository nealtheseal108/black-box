import json
from pathlib import Path
from src.agents.context_types import Prior
from context_agent import write_priors


def test_write_priors_emits_interface_json(tmp_path: Path):
    priors = [Prior(ticker="X", prior_prob=0.5, rationale="r", as_of="t", sources=["s"])]
    out = tmp_path / "2026-06-16.json"
    write_priors(priors, out)
    data = json.loads(out.read_text())
    assert data[0]["ticker"] == "X" and set(data[0]) == {"ticker", "prior_prob", "rationale", "as_of", "sources"}

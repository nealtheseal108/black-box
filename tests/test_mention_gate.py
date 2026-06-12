from src.backtest.gates import evaluate_mention_gate


def test_gate_passes_when_brier_below_threshold():
    rep = evaluate_mention_gate(brier=0.18, threshold=0.25)
    assert rep["pass"] is True
    assert rep["metric"] == 0.18
    assert rep["threshold"] == "< 0.25"


def test_gate_fails_when_brier_at_or_above_threshold():
    assert evaluate_mention_gate(brier=0.25, threshold=0.25)["pass"] is False
    assert evaluate_mention_gate(brier=0.40, threshold=0.25)["pass"] is False

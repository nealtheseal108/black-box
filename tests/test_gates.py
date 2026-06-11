from src.backtest.gates import evaluate_gates

def test_all_gates_pass():
    r = evaluate_gates({"g1_accuracy": 0.25, "g2_brier": 0.18,
                        "g3_positive_rate": 0.60, "g3_avg_net_edge": 0.07})
    assert r["g1"]["pass"] and r["g2"]["pass"] and r["g3"]["pass"]
    assert r["all_pass"] is True

def test_g3_fails_when_edge_below_floor():
    r = evaluate_gates({"g1_accuracy": 0.25, "g2_brier": 0.18,
                        "g3_positive_rate": 0.60, "g3_avg_net_edge": 0.05})
    assert r["g3"]["pass"] is False        # avg edge 0.05 < 0.06 required
    assert r["all_pass"] is False

def test_g1_fails_below_threshold():
    r = evaluate_gates({"g1_accuracy": 0.15, "g2_brier": 0.18,
                        "g3_positive_rate": 0.60, "g3_avg_net_edge": 0.07})
    assert r["g1"]["pass"] is False
    assert r["all_pass"] is False

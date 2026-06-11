from backtest import run_level1, run_level2

def test_run_level1_trains_and_scores_on_held_out():
    # tiny corpus with a repeated pattern the model can learn
    docs = [{"text": "inflation is a choice"} for _ in range(8)] + \
           [{"text": "the balance sheet is bloated"} for _ in range(2)]
    g1 = run_level1(docs, test_frac=0.2, seed=0)
    assert 0.0 <= g1 <= 1.0          # returns an accuracy

def test_run_level2_computes_g2_g3_from_injected_loader():
    # injected Powell data: forecasts + realized outcomes + signal edges
    def loader():
        return {"forecasts": [0.8, 0.3], "outcomes": [1, 0],
                "signal_outcomes": [{"edge": 0.10}, {"edge": 0.09}]}
    g2, g3 = run_level2(loader, fee=0.02)
    assert "brier" in g2 and "positive_rate" in g3

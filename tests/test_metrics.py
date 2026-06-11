from src.backtest.metrics import top1_accuracy, brier_score, market_edge_stats

class FakeModel:
    """predict_next returns a fixed top-1 so accuracy is deterministic."""
    def __init__(self, always): self._always = always
    def predict_next(self, context, k=1): return [(self._always, 1.0)]

def test_top1_accuracy_counts_correct_next_token_predictions():
    # text "a b a b" → positions predict the token after each context; model always says "b"
    model = FakeModel("b")
    docs = [{"text": "a b a b"}]
    acc = top1_accuracy(model, docs, min_context=1)
    # contexts -> actual next: [a]->b ✓, [a,b]->a ✗, [a,b,a]->b ✓  => 2/3
    assert abs(acc - 2/3) < 1e-9

def test_brier_score_is_mean_squared_error_of_forecasts():
    # forecasts vs binary outcomes
    assert abs(brier_score([0.9, 0.2], [1, 0]) - ((0.1**2 + 0.2**2)/2)) < 1e-9
    assert brier_score([], []) == 0.0

def test_market_edge_stats_positive_rate_and_avg_net_edge():
    # each entry: (gross_edge, realized_profit_sign_ignored) — we score edge after fee
    signal_outcomes = [{"edge": 0.10}, {"edge": 0.09}, {"edge": 0.04}]
    stats = market_edge_stats(signal_outcomes, fee=0.02)
    # net edges: 0.08, 0.07, 0.02 ; positive(>0): all 3 ; >0.06 floor: 2/3
    assert stats["n"] == 3
    assert abs(stats["avg_net_edge"] - (0.08+0.07+0.02)/3) < 1e-9
    assert abs(stats["positive_rate"] - 1.0) < 1e-9          # all net-positive
    assert abs(stats["above_floor_rate"] - 2/3) < 1e-9       # clear the $0.06 G3 floor

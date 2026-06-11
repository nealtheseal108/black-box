from src.dashboard.state import compute_pnl


def test_pnl_yes_win_and_loss():
    fills = [{"ticker": "A", "side": "yes", "count": 10, "fill_price": 0.55}]
    assert abs(compute_pnl(fills, {"A": 1})["realized"] - 10 * (1 - 0.55)) < 1e-9   # YES wins
    assert abs(compute_pnl(fills, {"A": 0})["realized"] - (-10 * 0.55)) < 1e-9       # YES loses


def test_pnl_no_side_wins_when_outcome_zero():
    fills = [{"ticker": "B", "side": "no", "count": 4, "fill_price": 0.40}]
    assert abs(compute_pnl(fills, {"B": 0})["realized"] - 4 * (1 - 0.40)) < 1e-9      # NO wins
    assert abs(compute_pnl(fills, {"B": 1})["realized"] - (-4 * 0.40)) < 1e-9


def test_pnl_unresolved_counts_as_open_not_realized():
    fills = [{"ticker": "C", "side": "yes", "count": 5, "fill_price": 0.5}]
    r = compute_pnl(fills, {})            # no resolution yet
    assert r["realized"] == 0.0
    assert r["open_positions"] == 1

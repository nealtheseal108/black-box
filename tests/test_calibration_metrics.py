import math
from src.backtest.calibration import log_loss, auc, reliability_diagram, calibration_summary


def test_log_loss_perfect_is_zero():
    assert math.isclose(log_loss([1.0, 0.0], [1, 0]), 0.0, abs_tol=1e-9)


def test_log_loss_clamps_and_penalizes_confident_wrong():
    v = log_loss([0.0], [1])
    assert v > 30 and math.isfinite(v)


def test_auc_hand_checked_case():
    assert math.isclose(auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]), 0.75, rel_tol=1e-9)


def test_auc_undefined_with_one_class_returns_none():
    assert auc([0.2, 0.7], [1, 1]) is None
    assert auc([0.2, 0.7], [0, 0]) is None


def test_reliability_diagram_bins_observed_frequency():
    bins = reliability_diagram([0.05, 0.05, 0.95], [1, 0, 1], n_bins=10)
    first = bins[0]
    assert first["count"] == 2
    assert math.isclose(first["observed"], 0.5, rel_tol=1e-9)
    last = bins[9]
    assert last["count"] == 1 and math.isclose(last["observed"], 1.0, rel_tol=1e-9)


def test_calibration_summary_bundles_all_metrics():
    pool = [(0.9, 1), (0.1, 0), (0.8, 1), (0.2, 0)]
    s = calibration_summary(pool)
    assert s["n"] == 4
    assert math.isclose(s["base_rate"], 0.5, rel_tol=1e-9)
    assert set(s) == {"n", "base_rate", "brier", "log_loss", "auc", "reliability"}
    assert 0.0 <= s["brier"] <= 1.0

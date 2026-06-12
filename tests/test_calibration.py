import math
from src.backtest.calibration import brier_score, calibration_report, split_before


def test_brier_perfect_prediction_is_zero():
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0


def test_brier_worst_prediction_is_one():
    assert brier_score([0.0, 1.0], [1, 0]) == 1.0


def test_brier_half_is_quarter():
    assert math.isclose(brier_score([0.5, 0.5], [1, 0]), 0.25, rel_tol=1e-9)


def test_calibration_report_aligns_terms_by_name():
    rep = calibration_report({"a": 0.9, "b": 0.1}, {"a": 1, "b": 0})
    assert rep["n"] == 2
    assert math.isclose(rep["brier"], (0.01 + 0.01) / 2, rel_tol=1e-9)


def test_split_before_is_strict_and_leakage_free():
    docs = [{"date": "2020-01-01", "text": "x"}, {"date": "2026-04-21", "text": "y"}]
    train, test = split_before(docs, "2026-04-21")
    assert [d["date"] for d in train] == ["2020-01-01"]
    assert [d["date"] for d in test] == ["2026-04-21"]

import math
from src.mentions.news_adjuster import clamp_multiplier, apply_multipliers, NullAdjuster, MULT_MIN, MULT_MAX


def test_clamp_bounds_the_multiplier():
    assert clamp_multiplier(10.0) == MULT_MAX      # 4.0
    assert clamp_multiplier(0.001) == MULT_MIN     # 0.25
    assert clamp_multiplier(2.0) == 2.0            # in-range unchanged


def test_apply_scales_then_clips_to_unit_interval():
    base = {"a": 0.2, "b": 0.5}
    out = apply_multipliers(base, {"a": 3.0, "b": 4.0})   # b: 0.5*4=2.0 -> clip to 1.0
    assert math.isclose(out["a"], 0.6, rel_tol=1e-9)
    assert out["b"] == 1.0


def test_missing_term_defaults_to_neutral_multiplier():
    out = apply_multipliers({"a": 0.3}, {})        # no entry for "a" -> x1.0
    assert math.isclose(out["a"], 0.3, rel_tol=1e-9)


def test_null_adjuster_returns_all_neutral():
    adj = NullAdjuster()
    assert adj.multipliers(["a", "b"], "any news") == {"a": 1.0, "b": 1.0}

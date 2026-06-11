import math
from src.live.inference import to_logodds, from_logodds, bayesian_update, diction_loglikelihood, chunk_words

def test_logodds_roundtrip():
    for p in (0.1, 0.5, 0.9):
        assert abs(from_logodds(to_logodds(p)) - p) < 1e-9

def test_logodds_clamps_extremes():
    assert 0.0 < from_logodds(to_logodds(0.0)) < 1e-3   # clamped, not -inf
    assert 1.0 - from_logodds(to_logodds(1.0)) < 1e-3

def test_bayesian_update_positive_delta_raises_prob():
    post = bayesian_update(0.50, +1.0)
    assert post > 0.50
    assert bayesian_update(0.50, -1.0) < 0.50
    assert abs(bayesian_update(0.50, 0.0) - 0.50) < 1e-9   # no evidence → unchanged

def test_diction_loglikelihood_matches_axis_adds_opposes_subtracts():
    signals = [
        {"phrase": "inflation is a choice", "axis": "hawkish", "weight": 0.9},
        {"phrase": "stronger not hotter", "axis": "dovish", "weight": 0.7},
        {"phrase": "fiscal dominance", "axis": "independence", "weight": 0.9},
    ]
    # market driven by hawkish axis: hawkish adds, dovish (opposite) subtracts, independence neutral
    d = diction_loglikelihood(signals, market_axis="hawkish", scale=1.0)
    assert abs(d - (0.9 - 0.7)) < 1e-9

def test_chunk_words_emits_every_n():
    chunks = list(chunk_words(["a","b","c","d","e","f","g"], every=3))
    assert chunks == [["a","b","c"], ["d","e","f"], ["g"]]

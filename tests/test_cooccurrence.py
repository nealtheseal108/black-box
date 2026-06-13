import math
from src.mentions.terms import MarketTerm
from src.live.cooccurrence import cooccurrence_weights

A = MarketTerm(canonical="dovish", patterns=(r"\bdovish\b",))
B = MarketTerm(canonical="rate cut", patterns=(r"rate cut",))
C = MarketTerm(canonical="rate hike", patterns=(r"rate hike",))


def _docs():
    return [
        {"text": "dovish stance favors a rate cut"},
        {"text": "dovish tone, another rate cut likely"},
        {"text": "a rate hike under a hawkish board"},
        {"text": "growth and productivity"},
    ]


def test_positive_weight_for_cooccurring_terms():
    w = cooccurrence_weights(_docs(), [A], [B, C], w_max=2.0)
    assert w[("dovish", "rate cut")] > 0


def test_negative_weight_for_anticorrelated_terms():
    w = cooccurrence_weights(_docs(), [A], [B, C], w_max=2.0)
    assert w[("dovish", "rate hike")] == -2.0


def test_weights_are_clamped_to_w_max():
    w = cooccurrence_weights(_docs(), [A], [B, C], w_max=2.0)
    assert all(-2.0 <= v <= 2.0 for v in w.values())


def test_self_pairs_excluded():
    w = cooccurrence_weights(_docs(), [A], [A], w_max=2.0)
    assert ("dovish", "dovish") not in w

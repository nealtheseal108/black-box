import math
from src.mentions.terms import MarketTerm
from src.live.tracker import LiveMentionTracker

DOVISH = MarketTerm(canonical="dovish", patterns=(r"\bdovish\b",))
CUT = MarketTerm(canonical="rate cut", patterns=(r"rate cut",))
HIKE = MarketTerm(canonical="rate hike", patterns=(r"rate hike",))

PRIORS = {"dovish": 0.5, "rate cut": 0.4, "rate hike": 0.4}
WEIGHTS = {("dovish", "rate cut"): 1.5, ("dovish", "rate hike"): -1.5}


def _tracker():
    return LiveMentionTracker(
        terms=[DOVISH, CUT, HIKE],
        priors=PRIORS,
        weights=WEIGHTS,
        evidence_terms=[DOVISH, CUT, HIKE],
    )


def test_term_resolves_to_one_when_spoken():
    t = _tracker()
    t.consume("the committee favors a rate cut")
    assert t.probabilities()["rate cut"] == 1.0


def test_evidence_raises_correlated_and_lowers_anticorrelated():
    t = _tracker()
    base = t.probabilities()
    t.consume("his tone was distinctly dovish today")
    after = t.probabilities()
    assert after["rate cut"] > base["rate cut"]
    assert after["rate hike"] < base["rate hike"]


def test_fire_once_repeated_evidence_does_not_double_count():
    t = _tracker()
    t.consume("dovish")
    once = t.probabilities()["rate cut"]
    t.consume("dovish dovish again")
    twice = t.probabilities()["rate cut"]
    assert math.isclose(once, twice, rel_tol=1e-9)


def test_resolved_term_is_retired_and_not_updated():
    t = _tracker()
    t.consume("rate cut")
    t.consume("very dovish")
    assert t.probabilities()["rate cut"] == 1.0

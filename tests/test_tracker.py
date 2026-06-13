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


def test_cumulative_propagation_is_capped_no_saturation():
    # Many distinct evidence terms each pushing one unsaid target up must NOT
    # saturate it toward 1.0 (the hearing-replay bug: 20 evidence terms summed to ~1.0).
    target = MarketTerm(canonical="target", patterns=(r"\btarget\b",))
    evid = [MarketTerm(canonical=f"e{i}", patterns=(rf"\be{i}\b",)) for i in range(10)]
    terms = [*evid, target]
    priors = {t.canonical: 0.1 for t in terms}                 # low prior for the target
    weights = {(f"e{i}", "target"): 2.0 for i in range(10)}    # 10 x +2.0 = +20 log-odds if uncapped
    t = LiveMentionTracker(terms=terms, priors=priors, weights=weights,
                           evidence_terms=evid, prop_cap=2.0)
    t.consume("e0 e1 e2 e3 e4 e5 e6 e7 e8 e9")
    p = t.probabilities()["target"]
    # uncapped this would be ~1.0; capped at +2.0 log-odds from a 0.1 prior it stays < 0.5
    assert p < 0.5

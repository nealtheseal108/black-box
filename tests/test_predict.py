import math
from src.mentions.terms import MarketTerm
from src.mentions.news_adjuster import NullAdjuster
from src.mentions.predict import predict_event_priors

DOCS = [
    {"context_type": "speech",    "text": "inflation is a choice"},
    {"context_type": "interview", "text": "inflation and growth"},
]
TERMS = [MarketTerm(canonical="inflation", patterns=(r"\binflation\b",))]


class _BoostAdjuster:
    def multipliers(self, terms, news_context):
        return {t: 4.0 for t in terms}    # raw 4x (already at clamp ceiling)


def test_null_adjuster_leaves_prior_equal_to_base_event_rate():
    out = predict_event_priors(DOCS, TERMS, news_context="", adjuster=NullAdjuster())
    row = out["inflation"]
    assert math.isclose(row["p_prior"], row["p_event_base"], rel_tol=1e-9)
    assert set(row) == {"p_prep", "p_qa", "p_event_base", "multiplier", "p_prior"}


def test_boost_adjuster_raises_prior_but_clips_at_one():
    out = predict_event_priors(DOCS, TERMS, news_context="", adjuster=_BoostAdjuster())
    row = out["inflation"]
    assert row["multiplier"] == 4.0
    assert row["p_prior"] >= row["p_event_base"]
    assert row["p_prior"] <= 1.0

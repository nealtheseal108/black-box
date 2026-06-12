import math
from src.mentions.terms import MarketTerm
from src.mentions.base_rate import phase_base_rate, combine_phases, event_base_rate, PREP_TYPES, QA_TYPES

INFL = MarketTerm(canonical="inflation", patterns=(r"\binflation\b",))


def _docs():
    return [
        {"context_type": "speech",    "text": "inflation is a choice and we will act"},
        {"context_type": "speech",    "text": "the labor market remains strong"},          # no inflation
        {"context_type": "lecture",   "text": "inflation expectations matter"},
        {"context_type": "interview", "text": "inflation has been too high lately"},
        {"context_type": "hearing",   "text": "we must finish the job on prices"},          # no inflation
    ]


def test_unseen_term_gets_nonzero_floor_not_zero():
    never = MarketTerm(canonical="stagflation", patterns=(r"\bstagflation\b",))
    p = phase_base_rate(_docs(), never, PREP_TYPES, k=0.5)
    assert p > 0.0
    assert p < 0.2


def test_prep_rate_counts_only_prepared_docs():
    # 3 prep docs (2 speech + 1 lecture); inflation in 2 → (2+0.5)/(3+1) = 0.625
    p = phase_base_rate(_docs(), INFL, PREP_TYPES, k=0.5)
    assert math.isclose(p, 2.5 / 4.0, rel_tol=1e-9)


def test_qa_rate_counts_only_qa_docs():
    # 2 qa docs (interview + hearing); inflation in 1 → (1+0.5)/(2+1) = 0.5
    p = phase_base_rate(_docs(), INFL, QA_TYPES, k=0.5)
    assert math.isclose(p, 1.5 / 3.0, rel_tol=1e-9)


def test_combine_phases_is_noisy_or():
    assert math.isclose(combine_phases(0.5, 0.5), 0.75, rel_tol=1e-9)


def test_event_base_rate_bundles_all_three():
    out = event_base_rate(_docs(), INFL, k=0.5)
    assert set(out) == {"p_prep", "p_qa", "p_event"}
    assert math.isclose(out["p_event"], combine_phases(out["p_prep"], out["p_qa"]), rel_tol=1e-9)


def test_empty_phase_contributes_zero_not_half():
    # The empty-phase bug: a phase with NO docs returned the n=0 Jeffreys prior (0.5),
    # which the noisy-OR then floored every term at >=0.5. No docs => no evidence => 0.0.
    prep_only = [
        {"context_type": "speech", "date": "2008-01-01", "text": "inflation"},
        {"context_type": "speech", "date": "2009-01-01", "text": "growth"},
    ]
    rare = MarketTerm(canonical="stagflation", patterns=(r"\bstagflation\b",))
    assert phase_base_rate(prep_only, rare, QA_TYPES, k=0.5) == 0.0   # QA phase unobserved


def test_event_rate_not_floored_when_one_phase_unobserved():
    # With only prep docs, P_event must equal the prep rate (no 0.5 noisy-OR floor).
    prep_only = [
        {"context_type": "speech", "date": "2008-01-01", "text": "inflation"},
        {"context_type": "speech", "date": "2009-01-01", "text": "growth"},
    ]
    rare = MarketTerm(canonical="stagflation", patterns=(r"\bstagflation\b",))
    out = event_base_rate(prep_only, rare, k=0.5)
    assert out["p_qa"] == 0.0
    assert math.isclose(out["p_event"], out["p_prep"], rel_tol=1e-9)
    assert out["p_event"] < 0.2   # rare term no longer inflated to ~0.5


def test_recency_weight_halves_each_half_life():
    from src.mentions.base_rate import recency_weight
    # a doc one half-life (4y) before the event gets weight 0.5; same-day gets 1.0
    assert math.isclose(recency_weight("2022-01-01", "2026-01-01", 4.0), 0.5, rel_tol=2e-2)
    assert math.isclose(recency_weight("2026-01-01", "2026-01-01", 4.0), 1.0, rel_tol=1e-9)
    # docs dated after the event are clamped to weight 1.0 (age floored at 0)
    assert math.isclose(recency_weight("2027-01-01", "2026-01-01", 4.0), 1.0, rel_tol=1e-9)


def test_recency_weighting_downweights_stale_vocabulary():
    docs = [
        {"context_type": "speech", "date": "2009-01-01", "text": "recession recession recession"},
        {"context_type": "speech", "date": "2025-01-01", "text": "inflation is a choice"},
    ]
    rec = MarketTerm(canonical="recession", patterns=(r"\brecession\b",))
    # unweighted: 1 of 2 prep docs mentions it -> (1+0.5)/(2+1) = 0.5
    p_unweighted = phase_base_rate(docs, rec, PREP_TYPES, k=0.5)
    # as of 2026 with a 4y half-life, the 2009 doc is heavily discounted -> lower rate
    p_recency = phase_base_rate(docs, rec, PREP_TYPES, k=0.5, as_of="2026-01-01", half_life_years=4.0)
    assert p_recency < p_unweighted


def test_as_of_none_reproduces_unweighted_rate():
    # backward compatibility: no as_of => identical to the original unweighted estimate
    p_default = phase_base_rate(_docs(), INFL, PREP_TYPES, k=0.5)
    p_explicit_none = phase_base_rate(_docs(), INFL, PREP_TYPES, k=0.5, as_of=None)
    assert p_default == p_explicit_none

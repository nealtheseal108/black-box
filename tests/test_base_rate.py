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

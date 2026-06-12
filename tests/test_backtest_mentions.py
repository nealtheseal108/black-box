from backtest_mentions import resolve_outcomes, run_calibration
from src.mentions.terms import MarketTerm


def test_resolve_outcomes_from_event_text():
    terms = [
        MarketTerm(canonical="inflation", patterns=(r"\binflation\b",)),
        MarketTerm(canonical="stagflation", patterns=(r"\bstagflation\b",)),
    ]
    outcomes = resolve_outcomes(terms, "inflation remains the focus")
    assert outcomes == {"inflation": 1, "stagflation": 0}


def test_run_calibration_on_real_corpus_produces_a_brier_and_gate():
    result = run_calibration()
    assert 0.0 <= result["report"]["brier"] <= 1.0
    assert result["report"]["n"] == 8
    assert "pass" in result["gate"]
    assert result["train_docs"] >= 20
    assert result["test_event_words"] > 1000

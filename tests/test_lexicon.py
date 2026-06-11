from src.warsh.lexicon import score_diction, phrase_signals, SIGNAL_LEXICON

def test_lexicon_covers_brief_seed_phrases():
    for phrase in ["inflation is a choice", "fiscal dominance", "foursquare within its role",
                   "bloated balance sheet", "stronger not hotter"]:
        assert phrase in SIGNAL_LEXICON

def test_score_diction_accumulates_axis_weights():
    text = "Inflation is a choice. We will be foursquare within its role."
    scores = score_diction(text)
    assert scores["hawkish"] > 0
    assert scores["independence"] > 0

def test_phrase_signals_returns_matched_phrases_with_axis():
    sig = phrase_signals("the balance sheet is bloated and fiscal dominance looms")
    keys = {s["phrase"] for s in sig}
    assert "bloated balance sheet" in keys or "fiscal dominance" in keys

from src.backtest.events import Event
from src.backtest.vocabulary import LexiconVocabulary
from pathlib import Path

LEXICON = Path("corpus/lexicon/fed_macro_terms.json")
EVENT = Event(speaker="warsh", date="2026-04-21", text="x", context_type="hearing")


def test_lexicon_loads_market_terms():
    vocab = LexiconVocabulary(LEXICON)
    terms = vocab.terms_for(EVENT)
    assert len(terms) >= 30
    canonicals = {t.canonical for t in terms}
    assert {"inflation", "rate cut", "independence", "trump"} <= canonicals


def test_lexicon_is_event_independent():
    vocab = LexiconVocabulary(LEXICON)
    other = Event(speaker="warsh", date="2008-01-01", text="y", context_type="speech")
    assert [t.canonical for t in vocab.terms_for(EVENT)] == [t.canonical for t in vocab.terms_for(other)]

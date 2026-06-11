from src.warsh.ngram import NgramModel

CORPUS = [
    "inflation is a choice not an accident",
    "inflation is a choice without excuse",
    "the balance sheet is bloated",
]

def test_predict_next_ranks_by_observed_continuation():
    m = NgramModel(max_n=4).train([t.split() for t in CORPUS])
    preds = dict(m.predict_next(["inflation", "is", "a"], k=3))
    assert "choice" in preds
    assert preds["choice"] > 0.0

def test_backoff_uses_lower_order_when_higher_unseen():
    m = NgramModel(max_n=4).train([t.split() for t in CORPUS])
    # "is" follows nothing seen in this exact 3-gram context, backs off to bigram/unigram
    preds = m.predict_next(["never", "seen", "is"], k=2)
    assert preds  # non-empty via backoff, not a crash

def test_probabilities_are_normalized_per_call():
    m = NgramModel(max_n=4).train([t.split() for t in CORPUS])
    preds = m.predict_next(["inflation", "is", "a"], k=10)
    assert abs(sum(p for _, p in preds) - 1.0) < 1e-6

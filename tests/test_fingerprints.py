from src.warsh.fingerprints import phrase_lift, top_fingerprints

def test_phrase_lift_high_when_corpus_overrepresents():
    corpus_counts = {"inflation is a choice": 5}
    corpus_total = 100
    baseline = {"inflation is a choice": 1}   # rare in baseline
    baseline_total = 1_000_000
    lift = phrase_lift("inflation is a choice", corpus_counts, corpus_total, baseline, baseline_total)
    assert lift > 100  # vastly over-represented vs baseline

def test_top_fingerprints_returns_sorted_by_lift():
    docs_tokens = [["inflation","is","a","choice"], ["inflation","is","a","choice"], ["the","cat","sat"]]
    baseline = {"the cat": 50, "cat sat": 50, "inflation is": 1}
    out = top_fingerprints(docs_tokens, baseline, baseline_total=1000, n=2, k=3)
    phrases = [p for p, _ in out]
    assert "inflation is" in phrases
    assert out == sorted(out, key=lambda x: -x[1])

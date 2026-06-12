from backtest_harness import run_harness


def test_run_harness_on_real_corpus_produces_full_summary():
    r = run_harness()
    s = r["summary"]
    assert r["n_scored"] >= 10            # LOO turns n=1 into many events
    assert s["n"] > 100                   # many (pred, outcome) pairs across events x lexicon
    assert 0.0 < s["base_rate"] < 1.0     # both positives and negatives present
    assert 0.0 <= s["brier"] <= 1.0
    assert s["log_loss"] >= 0.0
    assert (s["auc"] is None) or (0.0 <= s["auc"] <= 1.0)
    assert len(s["reliability"]) == 10

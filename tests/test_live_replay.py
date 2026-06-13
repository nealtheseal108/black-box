from live_run import replay_hearing


def test_hearing_replay_resolves_spoken_terms_only():
    result = replay_hearing()
    probs = result["final_probabilities"]
    for said in ["rate cut", "trump", "inflation", "independence", "quantitative easing"]:
        assert probs[said] == 1.0, f"{said} should have resolved"
    for unsaid in ["recession", "stagflation", "soft landing"]:
        assert probs[unsaid] < 0.99, f"{unsaid} should not be saturated by propagation"
    # Non-saturation guard: every term is either exactly resolved (1.0) or held
    # safely below certainty — no unsaid term may be pushed to ~1.0 by propagation alone.
    for term, p in probs.items():
        assert p == 1.0 or p < 0.99, f"{term} saturated to {p} without being spoken"
    assert result["n_chunks"] > 0
    assert len(probs) >= 30

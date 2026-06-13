from live_run import replay_hearing


def test_hearing_replay_resolves_spoken_terms_only():
    result = replay_hearing()
    probs = result["final_probabilities"]
    for said in ["rate cut", "trump", "inflation", "independence", "quantitative easing"]:
        assert probs[said] == 1.0, f"{said} should have resolved"
    for unsaid in ["recession", "stagflation", "soft landing"]:
        assert probs[unsaid] < 1.0, f"{unsaid} should not have resolved"
    assert result["n_chunks"] > 0
    assert len(probs) >= 30

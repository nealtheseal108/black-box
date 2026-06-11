from src.agents.assemble import assemble_context


def test_assemble_gathers_all_sources_into_snapshot():
    fetchers = {
        "data_prints": lambda: ["April CPI 3.4% (3yr high)"],
        "futures": lambda: ["Fed funds futures: 55% hold priced"],
        "news": lambda: ["Energy shock on supply disruption"],
        "speaker_recent": lambda: ["Apr-21 hearing: 'inflation is a choice'"],
    }
    snap = assemble_context(as_of="2026-06-14T00:00:00Z", fetchers=fetchers)
    assert snap.data_prints == ["April CPI 3.4% (3yr high)"]
    assert snap.speaker_recent and "inflation is a choice" in snap.speaker_recent[0]


def test_assemble_tolerates_a_failing_fetcher():
    def boom(): raise RuntimeError("source down")
    fetchers = {"data_prints": boom, "futures": lambda: [], "news": lambda: [], "speaker_recent": lambda: []}
    snap = assemble_context(as_of="t", fetchers=fetchers)
    assert snap.data_prints == []   # failure degrades gracefully, doesn't crash the run

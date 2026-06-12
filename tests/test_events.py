import json
from src.backtest.events import Event, JsonlCorpusLoader


def test_event_is_a_simple_value_object():
    e = Event(speaker="warsh", date="2026-04-21", text="inflation", context_type="hearing")
    assert e.speaker == "warsh" and e.date == "2026-04-21"


def test_jsonl_loader_reads_and_sorts_by_date(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(
        json.dumps({"date": "2026-01-01", "text": "b", "context_type": "speech"}) + "\n"
        + json.dumps({"date": "2006-01-01", "text": "a", "context_type": "speech"}) + "\n"
    )
    docs = JsonlCorpusLoader(p).docs_for("warsh")
    assert [d["date"] for d in docs] == ["2006-01-01", "2026-01-01"]  # ascending

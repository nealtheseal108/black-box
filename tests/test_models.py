import json
from src.warsh.models import Document, to_jsonl_line, from_jsonl_line, build_manifest


def test_document_roundtrips_through_jsonl():
    doc = Document(
        id="fraser-2007-market-liquidity",
        source="fraser",
        title="Market Liquidity and Risk",
        date="2007-03-05",
        url="https://fraser.stlouisfed.org/x",
        context_type="speech",
        text="Liquidity is a state of mind.",
    )
    line = to_jsonl_line(doc)
    assert "\n" not in line
    back = from_jsonl_line(line)
    assert back == doc
    assert back.word_count == 6  # derived, not stored


def test_build_manifest_summarizes_corpus():
    docs = [
        Document("a", "fraser", "T1", "2007-01-01", "u", "speech", "one two three"),
        Document("b", "hoover", "T2", "2025-04-01", "u", "essay", "four five"),
    ]
    m = build_manifest(docs)
    assert m["document_count"] == 2
    assert m["total_words"] == 5
    assert m["by_source"] == {"fraser": 1, "hoover": 1}

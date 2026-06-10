import json
from pathlib import Path
from src.warsh.models import Document
from scrape_warsh import write_corpus, dedupe_documents


def test_dedupe_keeps_first_by_id():
    a = Document("x", "fed", "T", "2007-01-01", "u", "speech", "alpha")
    b = Document("x", "fraser", "T", "2007-01-01", "u2", "speech", "beta")
    c = Document("y", "fed", "T2", "2008-01-01", "u3", "speech", "gamma")
    out = dedupe_documents([a, b, c])
    assert [d.id for d in out] == ["x", "y"]
    assert out[0].text == "alpha"  # first wins


def test_write_corpus_emits_jsonl_and_manifest(tmp_path: Path):
    docs = [Document("x", "fed", "T", "2007-01-01", "u", "speech", "one two three")]
    out = tmp_path / "warsh_corpus.jsonl"
    write_corpus(docs, out)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["id"] == "x"
    manifest = json.loads((tmp_path / "warsh_corpus.manifest.json").read_text())
    assert manifest["document_count"] == 1
    assert manifest["total_words"] == 3

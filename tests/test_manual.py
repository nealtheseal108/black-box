from pathlib import Path
from src.warsh.manual import ingest_manual_dir


def test_ingest_reads_txt_with_frontmatter(tmp_path: Path):
    f = tmp_path / "warsh-hearing-2026-04-21.txt"
    f.write_text(
        "title: Senate Confirmation Hearing\n"
        "date: 2026-04-21\n"
        "context_type: hearing\n"
        "url: manual\n"
        "---\n"
        "Senator, inflation is a choice. I will be foursquare within our role.\n",
        encoding="utf-8",
    )
    docs = ingest_manual_dir(tmp_path)
    assert len(docs) == 1
    d = docs[0]
    assert d.source == "manual"
    assert d.title == "Senate Confirmation Hearing"
    assert d.date == "2026-04-21"
    assert d.context_type == "hearing"
    assert "foursquare within our role" in d.text
    assert "title:" not in d.text  # frontmatter stripped


def test_ingest_falls_back_to_filename_when_no_frontmatter(tmp_path: Path):
    (tmp_path / "some-wsj-oped.txt").write_text("The Fed must not stray afield.", encoding="utf-8")
    docs = ingest_manual_dir(tmp_path)
    assert len(docs) == 1
    assert docs[0].title == "some-wsj-oped"
    assert "afield" in docs[0].text

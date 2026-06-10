# C1 — Warsh Corpus Scraper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scrape_warsh.py` — a 7-source scraper that assembles Kevin Warsh's full corpus into a normalized `corpus/warsh_corpus.jsonl`, with all parsing/normalization logic unit-tested offline and a thin network layer Neal runs locally (agent sandbox blocks every target domain).

**Architecture:** Separate **pure transforms** (HTML/PDF → clean `Document`, fully testable on inline fixtures) from **I/O** (network fetch + file write, owner-run). Sources funnel through one normalization path and emit one `Document` per text. A `corpus/manual/` ingestion path absorbs paywalled material (WSJ, the April-21 hearing) that Neal drops in by hand. Output is append-only JSONL plus a manifest for provenance.

**Tech Stack:** Python 3.11+, `requests` (fetch), `beautifulsoup4` (HTML), `pypdf` (PDF text), `pytest`. Pure functions accept already-fetched `str`/`bytes` so tests need no network and no heavy deps mocked.

---

## File Structure

- `src/warsh/models.py` — `Document` dataclass + JSONL (de)serialization, `build_manifest()`
- `src/warsh/normalize.py` — `clean_html()`, `normalize_whitespace()`, `word_count()`, `tag_context_type()`
- `src/warsh/parsers.py` — per-source parsers: `parse_fraser()`, `parse_fed_speech()`, `parse_hoover_html()`, `parse_pdf()`
- `src/warsh/manual.py` — `ingest_manual_dir()` for `corpus/manual/`
- `scrape_warsh.py` — owner-run CLI: `fetch()` + source registry + orchestration → `corpus/warsh_corpus.jsonl`
- `tests/test_models.py`, `tests/test_normalize.py`, `tests/test_parsers.py`, `tests/test_manual.py`

Sources (Brief §3): FRASER Governor speeches 2006–11 · FederalReserve.gov fallback · Hoover pages · Hoover PDFs (incl. "Commanding Heights" Apr 2025) · WSJ via archive.org · Senate hearing (April 21 2026) · `corpus/manual/` for paywalled.

---

### Task 1: Document model + JSONL + manifest

**Files:**
- Create: `src/warsh/__init__.py` (empty)
- Create: `src/warsh/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.warsh.models'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/warsh/models.py
import json
from dataclasses import dataclass, field


@dataclass
class Document:
    id: str
    source: str       # fraser | fed | hoover | hoover_pdf | wsj_archive | hearing | manual
    title: str
    date: str         # ISO YYYY-MM-DD, "" if unknown
    url: str
    context_type: str # speech | essay | op_ed | interview | hearing | lecture
    text: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def to_jsonl_line(doc: Document) -> str:
    return json.dumps(
        {
            "id": doc.id, "source": doc.source, "title": doc.title,
            "date": doc.date, "url": doc.url, "context_type": doc.context_type,
            "text": doc.text,
        },
        ensure_ascii=False,
    )


def from_jsonl_line(line: str) -> Document:
    d = json.loads(line)
    return Document(
        id=d["id"], source=d["source"], title=d["title"], date=d["date"],
        url=d["url"], context_type=d["context_type"], text=d["text"],
    )


def build_manifest(docs: list[Document]) -> dict:
    by_source: dict[str, int] = {}
    for d in docs:
        by_source[d.source] = by_source.get(d.source, 0) + 1
    return {
        "document_count": len(docs),
        "total_words": sum(d.word_count for d in docs),
        "by_source": by_source,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_models.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/warsh/__init__.py src/warsh/models.py tests/test_models.py
git commit -m "feat(c1): Document model with JSONL serialization and corpus manifest"
```

---

### Task 2: Text normalization

**Files:**
- Create: `src/warsh/normalize.py`
- Test: `tests/test_normalize.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_normalize.py
from src.warsh.normalize import clean_html, normalize_whitespace, word_count, tag_context_type


def test_clean_html_strips_tags_scripts_and_styles():
    html = """
    <html><head><style>.x{color:red}</style></head>
    <body><script>evil()</script>
    <p>Inflation is <b>a choice</b>.</p>
    <p>Without excuse or equivocation.</p></body></html>
    """
    out = clean_html(html)
    assert "evil" not in out
    assert "color:red" not in out
    assert "Inflation is a choice." in out
    assert "Without excuse or equivocation." in out


def test_normalize_whitespace_collapses_runs_and_trims():
    assert normalize_whitespace("  a\n\n\n  b\t\tc  ") == "a\n\nb c"


def test_word_count_counts_tokens():
    assert word_count("one two   three\nfour") == 4


def test_tag_context_type_classifies_by_source_and_title():
    assert tag_context_type("hearing", "Nomination Hearing") == "hearing"
    assert tag_context_type("fraser", "Remarks on Liquidity") == "speech"
    assert tag_context_type("wsj_archive", "The Fed's Mission Creep") == "op_ed"
    assert tag_context_type("hoover", "Commanding Heights Lecture") == "lecture"
    assert tag_context_type("hoover", "A Conversation with Kevin Warsh") == "interview"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_normalize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.warsh.normalize'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/warsh/normalize.py
import re
from bs4 import BeautifulSoup


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return normalize_whitespace(text)


def normalize_whitespace(text: str) -> str:
    # Collapse intra-line whitespace to single spaces, preserve paragraph breaks.
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(text.split())


def tag_context_type(source: str, title: str) -> str:
    t = title.lower()
    if source == "hearing" or "hearing" in t:
        return "hearing"
    if source == "wsj_archive" or "op-ed" in t or "mission creep" in t:
        return "op_ed"
    if source in ("hoover", "hoover_pdf"):
        if "conversation" in t or "interview" in t:
            return "interview"
        if "lecture" in t or "commanding heights" in t:
            return "lecture"
        return "essay"
    return "speech"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_normalize.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/warsh/normalize.py tests/test_normalize.py
git commit -m "feat(c1): HTML cleaning, whitespace normalization, context tagging"
```

---

### Task 3: Source parsers (FRASER, Fed, Hoover HTML, PDF)

**Files:**
- Create: `src/warsh/parsers.py`
- Test: `tests/test_parsers.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parsers.py
from src.warsh.parsers import parse_fed_speech, parse_hoover_html


FED_HTML = """
<html><body>
<h1 class="title">Market Liquidity and Risk</h1>
<p class="article__time">March 5, 2007</p>
<div id="article">
  <p>Inflation is a choice, not an accident.</p>
  <p>The balance sheet has become bloated.</p>
</div>
</body></html>
"""

HOOVER_HTML = """
<html><body>
<h1>Commanding Heights</h1>
<time datetime="2025-04-15">April 15, 2025</time>
<article><p>Regime change in monetary policy is overdue.</p></article>
</body></html>
"""


def test_parse_fed_speech_extracts_title_date_text():
    doc = parse_fed_speech(FED_HTML, url="https://federalreserve.gov/x.htm")
    assert doc.source == "fed"
    assert doc.title == "Market Liquidity and Risk"
    assert doc.date == "2007-03-05"
    assert "Inflation is a choice" in doc.text
    assert "bloated" in doc.text
    assert doc.context_type == "speech"
    assert doc.id  # non-empty slug


def test_parse_hoover_html_extracts_iso_date_from_time_tag():
    doc = parse_hoover_html(HOOVER_HTML, url="https://hoover.org/y")
    assert doc.source == "hoover"
    assert doc.date == "2025-04-15"
    assert doc.context_type == "lecture"
    assert "Regime change" in doc.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_parsers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.warsh.parsers'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/warsh/parsers.py
import re
from datetime import datetime
from bs4 import BeautifulSoup
from src.warsh.models import Document
from src.warsh.normalize import clean_html, normalize_whitespace, tag_context_type

_MONTHS = {m: i for i, m in enumerate(
    ["january","february","march","april","may","june","july",
     "august","september","october","november","december"], start=1)}


def _slug(title: str, date: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60]
    return f"{date or 'undated'}-{base}".strip("-")


def _parse_date(text: str) -> str:
    text = text.strip()
    # ISO already?
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if m:
        return m.group(0)
    # "March 5, 2007"
    m = re.search(r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})", text)
    if m and m.group(1).lower() in _MONTHS:
        mo, d, y = _MONTHS[m.group(1).lower()], int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"
    return ""


def _doc_from_parts(source, title, date, url, body_html_or_text, is_html=True):
    text = clean_html(body_html_or_text) if is_html else normalize_whitespace(body_html_or_text)
    ctype = tag_context_type(source, title)
    return Document(
        id=_slug(title, date), source=source, title=title,
        date=date, url=url, context_type=ctype, text=text,
    )


def parse_fed_speech(html: str, url: str) -> Document:
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.find(class_="title") or soup.find("h1")).get_text(strip=True)
    date_node = soup.find(class_="article__time") or soup.find("time")
    date = _parse_date(date_node.get_text() if date_node else "")
    body = soup.find(id="article") or soup.find("article") or soup.body
    return _doc_from_parts("fed", title, date, url, str(body), is_html=True)


def parse_fraser(html: str, url: str) -> Document:
    doc = parse_fed_speech(html, url)
    doc.source = "fraser"
    doc.context_type = tag_context_type("fraser", doc.title)
    return doc


def parse_hoover_html(html: str, url: str) -> Document:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("h1").get_text(strip=True)
    time_node = soup.find("time")
    raw_date = time_node.get("datetime") if time_node and time_node.get("datetime") \
        else (time_node.get_text() if time_node else "")
    date = _parse_date(raw_date)
    body = soup.find("article") or soup.body
    return _doc_from_parts("hoover", title, date, url, str(body), is_html=True)


def parse_pdf(pdf_bytes: bytes, url: str, title: str, date: str, source: str = "hoover_pdf") -> Document:
    from io import BytesIO
    from pypdf import PdfReader
    reader = PdfReader(BytesIO(pdf_bytes))
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return _doc_from_parts(source, title, date, url, text, is_html=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_parsers.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/warsh/parsers.py tests/test_parsers.py
git commit -m "feat(c1): FRASER/Fed/Hoover HTML + PDF parsers with date extraction"
```

---

### Task 4: Manual ingestion of paywalled material

**Files:**
- Create: `src/warsh/manual.py`
- Test: `tests/test_manual.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_manual.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_manual.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.warsh.manual'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/warsh/manual.py
from pathlib import Path
from src.warsh.models import Document
from src.warsh.normalize import normalize_whitespace, tag_context_type


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    if "\n---\n" not in raw:
        return {}, raw
    head, body = raw.split("\n---\n", 1)
    meta = {}
    for line in head.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    # Only treat as frontmatter if it looks like key:value metadata
    if not meta:
        return {}, raw
    return meta, body


def ingest_manual_dir(path) -> list[Document]:
    path = Path(path)
    docs: list[Document] = []
    for f in sorted(path.glob("*.txt")) + sorted(path.glob("*.md")):
        raw = f.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        title = meta.get("title", f.stem)
        date = meta.get("date", "")
        source = "manual"
        ctype = meta.get("context_type") or tag_context_type(source, title)
        docs.append(Document(
            id=f"manual-{f.stem}", source=source, title=title, date=date,
            url=meta.get("url", "manual"), context_type=ctype,
            text=normalize_whitespace(body),
        ))
    return docs
```

- [ ] **Step 2 note:** `tag_context_type("manual", "...hearing...")` returns "hearing" via the title check, so frontmatter-less hearing files still classify correctly.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_manual.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/warsh/manual.py tests/test_manual.py
git commit -m "feat(c1): manual corpus ingestion with frontmatter for paywalled docs"
```

---

### Task 5: Owner-run orchestration CLI

**Files:**
- Create: `scrape_warsh.py`
- Create: `requirements.txt`
- Test: `tests/test_orchestration.py` (tests the pure registry/dedupe/write logic, not network)

> The `fetch()` function is the only network code. It is **OWNER-EXECUTED** — Neal runs `python scrape_warsh.py`; the agent sandbox blocks fraser.stlouisfed.org, federalreserve.gov, hoover.org, web.archive.org. Tests cover everything *except* live fetch by injecting a fake fetcher.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_orchestration.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_orchestration.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scrape_warsh'`

- [ ] **Step 3: Write minimal implementation**

```python
# scrape_warsh.py
"""C1 — Warsh corpus scraper. OWNER-EXECUTED: run locally; agent sandbox blocks target domains.

Usage:
    python scrape_warsh.py                 # fetch all sources + ingest corpus/manual/
    python scrape_warsh.py --manual-only   # skip network; only re-ingest corpus/manual/
"""
import argparse
import json
import sys
import time
from pathlib import Path

from src.warsh.models import Document, to_jsonl_line, build_manifest
from src.warsh.manual import ingest_manual_dir
from src.warsh import parsers

CORPUS_DIR = Path("corpus")
OUT_PATH = CORPUS_DIR / "warsh_corpus.jsonl"

# (source_url, parser, kwargs) — seed registry; expand as URLs are confirmed.
# FRASER 2006-11 Governor speeches, Fed fallback, Hoover pages/PDFs, WSJ via archive.org, hearing.
SOURCES = [
    # ("https://fraser.stlouisfed.org/...", parsers.parse_fraser, {}),
    # ("https://www.hoover.org/...",        parsers.parse_hoover_html, {}),
]


def fetch(url: str) -> bytes:
    """Network I/O — owner-run only."""
    import requests
    resp = requests.get(url, headers={"User-Agent": "speechedge-corpus/1.0"}, timeout=30)
    resp.raise_for_status()
    return resp.content


def dedupe_documents(docs: list[Document]) -> list[Document]:
    seen: set[str] = set()
    out: list[Document] = []
    for d in docs:
        if d.id in seen:
            continue
        seen.add(d.id)
        out.append(d)
    return out


def write_corpus(docs: list[Document], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(to_jsonl_line(d) + "\n")
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(build_manifest(docs), indent=2), encoding="utf-8")


def collect(manual_only: bool = False) -> list[Document]:
    docs: list[Document] = []
    if not manual_only:
        for url, parser, kwargs in SOURCES:
            try:
                raw = fetch(url)
                html = raw.decode("utf-8", errors="replace")
                docs.append(parser(html, url=url, **kwargs))
                print(f"[ok]   {url}", file=sys.stderr)
                time.sleep(1.0)  # be polite
            except Exception as e:  # noqa: BLE001 — owner sees per-source failures
                print(f"[fail] {url}: {e}", file=sys.stderr)
    manual = ingest_manual_dir(CORPUS_DIR / "manual")
    print(f"[manual] {len(manual)} document(s) ingested", file=sys.stderr)
    docs.extend(manual)
    return dedupe_documents(docs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual-only", action="store_true")
    args = ap.parse_args()
    docs = collect(manual_only=args.manual_only)
    write_corpus(docs, OUT_PATH)
    m = build_manifest(docs)
    print(f"Wrote {m['document_count']} docs / {m['total_words']} words → {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

```text
# requirements.txt
requests>=2.31
beautifulsoup4>=4.12
pypdf>=4.0
pytest>=8.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_orchestration.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full suite + commit**

Run: `python -m pytest -v`
Expected: PASS (all tasks' tests green)

```bash
git add scrape_warsh.py requirements.txt tests/test_orchestration.py
git commit -m "feat(c1): owner-run orchestration CLI with dedupe, JSONL + manifest output"
```

---

## Owner hand-off note (R1 / R2)

After this plan is green, Neal runs:
```bash
cd ~/Downloads/SpeechEdge && pip install -r requirements.txt && python scrape_warsh.py
```
Then drops the **April 21 2026 confirmation hearing** transcript and paywalled WSJ op-eds into `corpus/manual/` (one `.txt` per doc, optional `title:/date:/context_type:` frontmatter + `---`) and re-runs `python scrape_warsh.py --manual-only`. The `SOURCES` registry is seeded empty-but-structured; confirmed URLs get appended as Neal verifies them live (the agent can't reach the domains to confirm them).

## Self-review
- **Spec coverage:** 7 sources of Brief §3 → FRASER (`parse_fraser`), Fed fallback (`parse_fed_speech`), Hoover pages (`parse_hoover_html`), Hoover PDFs (`parse_pdf`), WSJ via archive.org (HTML → `parse_fed_speech`/`parse_hoover_html` on the archived page), hearing + paywalled (`ingest_manual_dir`). ✓
- **Placeholders:** `SOURCES` is intentionally seeded empty because live URLs can't be agent-verified (sandbox) — this is a documented owner step, not a code placeholder. All functions are fully implemented.
- **Type consistency:** `Document` field order/names identical across all tasks; `parse_*` all return `Document`; `tag_context_type(source, title)` signature consistent in normalize/parsers/manual.

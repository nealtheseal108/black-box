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
_FED = "https://www.federalreserve.gov/newsevents/speech/"
# Verified live (2026-06-11) — Warsh's complete Board-of-Governors speech archive 2006–2010.
SOURCES = [
    (_FED + "warsh20060718a.htm", parsers.parse_fed_speech, {}),
    (_FED + "warsh20061121a.htm", parsers.parse_fed_speech, {}),
    (_FED + "warsh20070305a.htm", parsers.parse_fed_speech, {}),
    (_FED + "warsh20070605a.htm", parsers.parse_fed_speech, {}),
    (_FED + "warsh20070921a.htm", parsers.parse_fed_speech, {}),
    (_FED + "warsh20071005a.htm", parsers.parse_fed_speech, {}),
    (_FED + "warsh20071107a.htm", parsers.parse_fed_speech, {}),
    (_FED + "warsh20080414a.htm", parsers.parse_fed_speech, {}),
    (_FED + "warsh20080521a.htm", parsers.parse_fed_speech, {}),
    (_FED + "warsh20080728a.htm", parsers.parse_fed_speech, {}),
    (_FED + "warsh20081106a.htm", parsers.parse_fed_speech, {}),
    (_FED + "warsh20090406a.htm", parsers.parse_fed_speech, {}),
    (_FED + "warsh20090616a.htm", parsers.parse_fed_speech, {}),
    (_FED + "warsh20090925a.htm", parsers.parse_fed_speech, {}),
    (_FED + "warsh20100203a.htm", parsers.parse_fed_speech, {}),
    (_FED + "warsh20100326a.htm", parsers.parse_fed_speech, {}),
    (_FED + "warsh20100628a.htm", parsers.parse_fed_speech, {}),
    (_FED + "warsh20101108a.htm", parsers.parse_fed_speech, {}),
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


def write_corpus(docs: list[Document], out_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(to_jsonl_line(d) + "\n")
    manifest = build_manifest(docs)
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


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
    m = write_corpus(docs, OUT_PATH)
    print(f"Wrote {m['document_count']} docs / {m['total_words']} words → {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()

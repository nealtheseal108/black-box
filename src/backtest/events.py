"""Events and a speaker-keyed corpus loader for the calibration harness.

An Event is one held-out speaking occasion. CorpusLoader is the seam that lets a
future Powell/SOTU corpus plug in behind the same interface as Warsh's jsonl.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Event:
    speaker: str
    date: str
    text: str
    context_type: str


class CorpusLoader(Protocol):
    def docs_for(self, speaker: str) -> list[dict]:
        """Return that speaker's dated docs (each a dict with date/text/context_type)."""
        ...


class JsonlCorpusLoader:
    """CorpusLoader over a single-speaker JSONL corpus (e.g. corpus/warsh_corpus.jsonl)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def docs_for(self, speaker: str) -> list[dict]:
        docs = [json.loads(l) for l in self._path.read_text().splitlines() if l.strip()]
        return sorted(docs, key=lambda d: d.get("date", ""))

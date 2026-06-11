"""Context assembly for C4 Mode-1 Context Agent.

`assemble_context` orchestrates a set of injectable source fetchers into a
`MacroSnapshot`.  Each fetcher is a ``Callable[[], list[str]]``.  If a fetcher
raises, the error is logged to stderr and that slot is left as ``[]``, so a
single broken data source never aborts a run.
"""
from __future__ import annotations

import sys
from typing import Callable, Dict, List

from src.agents.context_types import MacroSnapshot

# Type alias: each fetcher returns a list of plain-text snippets.
Fetcher = Callable[[], List[str]]
FetcherMap = Dict[str, Fetcher]

_KNOWN_SLOTS = ("data_prints", "futures", "news", "speaker_recent")


def _safe_fetch(name: str, fn: Fetcher) -> List[str]:
    """Call *fn* and return its result; log + return [] on any exception."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        print(f"[assemble] fetcher '{name}' failed: {exc}", file=sys.stderr)
        return []


def assemble_context(as_of: str, fetchers: FetcherMap) -> MacroSnapshot:
    """Gather all source snippets into a :class:`MacroSnapshot`.

    Parameters
    ----------
    as_of:
        ISO-8601 timestamp for this assembly run.
    fetchers:
        Mapping of slot name → callable.  Expected keys: ``data_prints``,
        ``futures``, ``news``, ``speaker_recent``.  Unknown keys are ignored;
        missing keys default to ``[]``.
    """
    slots: Dict[str, List[str]] = {slot: [] for slot in _KNOWN_SLOTS}
    for slot in _KNOWN_SLOTS:
        if slot in fetchers:
            slots[slot] = _safe_fetch(slot, fetchers[slot])

    return MacroSnapshot(
        as_of=as_of,
        data_prints=slots["data_prints"],
        futures=slots["futures"],
        news=slots["news"],
        speaker_recent=slots["speaker_recent"],
    )

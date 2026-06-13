"""Caption sources for the live loop.

CaptionStream.stream() yields text deltas. ReplayStream feeds a stored transcript
(offline tests / the hearing replay). StreamTextClient polls the Fed's StreamText
Realtime Caption Pull API (JSON, moving `last` cursor); `requests` is imported
lazily so the test suite never needs network or the dependency.
"""
from __future__ import annotations

from typing import Callable, Iterable, Protocol


class CaptionStream(Protocol):
    def stream(self) -> Iterable[str]:
        """Yield transcript text deltas as they arrive."""
        ...


class ReplayStream:
    """Replay a stored transcript as fixed-size word chunks (no network)."""

    def __init__(self, text: str, chunk_words: int = 20) -> None:
        self._words = text.split()
        self._chunk = chunk_words

    def stream(self) -> Iterable[str]:
        for i in range(0, len(self._words), self._chunk):
            yield " ".join(self._words[i:i + self._chunk])


def _live_fetch(event: str, last: int) -> dict:
    import requests  # lazy: only the live path needs it
    url = "https://www.streamtext.net/captions"
    resp = requests.get(url, params={"event": event, "last": last, "length": 80, "language": "en"}, timeout=15)
    resp.raise_for_status()
    return resp.json()


class StreamTextClient:
    """Poll the StreamText Realtime Caption Pull API, yielding caption content deltas.

    `fetch(event, last) -> {"content","lastPosition",...}` is injectable for tests.
    Stops after `poll_idle_limit` consecutive empty pages (live: raise the limit / loop).
    """

    def __init__(self, event_id: str, fetch: Callable[[str, int], dict] | None = None,
                 poll_idle_limit: int = 3) -> None:
        self._event = event_id
        self._fetch = fetch or _live_fetch
        self._idle_limit = poll_idle_limit

    def stream(self) -> Iterable[str]:
        last = 0
        idle = 0
        while idle < self._idle_limit:
            page = self._fetch(self._event, last)
            content = page.get("content", "")
            if content:
                yield content
                last = page.get("lastPosition", last)
                idle = 0
            else:
                idle += 1

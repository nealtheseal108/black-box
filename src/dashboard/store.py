"""State store protocol and implementations for the dashboard."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class StateStore(Protocol):
    """Returns a snapshot of current pipeline state."""

    def snapshot(self) -> tuple[str, dict, list[dict], list[dict], dict]:
        """Returns (transcript, tone, markets, fills, resolutions)."""
        ...


# ---------------------------------------------------------------------------
# In-memory store (for tests)
# ---------------------------------------------------------------------------

class InMemoryStateStore:
    """Holds fixed state; useful for unit/integration tests."""

    def __init__(
        self,
        transcript: str = "",
        tone: dict[str, Any] | None = None,
        markets: list[dict[str, Any]] | None = None,
        fills: list[dict[str, Any]] | None = None,
        resolutions: dict[str, int] | None = None,
    ) -> None:
        self._transcript = transcript
        self._tone = tone or {"hawkish": 0.0, "dovish": 0.0, "independence": 0.0, "qt": 0.0}
        self._markets = markets or []
        self._fills = fills or []
        self._resolutions = resolutions or {}

    def snapshot(self) -> tuple[str, dict, list[dict], list[dict], dict]:
        return (
            self._transcript,
            self._tone,
            self._markets,
            self._fills,
            self._resolutions,
        )


# ---------------------------------------------------------------------------
# File-backed store (production)
# ---------------------------------------------------------------------------

class FileStateStore:
    """Reads pipeline artifacts from disk on every snapshot() call.

    - ``paper_log_path``: JSONL of paper fills written by C7 (output/paper_trades.jsonl)
    - ``live_state_path``: JSON blob written by the live predictor with keys
      ``transcript``, ``tone``, ``markets``, ``resolutions``

    Missing files are handled gracefully — returns empty defaults.
    """

    def __init__(self, paper_log_path: Path, live_state_path: Path) -> None:
        self.paper_log_path = Path(paper_log_path)
        self.live_state_path = Path(live_state_path)

    def _read_fills(self) -> list[dict[str, Any]]:
        if not self.paper_log_path.exists():
            return []
        fills: list[dict[str, Any]] = []
        for line in self.paper_log_path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    fills.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return fills

    def _read_live_state(self) -> dict[str, Any]:
        if not self.live_state_path.exists():
            return {}
        try:
            return json.loads(self.live_state_path.read_text())
        except json.JSONDecodeError:
            return {}

    def snapshot(self) -> tuple[str, dict, list[dict], list[dict], dict]:
        fills = self._read_fills()
        live = self._read_live_state()
        transcript: str = live.get("transcript", "")
        tone: dict = live.get("tone", {"hawkish": 0.0, "dovish": 0.0, "independence": 0.0, "qt": 0.0})
        markets: list[dict] = live.get("markets", [])
        resolutions: dict = live.get("resolutions", {})
        return transcript, tone, markets, fills, resolutions

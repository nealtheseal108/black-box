"""SpeechEdge dashboard entry point.

Usage:
    python dashboard.py
    # then open http://127.0.0.1:8000
"""
from __future__ import annotations

from pathlib import Path

import uvicorn

from src.dashboard.store import FileStateStore

_REPO_ROOT = Path(__file__).parent
_DEFAULT_PAPER_LOG = _REPO_ROOT / "output" / "paper_trades.jsonl"
_DEFAULT_LIVE_STATE = _REPO_ROOT / "output" / "live_state.json"


def build_default_store(
    paper_log: Path = _DEFAULT_PAPER_LOG,
    live_state: Path = _DEFAULT_LIVE_STATE,
) -> FileStateStore:
    """Build a FileStateStore from default (or overridden) artifact paths."""
    return FileStateStore(paper_log_path=paper_log, live_state_path=live_state)


def main() -> None:
    from src.dashboard.app import create_app  # local import avoids circular issues at module level

    store = build_default_store()
    uvicorn.run(create_app(store), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()

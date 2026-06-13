"""RL-episode logger — collection only.

Appends (state, action, reward) tuples to a JSONL replay buffer for FUTURE offline
training. The live loop never trains or updates weights (A.3); it only records.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class EpisodeLogger:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def log(self, state: dict, action: dict, reward: float) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "state": state,
            "action": action,
            "reward": reward,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

from pathlib import Path
from dashboard import build_default_store
from src.dashboard.app import create_app
from fastapi.testclient import TestClient


def test_default_store_app_boots_with_empty_artifacts(tmp_path: Path):
    # no artifacts present → app still serves a valid (empty) state, no crash
    store = build_default_store(paper_log=tmp_path / "none.jsonl", live_state=tmp_path / "none.json")
    client = TestClient(create_app(store))
    body = client.get("/api/state").json()
    assert body["transcript"] == ""
    assert body["markets"] == []
    assert body["pnl"]["realized"] == 0.0

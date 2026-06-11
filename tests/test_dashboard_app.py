from fastapi.testclient import TestClient
from src.dashboard.store import InMemoryStateStore
from src.dashboard.app import create_app


def _store():
    return InMemoryStateStore(
        transcript="inflation is a choice",
        tone={"hawkish": 0.9, "dovish": 0.0, "independence": 0.0, "qt": 0.0},
        markets=[{"ticker": "FED-HOLD-JUN", "prior_prob": 0.55, "model_prob": 0.66,
                  "yes_price": 0.55, "edge": 0.11, "side": "yes"}],
        fills=[{"ticker": "FED-HOLD-JUN", "side": "yes", "count": 10, "fill_price": 0.55,
                "simulated": True, "timestamp": "t"}],
        resolutions={},
    )


def test_api_state_returns_assembled_state():
    client = TestClient(create_app(_store()))
    r = client.get("/api/state")
    assert r.status_code == 200
    body = r.json()
    assert body["transcript"] == "inflation is a choice"
    assert body["markets"][0]["edge"] == 0.11
    assert body["pnl"]["open_positions"] == 1


def test_root_serves_html():
    client = TestClient(create_app(_store()))
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]

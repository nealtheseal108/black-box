# C8 — Dashboard (FastAPI + simple frontend) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: subagent-driven-development / executing-plans. Reuses C2 (diction tone), C6 (market states/posteriors), C7 (paper fills). Pure aggregation + P&L are unit-tested; FastAPI endpoints tested with `TestClient` (in-process, no network).

**Goal:** Build `dashboard.py` + `src/dashboard/` — a FastAPI app exposing live SpeechEdge state with a minimal polled frontend (Brief §3 panels): live transcript, tone meter, market cards (prior → posterior vs price, edge), micro-bet log, and P&L. One stack, one process for the June-16 run; reads the same artifacts the pipeline writes.

**Architecture:** Pure core (unit-tested): `compute_pnl(fills, resolutions)` and `build_state(...)` that assembles a `DashboardState` from transcript + diction tone + market states + fills. A file-backed `StateStore` reads `output/paper_trades.jsonl` (C7) + a live state JSON (transcript/tone/markets the predictor writes) — injectable so tests use fakes. FastAPI `create_app(store)` serves `/` (HTML) and `/api/state` (JSON); the frontend polls `/api/state` every second.

**Tech Stack:** Python 3.11+, `fastapi`, `uvicorn`, `pydantic` (already a dep), `httpx` (TestClient), pytest. Frontend: one static `index.html` with vanilla JS — no build step.

---

## File Structure
- `src/dashboard/state.py` — `DashboardState`, `MarketCard`, `BetLogEntry`, `ToneMeter` (pydantic); `compute_pnl`, `build_state`
- `src/dashboard/store.py` — `StateStore` protocol; `FileStateStore` (reads artifacts); `InMemoryStateStore` (tests)
- `src/dashboard/app.py` — `create_app(store) -> FastAPI`
- `src/dashboard/static/index.html` — polled dashboard UI
- `dashboard.py` — entry: build `FileStateStore`, `uvicorn.run(create_app(store))`
- Tests under `tests/`.

---

### Task 1: P&L computation (pure)

**Files:** Create `src/dashboard/__init__.py`, `src/dashboard/state.py` (P&L part); Test `tests/test_pnl.py`

- [ ] **Step 1: Failing test**
```python
# tests/test_pnl.py
from src.dashboard.state import compute_pnl

def test_pnl_yes_win_and_loss():
    fills = [{"ticker": "A", "side": "yes", "count": 10, "fill_price": 0.55}]
    assert abs(compute_pnl(fills, {"A": 1})["realized"] - 10 * (1 - 0.55)) < 1e-9   # YES wins
    assert abs(compute_pnl(fills, {"A": 0})["realized"] - (-10 * 0.55)) < 1e-9       # YES loses

def test_pnl_no_side_wins_when_outcome_zero():
    fills = [{"ticker": "B", "side": "no", "count": 4, "fill_price": 0.40}]
    assert abs(compute_pnl(fills, {"B": 0})["realized"] - 4 * (1 - 0.40)) < 1e-9      # NO wins
    assert abs(compute_pnl(fills, {"B": 1})["realized"] - (-4 * 0.40)) < 1e-9

def test_pnl_unresolved_counts_as_open_not_realized():
    fills = [{"ticker": "C", "side": "yes", "count": 5, "fill_price": 0.5}]
    r = compute_pnl(fills, {})            # no resolution yet
    assert r["realized"] == 0.0
    assert r["open_positions"] == 1
```
- [ ] **Step 2: Run — FAIL**
- [ ] **Step 3: Implement**
```python
# src/dashboard/state.py  (P&L section)
def compute_pnl(fills: list[dict], resolutions: dict[str, int]) -> dict:
    """Kalshi binary contract: pays $1 if your side wins, $0 if it loses.
    realized P&L per fill = (1 - fill_price) on a win, -fill_price on a loss."""
    realized = 0.0
    open_positions = 0
    for f in fills:
        outcome = resolutions.get(f["ticker"])
        if outcome is None:
            open_positions += 1
            continue
        win = (f["side"] == "yes" and outcome == 1) or (f["side"] == "no" and outcome == 0)
        per = (1 - f["fill_price"]) if win else -f["fill_price"]
        realized += f["count"] * per
    return {"realized": round(realized, 4), "open_positions": open_positions}
```
- [ ] **Step 4: Run — PASS**
- [ ] **Step 5: Commit** `feat(c8): P&L computation for paper fills`

---

### Task 2: DashboardState builder (pure)

**Files:** Extend `src/dashboard/state.py`; Test `tests/test_dashboard_state.py`

- [ ] **Step 1: Failing test**
```python
# tests/test_dashboard_state.py
from src.dashboard.state import build_state

def test_build_state_assembles_all_panels():
    transcript = "inflation is a choice"
    tone = {"hawkish": 0.9, "dovish": 0.0, "independence": 0.0, "qt": 0.0}
    markets = [{"ticker": "FED-HOLD-JUN", "prior_prob": 0.55, "model_prob": 0.66,
                "yes_price": 0.55, "edge": 0.11, "side": "yes"}]
    fills = [{"ticker": "FED-HOLD-JUN", "side": "yes", "count": 10, "fill_price": 0.55,
              "simulated": True, "timestamp": "t"}]
    st = build_state(transcript, tone, markets, fills, resolutions={})
    d = st.model_dump()
    assert d["transcript"] == transcript
    assert d["tone"]["hawkish"] == 0.9
    assert d["markets"][0]["ticker"] == "FED-HOLD-JUN" and d["markets"][0]["edge"] == 0.11
    assert d["bets"][0]["count"] == 10 and d["bets"][0]["simulated"] is True
    assert d["pnl"]["open_positions"] == 1
```
- [ ] **Step 2: Run — FAIL**
- [ ] **Step 3: Implement** pydantic models `ToneMeter`, `MarketCard`, `BetLogEntry`, `DashboardState{transcript, tone, markets, bets, pnl}` and `build_state(transcript, tone, markets, fills, resolutions)` that maps dicts → models and calls `compute_pnl`.
- [ ] **Step 4: Run — PASS**
- [ ] **Step 5: Commit** `feat(c8): DashboardState builder assembling all panels`

---

### Task 3: State store + FastAPI app

**Files:** Create `src/dashboard/store.py`, `src/dashboard/app.py`; Test `tests/test_dashboard_app.py`

- [ ] **Step 1: Failing test**
```python
# tests/test_dashboard_app.py
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
```
- [ ] **Step 2: Run — FAIL**
- [ ] **Step 3: Implement**
  - `StateStore` protocol: `snapshot() -> tuple(transcript, tone, markets, fills, resolutions)`.
  - `InMemoryStateStore(**kwargs)` for tests.
  - `FileStateStore(paper_log_path, live_state_path)`: reads `output/paper_trades.jsonl` (fills) + a live state JSON (transcript/tone/markets/resolutions) written by the predictor; missing files → empty defaults.
  - `create_app(store)`: FastAPI with `GET /api/state` → `build_state(*store.snapshot()).model_dump()`; `GET /` → serve `src/dashboard/static/index.html` (FileResponse).
- [ ] **Step 4: Run — PASS**
- [ ] **Step 5: Commit** `feat(c8): state store + FastAPI app (/api/state, /)`

---

### Task 4: Frontend + launch entry

**Files:** Create `src/dashboard/static/index.html`, `dashboard.py`; Test `tests/test_dashboard_entry.py`

- [ ] **Step 1: Failing test**
```python
# tests/test_dashboard_entry.py
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
```
- [ ] **Step 2: Run — FAIL**
- [ ] **Step 3: Implement**
  - `index.html`: minimal vanilla-JS page — panels for transcript, a hawkish↔dovish tone gauge, market cards (prior→posterior, price, edge, side), a bet-log table, and a P&L readout; `setInterval(fetchState, 1000)` polling `/api/state`. No framework/build.
  - `dashboard.py`: `build_default_store(paper_log, live_state)` → `FileStateStore`; `main()` → `uvicorn.run(create_app(build_default_store(...)), host="127.0.0.1", port=8000)`.
- [ ] **Step 4: Run — PASS**; full suite green.
- [ ] **Step 5: Commit** `feat(c8): dashboard frontend + uvicorn launch entry`

## Self-review
- **Spec (Brief §3 dashboard):** live transcript ✓ · tone meter (C2 diction) ✓ · market cards ✓ · micro-bet log (C7 fills) ✓ · P&L ✓. "Claude API next-phrase inference" panel can read C2 `predict_next` later — note as a follow-up, not blocking.
- **Reuse:** consumes the same artifacts C6/C7 write (`output/paper_trades.jsonl`); no duplicate sources of truth.
- **Testability:** P&L + state builder pure; endpoints via in-process `TestClient` (no network). Empty-artifact boot is tested (graceful).
- **Deps:** add `fastapi`, `uvicorn`, `httpx` to `requirements.txt`.

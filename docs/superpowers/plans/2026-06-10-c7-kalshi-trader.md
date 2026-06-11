# C7 — Kalshi Trader (Paper Mode First) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: subagent-driven-development / executing-plans. Build against `docs/INTERFACES.md` §3. **This is order-placing code — the `trading-code-reviewer` agent reviews it before merge.** Every safety invariant below is load-bearing.

**Goal:** Build `kalshi_trader.py` + `src/trading/` — the execution layer: consume `Signal`s (model_prob vs market_price), size with capped Kelly, and place orders. **Paper-trading mode is the default and the only mode that works without an explicit, loud opt-in** (Authority Matrix E4). June 16 runs paper.

**Architecture:** Three pure cores + one I/O shell. Pure (fully unit-tested, no network): **signal gate** (`abs(edge) > threshold` AND net-of-fees edge > G3 floor), **Kelly sizer** (fraction capped, refuses to size live without a bankroll), **position manager** (dedupe, aggregate exposure cap). I/O: a `KalshiClient` *protocol* with two implementations — `PaperKalshiClient` (simulated fills, default) and `LiveKalshiClient` (real REST/WebSocket, gated behind `SPEECHEDGE_MODE=live` + explicit construction). Tests target the pure cores and the paper client only.

**Tech Stack:** Python 3.11+, `requests`/`websockets` (live only), pytest. Config via env (INTERFACES §5): `SPEECHEDGE_MODE` (default `paper`), `SIGNAL_THRESHOLD` (default `0.08`), `BANKROLL` (unset until E1), `KALSHI_API_KEY`/`KALSHI_API_SECRET` (live only, never committed).

---

## File Structure
- `src/trading/types.py` — `KalshiMarket`, `Signal` (INTERFACES §3), `Order`, `Fill`, `Position`
- `src/trading/gate.py` — `passes_gate(signal, threshold, fee) -> bool`; `net_edge(signal, fee)`
- `src/trading/sizing.py` — `kelly_fraction(prob, price)`, `size_order(signal, bankroll, mode, cfg) -> Order | None`
- `src/trading/positions.py` — `PositionManager` (dedupe by (ticker,side), aggregate cap)
- `src/trading/client.py` — `KalshiClient` protocol; `PaperKalshiClient`; `LiveKalshiClient` (gated)
- `kalshi_trader.py` — CLI/loop: load config, wire client by mode, consume signals → size → place
- Tests under `tests/`.

---

### Task 1: Trading types (INTERFACES §3)

**Files:** Create `src/trading/__init__.py`, `src/trading/types.py`; Test `tests/test_trading_types.py`

- [ ] **Step 1: Failing test**
```python
# tests/test_trading_types.py
from src.trading.types import Signal, KalshiMarket, Order

def test_signal_computes_edge_and_side_from_prob_vs_price():
    s = Signal.from_quote(ticker="FED-HOLD-JUN", model_prob=0.62, market_price=0.55,
                          timestamp="2026-06-16T18:00:00Z")
    assert abs(s.edge - 0.07) < 1e-9
    assert s.side == "yes"   # model thinks YES underpriced

def test_signal_side_is_no_when_model_below_price():
    s = Signal.from_quote("X", model_prob=0.30, market_price=0.50, timestamp="t")
    assert s.side == "no"
    assert s.edge < 0   # gross edge sign carries direction
```
- [ ] **Step 2: Run — FAIL**
- [ ] **Step 3: Implement** dataclasses per INTERFACES §3 plus `Signal.from_quote` (sets `edge = model_prob - market_price`, `side = "yes" if edge > 0 else "no"`), `Order(ticker, side, count, limit_price, mode)`, `Fill`, `Position`.
- [ ] **Step 4: Run — PASS**
- [ ] **Step 5: Commit** `feat(c7): trading types + Signal.from_quote (INTERFACES §3)`

---

### Task 2: Signal gate (threshold + net-of-fees)

**Files:** Create `src/trading/gate.py`; Test `tests/test_gate.py`

Invariant (Brief §6 G3): fire only when `abs(edge) > threshold` **and** net edge after fees `> 0.06`. An edge that vanishes after fees is not an edge.

- [ ] **Step 1: Failing test**
```python
# tests/test_gate.py
from src.trading.types import Signal
from src.trading.gate import passes_gate, net_edge

def sig(prob, price): return Signal.from_quote("X", prob, price, "t")

def test_gate_rejects_below_threshold():
    assert not passes_gate(sig(0.57, 0.55), threshold=0.08, fee=0.0)  # edge 0.02 < 0.08

def test_gate_rejects_when_fees_eat_the_edge():
    # gross edge 0.09 > 0.08 threshold, but fee 0.04 leaves net 0.05 < 0.06 floor
    assert not passes_gate(sig(0.64, 0.55), threshold=0.08, fee=0.04)

def test_gate_passes_when_gross_and_net_clear():
    assert passes_gate(sig(0.65, 0.55), threshold=0.08, fee=0.02)  # edge .10, net .08

def test_net_edge_subtracts_fee_from_abs_edge():
    assert abs(net_edge(sig(0.65, 0.55), fee=0.02) - 0.08) < 1e-9
```
- [ ] **Step 2: Run — FAIL**
- [ ] **Step 3: Implement**
```python
# src/trading/gate.py
G3_NET_EDGE_FLOOR = 0.06

def net_edge(signal, fee: float) -> float:
    return abs(signal.edge) - fee

def passes_gate(signal, threshold: float, fee: float) -> bool:
    return abs(signal.edge) > threshold and net_edge(signal, fee) > G3_NET_EDGE_FLOOR
```
- [ ] **Step 4: Run — PASS**
- [ ] **Step 5: Commit** `feat(c7): signal gate (threshold + net-of-fees floor)`

---

### Task 3: Capped Kelly sizer (refuses live without bankroll)

**Files:** Create `src/trading/sizing.py`; Test `tests/test_sizing.py`

Invariants: Kelly fraction capped (no all-in), never negative; **`size_order` returns `None` (does not size) when `mode=="live"` and `bankroll is None`** (E1 not yet provided); paper mode uses a notional bankroll if none set, but is clearly marked simulated.

- [ ] **Step 1: Failing test**
```python
# tests/test_sizing.py
from src.trading.types import Signal
from src.trading.sizing import kelly_fraction, size_order

def sig(prob, price): return Signal.from_quote("X", prob, price, "t")

def test_kelly_fraction_capped_and_nonnegative():
    f = kelly_fraction(prob=0.65, price=0.55)
    assert 0 < f <= 0.25   # capped at quarter-Kelly per default cap

def test_kelly_fraction_zero_when_no_edge():
    assert kelly_fraction(prob=0.50, price=0.50) == 0.0

def test_size_order_refuses_live_without_bankroll():
    assert size_order(sig(0.65, 0.55), bankroll=None, mode="live") is None

def test_size_order_sizes_in_paper_with_notional_bankroll():
    o = size_order(sig(0.65, 0.55), bankroll=None, mode="paper")
    assert o is not None and o.mode == "paper" and o.count >= 1
```
- [ ] **Step 2: Run — FAIL**
- [ ] **Step 3: Implement**
```python
# src/trading/sizing.py
from src.trading.types import Order

KELLY_CAP = 0.25            # quarter-Kelly ceiling — size with evidence, never ahead of it
PAPER_NOTIONAL = 1000.0     # simulated bankroll for paper mode when BANKROLL unset

def kelly_fraction(prob: float, price: float) -> float:
    # Binary contract priced 0..1; payoff if YES = (1-price)/price, prob of win = prob.
    if not (0 < price < 1):
        return 0.0
    b = (1 - price) / price
    f = (b * prob - (1 - prob)) / b      # Kelly criterion
    f = max(0.0, f)                       # never negative (no shorting via this path)
    return min(f, KELLY_CAP)              # cap

def size_order(signal, bankroll, mode: str, cap: float = KELLY_CAP) -> Order | None:
    if mode == "live" and bankroll is None:
        return None                       # E1 not provided — refuse to size live. Hard stop.
    bank = bankroll if bankroll is not None else PAPER_NOTIONAL
    prob = signal.model_prob if signal.side == "yes" else 1 - signal.model_prob
    price = signal.market_price if signal.side == "yes" else 1 - signal.market_price
    frac = kelly_fraction(prob, price)
    if frac <= 0:
        return None
    stake = bank * frac
    count = max(1, int(stake / max(price, 0.01)))
    return Order(ticker=signal.ticker, side=signal.side, count=count,
                 limit_price=round(price, 2), mode=mode)
```
- [ ] **Step 4: Run — PASS**
- [ ] **Step 5: Commit** `feat(c7): capped Kelly sizer; refuses live sizing without bankroll`

---

### Task 4: Position manager (dedupe + aggregate cap)

**Files:** Create `src/trading/positions.py`; Test `tests/test_positions.py`

- [ ] **Step 1: Failing test**
```python
# tests/test_positions.py
from src.trading.types import Order
from src.trading.positions import PositionManager

def order(tkr, n): return Order(ticker=tkr, side="yes", count=n, limit_price=0.55, mode="paper")

def test_dedupe_rejects_second_order_same_ticker_side():
    pm = PositionManager(max_aggregate=10_000)
    assert pm.accept(order("A", 1))
    assert not pm.accept(order("A", 1))   # already have A/yes — no double-fire

def test_aggregate_cap_blocks_overexposure():
    pm = PositionManager(max_aggregate=100)
    assert pm.accept(order("A", 100))     # 100 * 0.55 = 55 ok
    assert not pm.accept(order("B", 1000))  # would blow the aggregate cap
```
- [ ] **Step 2: Run — FAIL**
- [ ] **Step 3: Implement** `PositionManager(max_aggregate)` tracking open (ticker,side) keys and summed notional; `accept(order)` returns False on duplicate key or if adding `count*limit_price` exceeds `max_aggregate`.
- [ ] **Step 4: Run — PASS**
- [ ] **Step 5: Commit** `feat(c7): position manager (dedupe + aggregate exposure cap)`

---

### Task 5: Kalshi client protocol + paper client + gated live client + CLI

**Files:** Create `src/trading/client.py`, `kalshi_trader.py`; Test `tests/test_paper_client.py`

- [ ] **Step 1: Failing test**
```python
# tests/test_paper_client.py
import pytest
from src.trading.types import Order
from src.trading.client import PaperKalshiClient, make_client

def test_paper_client_simulates_fill_without_network():
    c = PaperKalshiClient()
    fill = c.place(Order(ticker="X", side="yes", count=3, limit_price=0.55, mode="paper"))
    assert fill.ticker == "X" and fill.count == 3 and fill.simulated is True

def test_make_client_defaults_to_paper():
    assert isinstance(make_client(mode="paper"), PaperKalshiClient)

def test_make_client_refuses_live_without_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("SPEECHEDGE_ALLOW_LIVE", raising=False)
    with pytest.raises(RuntimeError, match="live"):
        make_client(mode="live")   # live requires explicit, loud opt-in (E4)
```
- [ ] **Step 2: Run — FAIL**
- [ ] **Step 3: Implement**
  - `KalshiClient` protocol: `place(order) -> Fill`, `quotes() -> list[KalshiMarket]`.
  - `PaperKalshiClient`: returns `Fill(..., simulated=True)`, logs to `output/paper_trades.jsonl`. No network.
  - `LiveKalshiClient`: real REST orders + WebSocket quotes (`KALSHI_API_KEY`/`SECRET` from env). Body may be a thin stub with `# TODO: wire Kalshi REST/WS` — **but it must never be reachable except through the gated `make_client`.**
  - `make_client(mode)`: returns `PaperKalshiClient` for `paper`; for `live`, **raises `RuntimeError` unless `SPEECHEDGE_ALLOW_LIVE=1`** AND keys present (E4 hard gate).
  - `kalshi_trader.py`: load config (INTERFACES §5), `make_client(mode)`, consume signals → `passes_gate` → `size_order` → `PositionManager.accept` → `client.place`; log every signal with timestamped model-vs-market divergence (Brief §5 June-16 requirement).
- [ ] **Step 4: Run — PASS**; full suite green.
- [ ] **Step 5: Commit** `feat(c7): Kalshi client protocol, paper client, gated live, trader CLI`

## Self-review (trading-code-reviewer will re-check)
- **Live/paper:** default paper; `make_client("live")` raises without `SPEECHEDGE_ALLOW_LIVE=1` + keys (Tasks 5). ✓
- **Sizing safety:** Kelly capped at 0.25, never negative, refuses live without bankroll (E1). ✓
- **Fees:** gate uses net-of-fees floor 0.06 (G3). ✓ — pass a realistic Kalshi fee into `passes_gate` from the CLI.
- **Double-fire:** PositionManager dedupes (ticker,side) + aggregate cap. ✓
- **Credentials:** keys env-only, never logged. ✓
- **Open item for reviewer:** confirm Kalshi's real fee schedule + tick size before any live use; the paper path is safe to run June 16 regardless.

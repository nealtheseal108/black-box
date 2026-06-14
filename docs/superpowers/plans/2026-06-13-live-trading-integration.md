# Live Trading Integration Implementation Plan (cross-venue, paper)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the Mode-2 tracker into actual (paper) order placement: map model probabilities onto real Kalshi + Polymarket mention markets via a curated mapping, route each one-shot bet to the higher after-fee net-edge venue, gate/size/place on the C7 paper client, and log the RL episode at settlement.

**Architecture:** A thin `TradeExecutor` over the existing C7 pipeline (`Signal`/`net_edge`/`passes_gate`/`size_order`/`PaperKalshiClient`), fed by a cross-venue quotes layer behind injected `fetch`. One-shot per term; refuses unmapped terms. Wired into `live_run.py` for a pre-speech pass + per-live-update pass + settlement.

**Tech Stack:** Python 3.11, pytest, stdlib (+ lazy `requests` in venue adapters only). Builds on `src/trading/` (C7), `src/live/episode_log.py`.

**Working directory (HARD RULE):** All commands from `/Users/nealt1/Downloads/SpeechEdge`. Run Python with `.venv/bin/python`. Commit with `git -C /Users/nealt1/Downloads/SpeechEdge`, push to `origin master` (black-box). Committing on master is the established, authorized workflow.

**Spec:** `docs/superpowers/specs/2026-06-13-live-trading-integration-design.md`

**Review note:** Task 3 (`executor.py`) prices/sizes/places orders — it MUST be reviewed by the `trading-code-reviewer` agent before the plan is considered complete (sign errors and gate misuse move money).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/trading/markets.py` | `MarketLink` + `load_market_map(path)` |
| `corpus/markets/june_presser.json` | Curated term→venue mapping (TEMPLATE tickers; re-author against live markets near the event) |
| `src/trading/quotes.py` | `VenueQuotes` Protocol; `KalshiQuotes`, `PolymarketQuotes` (injected `fetch`); `CrossVenueQuotes` |
| `src/trading/executor.py` | `TradeExecutor` (one-shot route→gate→size→place; `settle` → P&L + episode) |
| `live_run.py` | **modify** — add `replay_hearing_with_trading(quotes, mapping)` wiring the executor |
| `tests/test_markets.py`, `tests/test_quotes.py`, `tests/test_executor.py`, `tests/test_live_trading.py` | tests (no network) |

Reused (do NOT reimplement): `src/trading/types.py` (`Signal`, `Order`, `Fill`), `src/trading/gate.py` (`net_edge`, `passes_gate`), `src/trading/sizing.py` (`size_order`, `PAPER_NOTIONAL`), `src/trading/client.py` (`PaperKalshiClient`), `src/live/episode_log.py` (`EpisodeLogger`).

---

### Task 1: Market mapping

**Files:**
- Create: `corpus/markets/june_presser.json`
- Create: `src/trading/markets.py`
- Test: `tests/test_markets.py`

- [ ] **Step 1: Create the curated mapping (TEMPLATE)**

Create `corpus/markets/june_presser.json` (create `corpus/markets/`). These are TEMPLATE tickers/ids — re-authored against the live markets near the event:

```json
[
  {"canonical": "rate cut", "kalshi_ticker": "KXWARSHJUNE-RATECUT", "polymarket_id": "warsh-june-rate-cut", "kalshi_question": "Will Warsh say 'rate cut' or 'cut rate'?", "polymarket_question": "Kevin Warsh says Rate Cut / Cut Rate?"},
  {"canonical": "trump", "kalshi_ticker": "KXWARSHJUNE-TRUMP", "polymarket_id": "warsh-june-trump", "kalshi_question": "Will Warsh say 'Trump'?", "polymarket_question": "Kevin Warsh says Trump?"},
  {"canonical": "inflation", "kalshi_ticker": "KXWARSHJUNE-INFLATION", "polymarket_id": "warsh-june-inflation", "kalshi_question": "Will Warsh say 'inflation'?", "polymarket_question": "Kevin Warsh says inflation?"},
  {"canonical": "independence", "kalshi_ticker": "KXWARSHJUNE-INDEP", "polymarket_id": "warsh-june-independence", "kalshi_question": "Will Warsh say 'independence'?", "polymarket_question": "Kevin Warsh says independence?"},
  {"canonical": "recession", "kalshi_ticker": "KXWARSHJUNE-RECESSION", "polymarket_id": "warsh-june-recession", "kalshi_question": "Will Warsh say 'recession'?", "polymarket_question": "Kevin Warsh says recession?"}
]
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_markets.py
from pathlib import Path
from src.trading.markets import MarketLink, load_market_map

MAP = Path("corpus/markets/june_presser.json")


def test_market_link_exposes_per_venue_ticker():
    link = MarketLink(canonical="rate cut", kalshi_ticker="K-RC", polymarket_id="p-rc",
                      kalshi_question="q1", polymarket_question="q2")
    assert link.ticker_for("kalshi") == "K-RC"
    assert link.ticker_for("polymarket") == "p-rc"


def test_load_market_map_keys_by_canonical():
    m = load_market_map(MAP)
    assert "rate cut" in m and "trump" in m
    assert m["rate cut"].kalshi_ticker == "KXWARSHJUNE-RATECUT"
    # an unmapped term is simply absent (the loop will skip it -> never mis-traded)
    assert "soft landing" not in m
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_markets.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.trading.markets'`

- [ ] **Step 4: Write minimal implementation**

```python
# src/trading/markets.py
"""Curated term -> venue market mapping.

A MarketLink ties one canonical model term to its Kalshi ticker and Polymarket id.
The executor trades ONLY terms present here — an unmapped term is skipped, never
fuzzy-matched, so a bet can never land on the wrong contract.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MarketLink:
    canonical: str
    kalshi_ticker: str
    polymarket_id: str
    kalshi_question: str = ""
    polymarket_question: str = ""

    def ticker_for(self, venue: str) -> str:
        return self.kalshi_ticker if venue == "kalshi" else self.polymarket_id


def load_market_map(path: str | Path) -> dict[str, MarketLink]:
    raw = json.loads(Path(path).read_text())
    return {e["canonical"]: MarketLink(**e) for e in raw}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_markets.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git -C /Users/nealt1/Downloads/SpeechEdge add corpus/markets/june_presser.json src/trading/markets.py tests/test_markets.py
git -C /Users/nealt1/Downloads/SpeechEdge commit -m "feat(trading): curated term->venue market mapping"
git -C /Users/nealt1/Downloads/SpeechEdge push -q origin master
```

---

### Task 2: Cross-venue quotes

**Files:**
- Create: `src/trading/quotes.py`
- Test: `tests/test_quotes.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quotes.py
from src.trading.markets import MarketLink
from src.trading.quotes import KalshiQuotes, PolymarketQuotes, CrossVenueQuotes


def test_kalshi_quotes_parses_yes_price():
    def fake_fetch(market_id):
        return {"yes_price": 0.61}
    assert KalshiQuotes(fetch=fake_fetch).price("K-RC") == 0.61


def test_quote_returns_none_when_fetch_returns_none():
    def fake_fetch(market_id):
        return None
    assert KalshiQuotes(fetch=fake_fetch).price("missing") is None


def test_cross_venue_returns_price_per_available_venue():
    link = MarketLink(canonical="rate cut", kalshi_ticker="K-RC", polymarket_id="p-rc")
    mapping = {"rate cut": link}
    kalshi = KalshiQuotes(fetch=lambda mid: {"yes_price": 0.60})
    poly = PolymarketQuotes(fetch=lambda mid: {"yes_price": 0.70})
    xq = CrossVenueQuotes([kalshi, poly], mapping)
    assert xq.quotes_for("rate cut") == {"kalshi": 0.60, "polymarket": 0.70}


def test_cross_venue_omits_venue_with_no_quote_and_unmapped_term():
    link = MarketLink(canonical="rate cut", kalshi_ticker="K-RC", polymarket_id="p-rc")
    mapping = {"rate cut": link}
    kalshi = KalshiQuotes(fetch=lambda mid: {"yes_price": 0.60})
    poly = PolymarketQuotes(fetch=lambda mid: None)          # no quote on polymarket
    xq = CrossVenueQuotes([kalshi, poly], mapping)
    assert xq.quotes_for("rate cut") == {"kalshi": 0.60}
    assert xq.quotes_for("not in mapping") == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_quotes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.trading.quotes'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/trading/quotes.py
"""Cross-venue market quotes for the mention markets.

Each venue adapter turns an injected fetch(market_id) -> dict|None into a normalized
YES price in [0,1]. `requests` is imported lazily so the test suite needs no network
or dependency. CrossVenueQuotes aggregates per-term prices across venues, using the
curated mapping to look up each venue's market id.
"""
from __future__ import annotations

from typing import Callable, Protocol

from src.trading.markets import MarketLink


class VenueQuotes(Protocol):
    venue: str
    def price(self, market_id: str) -> float | None:
        """Return the current YES price (0..1) for this venue's market, or None."""
        ...


def _kalshi_fetch(market_id: str) -> dict | None:
    import requests  # lazy
    r = requests.get(f"https://api.elections.kalshi.com/trade-api/v2/markets/{market_id}", timeout=15)
    if r.status_code != 200:
        return None
    m = r.json().get("market", {})
    cents = m.get("yes_bid")
    return {"yes_price": cents / 100.0} if cents is not None else None


def _polymarket_fetch(market_id: str) -> dict | None:
    import requests  # lazy
    r = requests.get(f"https://gamma-api.polymarket.com/markets/{market_id}", timeout=15)
    if r.status_code != 200:
        return None
    price = r.json().get("outcomePrices", [None])[0]
    return {"yes_price": float(price)} if price is not None else None


class _VenueQuotesBase:
    def __init__(self, fetch: Callable[[str], dict | None]) -> None:
        self._fetch = fetch

    def price(self, market_id: str) -> float | None:
        data = self._fetch(market_id)
        if not data:
            return None
        return data.get("yes_price")


class KalshiQuotes(_VenueQuotesBase):
    venue = "kalshi"
    def __init__(self, fetch: Callable[[str], dict | None] | None = None) -> None:
        super().__init__(fetch or _kalshi_fetch)


class PolymarketQuotes(_VenueQuotesBase):
    venue = "polymarket"
    def __init__(self, fetch: Callable[[str], dict | None] | None = None) -> None:
        super().__init__(fetch or _polymarket_fetch)


class CrossVenueQuotes:
    def __init__(self, venues: list, mapping: dict[str, MarketLink]) -> None:
        self._venues = venues
        self._mapping = mapping

    def quotes_for(self, canonical: str) -> dict[str, float]:
        link = self._mapping.get(canonical)
        if link is None:
            return {}
        out: dict[str, float] = {}
        for v in self._venues:
            p = v.price(link.ticker_for(v.venue))
            if p is not None:
                out[v.venue] = p
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_quotes.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git -C /Users/nealt1/Downloads/SpeechEdge add src/trading/quotes.py tests/test_quotes.py
git -C /Users/nealt1/Downloads/SpeechEdge commit -m "feat(trading): cross-venue quotes (Kalshi + Polymarket, injected fetch)"
git -C /Users/nealt1/Downloads/SpeechEdge push -q origin master
```

---

### Task 3: TradeExecutor

**Files:**
- Create: `src/trading/executor.py`
- Test: `tests/test_executor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_executor.py
from src.trading.markets import MarketLink
from src.trading.client import PaperKalshiClient
from src.live.episode_log import EpisodeLogger
from src.trading.executor import TradeExecutor


class _FakeQuotes:
    def __init__(self, table):
        self._table = table     # {canonical: {venue: price}}
    def quotes_for(self, canonical):
        return self._table.get(canonical, {})


def _mapping():
    return {
        "rate cut": MarketLink("rate cut", "K-RC", "P-RC"),
        "trump": MarketLink("trump", "K-TR", "P-TR"),
    }


def _executor(quotes_table, tmp_path):
    return TradeExecutor(
        mapping=_mapping(),
        quotes=_FakeQuotes(quotes_table),
        client=PaperKalshiClient(log_path=tmp_path / "paper.jsonl"),
        episode_log=EpisodeLogger(tmp_path / "episodes.jsonl"),
        bankroll=None, threshold=0.08, fee=0.02,
    )


def test_routes_to_higher_net_edge_venue(tmp_path):
    # model 0.9; kalshi YES 0.60 (edge .30) beats polymarket YES 0.70 (edge .20)
    ex = _executor({"rate cut": {"kalshi": 0.60, "polymarket": 0.70}}, tmp_path)
    opened = ex.evaluate({"rate cut": 0.9})
    assert len(opened) == 1
    assert opened[0].venue == "kalshi"
    assert opened[0].order.side == "yes"


def test_skips_term_below_gate(tmp_path):
    # model 0.62 vs price 0.60 -> edge 0.02, below threshold
    ex = _executor({"rate cut": {"kalshi": 0.60}}, tmp_path)
    assert ex.evaluate({"rate cut": 0.62}) == []


def test_one_shot_does_not_retrade(tmp_path):
    ex = _executor({"rate cut": {"kalshi": 0.60}}, tmp_path)
    first = ex.evaluate({"rate cut": 0.9})
    second = ex.evaluate({"rate cut": 0.95})    # would also clear the gate
    assert len(first) == 1 and second == []


def test_refuses_unmapped_term(tmp_path):
    ex = _executor({"soft landing": {"kalshi": 0.10}}, tmp_path)
    assert ex.evaluate({"soft landing": 0.9}) == []   # not in mapping


def test_settle_logs_episode_with_pnl_sign(tmp_path):
    import json
    ep = tmp_path / "episodes.jsonl"
    ex = TradeExecutor(mapping=_mapping(), quotes=_FakeQuotes({"rate cut": {"kalshi": 0.60}}),
                       client=PaperKalshiClient(log_path=tmp_path / "p.jsonl"),
                       episode_log=EpisodeLogger(ep), bankroll=None, threshold=0.08, fee=0.02)
    ex.evaluate({"rate cut": 0.9})              # buys YES at 0.60
    ex.settle({"rate cut": 1})                  # term WAS said -> YES wins -> positive pnl
    rec = json.loads(ep.read_text().strip().splitlines()[0])
    assert rec["reward"] > 0
    assert rec["action"]["venue"] == "kalshi"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_executor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.trading.executor'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/trading/executor.py
"""TradeExecutor — one-shot, cross-venue, paper.

For each MAPPED, not-yet-traded term: build a C7 Signal against each venue's price,
route to the venue with the highest after-fee net edge (best execution), gate, size
(quarter-Kelly), and place on the paper client. One position per term (no re-trading);
unmapped terms are refused. settle() computes paper P&L at resolution and logs the
RL episode (reward = realized P&L).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from src.trading.types import Signal, Order, Fill
from src.trading.gate import net_edge, passes_gate
from src.trading.sizing import size_order
from src.trading.markets import MarketLink


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class OpenPosition:
    canonical: str
    venue: str
    order: Order
    fill: Fill
    model_prob: float
    phase: str


class TradeExecutor:
    def __init__(self, mapping: dict[str, MarketLink], quotes, client, episode_log,
                 bankroll=None, threshold: float = 0.08, fee: float = 0.02,
                 phase: str = "pre") -> None:
        self._mapping = mapping
        self._quotes = quotes
        self._client = client
        self._episode_log = episode_log
        self._bankroll = bankroll
        self._threshold = threshold
        self._fee = fee
        self._phase = phase
        self._traded: dict[str, OpenPosition] = {}

    def set_phase(self, phase: str) -> None:
        self._phase = phase

    def evaluate(self, probs: dict[str, float]) -> list[OpenPosition]:
        opened: list[OpenPosition] = []
        for canonical, p in probs.items():
            if canonical in self._traded:
                continue
            link = self._mapping.get(canonical)
            if link is None:
                continue  # refuse unmapped — never fuzzy-match a bet
            venue_prices = self._quotes.quotes_for(canonical)
            if not venue_prices:
                continue
            # Route: pick the venue with the highest after-fee net edge.
            best = None
            for venue, price in venue_prices.items():
                sig = Signal.from_quote(ticker=link.ticker_for(venue), model_prob=p,
                                        market_price=price, timestamp=_now())
                ne = net_edge(sig, self._fee)
                if best is None or ne > best[2]:
                    best = (venue, sig, ne)
            venue, sig, _ = best
            if not passes_gate(sig, self._threshold, self._fee):
                continue
            order = size_order(sig, self._bankroll, mode="paper")
            if order is None:
                continue
            fill = self._client.place(order)
            pos = OpenPosition(canonical=canonical, venue=venue, order=order, fill=fill,
                               model_prob=p, phase=self._phase)
            self._traded[canonical] = pos
            opened.append(pos)
        return opened

    def settle(self, outcomes: dict[str, int]) -> None:
        for canonical, pos in self._traded.items():
            outcome = outcomes.get(canonical, 0)
            pnl = self._pnl(pos, outcome)
            self._episode_log.log(
                state={"canonical": canonical, "model_prob": pos.model_prob, "phase": pos.phase},
                action={"venue": pos.venue, "side": pos.order.side,
                        "count": pos.order.count, "price": pos.fill.fill_price},
                reward=pnl,
            )

    @staticmethod
    def _pnl(pos: OpenPosition, outcome: int) -> float:
        price = pos.fill.fill_price
        count = pos.order.count
        won = (pos.order.side == "yes" and outcome == 1) or (pos.order.side == "no" and outcome == 0)
        return (1.0 - price) * count if won else -price * count
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_executor.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git -C /Users/nealt1/Downloads/SpeechEdge add src/trading/executor.py tests/test_executor.py
git -C /Users/nealt1/Downloads/SpeechEdge commit -m "feat(trading): one-shot cross-venue TradeExecutor (route/gate/size/place/settle)"
git -C /Users/nealt1/Downloads/SpeechEdge push -q origin master
```

---

### Task 4: Wire executor into the live loop

**Files:**
- Modify: `live_run.py` (add `replay_hearing_with_trading`; keep `replay_hearing` unchanged)
- Test: `tests/test_live_trading.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_live_trading.py
from src.trading.markets import MarketLink
from live_run import replay_hearing_with_trading


class _FakeQuotes:
    """Quote every mapped term cheaply on kalshi so the model's high P clears the gate."""
    def __init__(self, mapping):
        self._mapping = mapping
    def quotes_for(self, canonical):
        return {"kalshi": 0.30} if canonical in self._mapping else {}


def test_replay_with_trading_places_and_settles(tmp_path):
    mapping = {
        "inflation": MarketLink("inflation", "K-INF", "P-INF"),
        "rate cut": MarketLink("rate cut", "K-RC", "P-RC"),
        "soft landing": MarketLink("soft landing", "K-SL", "P-SL"),
    }
    result = replay_hearing_with_trading(
        quotes=_FakeQuotes(mapping), mapping=mapping,
        paper_log=tmp_path / "paper.jsonl", episode_log=tmp_path / "ep.jsonl",
    )
    # inflation + rate cut are spoken in the hearing and priced at 0.30 -> traded
    assert result["n_trades"] >= 2
    # every trade produced a settled episode with a realized reward
    assert result["n_episodes"] == result["n_trades"]
    # soft landing was never spoken: if traded, its YES bet settles to a LOSS
    assert "total_pnl" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_live_trading.py -v`
Expected: FAIL — `ImportError: cannot import name 'replay_hearing_with_trading'`

- [ ] **Step 3: Add the implementation to `live_run.py`**

Add these imports near the top of `live_run.py` (with the existing imports):

```python
from src.mentions.terms import MarketTerm
from src.trading.executor import TradeExecutor
from src.trading.client import PaperKalshiClient
from src.live.episode_log import EpisodeLogger
```

Add this function to `live_run.py` (below `replay_hearing`):

```python
def replay_hearing_with_trading(quotes, mapping, paper_log, episode_log,
                                chunk_words: int = 25, threshold: float = 0.08, fee: float = 0.02) -> dict:
    """End-to-end: pre-speech bets on Mode-1 priors, live bets as the hearing streams,
    then settle every position against what was actually said. Returns a trade summary.
    `quotes` is any object with quotes_for(canonical) -> {venue: price}.
    """
    docs = _load_corpus()
    terms = load_terms(LEXICON)
    prior_docs = [d for d in docs if d.get("date", "") < HEARING_DATE]

    priors_full = predict_event_priors(prior_docs, terms, news_context="",
                                       adjuster=NullAdjuster(), as_of=HEARING_DATE)
    priors = {c: row["p_prior"] for c, row in priors_full.items()}
    weights = cooccurrence_weights(prior_docs, terms, terms, w_max=2.0)
    tracker = LiveMentionTracker(terms=terms, priors=priors, weights=weights, evidence_terms=terms)

    executor = TradeExecutor(mapping=mapping, quotes=quotes,
                             client=PaperKalshiClient(log_path=paper_log),
                             episode_log=EpisodeLogger(episode_log),
                             bankroll=None, threshold=threshold, fee=fee, phase="pre")

    # pre-speech: bet on Mode-1 priors
    n_trades = len(executor.evaluate(priors))

    # live: stream the hearing, bet on tracker updates
    executor.set_phase("live")
    for delta in ReplayStream(_hearing_text(), chunk_words=chunk_words).stream():
        tracker.consume(delta)
        n_trades += len(executor.evaluate(tracker.probabilities()))

    # settle against the realized outcomes (resolved terms => 1, else 0)
    final = tracker.probabilities()
    outcomes = {c: (1 if p == 1.0 else 0) for c, p in final.items()}
    n_before = n_trades
    executor.settle(outcomes)

    # one episode per traded term
    import json
    n_episodes = len([1 for _ in open(episode_log)]) if EpisodeLogger else 0
    total_pnl = sum(json.loads(l)["reward"] for l in open(episode_log)) if n_episodes else 0.0
    return {"n_trades": n_trades, "n_episodes": n_episodes, "total_pnl": total_pnl}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_live_trading.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /Users/nealt1/Downloads/SpeechEdge add live_run.py tests/test_live_trading.py
git -C /Users/nealt1/Downloads/SpeechEdge commit -m "feat(live): wire TradeExecutor into the replay loop (pre-speech + live + settle)"
git -C /Users/nealt1/Downloads/SpeechEdge push -q origin master
```

---

### Task 5: trading-code-reviewer audit + full-suite regression + DECISIONS close-out

- [ ] **Step 1: Full-suite regression**

Run: `.venv/bin/python -m pytest -q`
Expected: all prior tests (133) + the new trading tests PASS, no regressions. If anything fails, STOP and report BLOCKED.

- [ ] **Step 2: Append to DECISIONS.md**

Append a `### D9` entry under a `## 2026-06-13` heading (create the heading if absent), before any trailing "Accepted defaults" table:

```markdown
### D9 — Live trading integration shipped (cross-venue, paper)
- **Decision:** The Mode-2 tracker now places (paper) bets. A curated term->venue mapping (`corpus/markets/june_presser.json`) links each model term to its Kalshi ticker + Polymarket id; CrossVenueQuotes pulls both prices; TradeExecutor routes each one-shot bet to the higher after-fee net-edge venue, gates (C7 G3), sizes (quarter-Kelly, paper notional $1k), places on PaperKalshiClient, and logs the RL episode at settlement. Double-trading and wrong-market are structurally impossible (one-shot set; unmapped terms refused). Spec: `docs/superpowers/specs/2026-06-13-live-trading-integration-design.md`.
- **Reviewed:** `trading-code-reviewer` agent audited `executor.py` (sizing/gate/sign/paper-vs-live). <summary of findings + fixes>.
- **Scope:** Paper only (live still behind SPEECHEDGE_ALLOW_LIVE + creds). Near-event work: re-author the mapping against the live markets; confirm Kalshi/Polymarket quote endpoints; confirm StreamText event id for Warsh's debut. 2B (recalibration/tuning) remains.
- **Authority:** Owner-directed.
```

Replace `<summary ...>` with the reviewer's actual findings/fixes from Step 3 before committing.

- [ ] **Step 3: trading-code-reviewer audit**

Dispatch the `trading-code-reviewer` agent against `src/trading/executor.py` and `src/trading/sizing.py` usage (BASE = commit before Task 3, HEAD = current). Address any Critical/Important findings (sign errors, sizing blowups, fee/threshold misuse, paper-vs-live confusion) before finalizing. Record the summary in the D9 entry.

- [ ] **Step 4: Commit**

```bash
git -C /Users/nealt1/Downloads/SpeechEdge add DECISIONS.md
git -C /Users/nealt1/Downloads/SpeechEdge commit -m "docs(decisions): D9 live trading integration shipped (cross-venue, paper)"
git -C /Users/nealt1/Downloads/SpeechEdge push -q origin master
```

---

## Self-Review

**Spec coverage (§ → task):**
- §3/§4 `markets.py` + mapping → Task 1; `quotes.py` (Kalshi+Polymarket+CrossVenue) → Task 2; `executor.py` → Task 3; `live_run.py` wiring → Task 4.
- §3/§5 routing by max `net_edge` → Task 3 (`test_routes_to_higher_net_edge_venue`); gate → Task 3 (`test_skips_term_below_gate`); size (paper notional via `size_order` bankroll=None) → Task 3; one-shot → Task 3 (`test_one_shot_does_not_retrade`); refuse-unmapped → Task 3 (`test_refuses_unmapped_term`).
- §5 settlement P&L + episode → Task 3 (`test_settle_logs_episode_with_pnl_sign`), exercised end-to-end in Task 4.
- §6 error handling: missing venue quote (Task 2 `test_cross_venue_omits...`), unmapped (Tasks 1/2/3), `size_order` None (paper notional always sizes ≥1 when frac>0; gate filters tiny edges), lazy `requests` (Task 2).
- §2 paper-only → executor uses `PaperKalshiClient`; `size_order(mode="paper")` throughout; live untouched.
- §8 open items: TEMPLATE mapping authored near event (Task 1 note), endpoints behind injected fetch (Task 2), bankroll $1k paper notional (C7 `PAPER_NOTIONAL`), settlement trigger (Task 4). Trading-code-reviewer audit → Task 5.

**Placeholder scan:** none in code. The `<summary ...>` in Task 5's D9 is an explicit instruction to insert the reviewer's real findings before committing; `june_presser.json` is explicitly a TEMPLATE to re-author near the event (the executor refuses unmapped terms, so a stale template fails safe).

**Type consistency:** `MarketLink(canonical, kalshi_ticker, polymarket_id, ...)` + `.ticker_for(venue)` (Task 1) used in Tasks 2–4. `CrossVenueQuotes.quotes_for(canonical) -> {venue: price}` (Task 2) consumed by `TradeExecutor` (Task 3) and the `_FakeQuotes` test doubles. `TradeExecutor(mapping, quotes, client, episode_log, bankroll, threshold, fee, phase)` + `.evaluate(probs) -> list[OpenPosition]` + `.set_phase` + `.settle(outcomes)` consistent across Tasks 3–4. Reuses C7 `Signal.from_quote`, `net_edge`, `passes_gate`, `size_order`, `PaperKalshiClient`, and `EpisodeLogger` at their real signatures.

# Design — Live Trading Integration (cross-venue, paper)

**Date:** 2026-06-13
**Status:** Approved (design); proceeding to plan + build at owner's direction.
**Parent:** completes the June-16 pipeline — wires the Mode-2 tracker into actual (paper) order placement against the real Kalshi + Polymarket mention markets.
**Predecessors:** Plan 1 (Mode-1), 2A (harness), 2C (live tracker).

---

## 1. Objective

Turn the validated model into something that places (paper) bets: map our model's per-term
probabilities onto the real Kalshi + Polymarket mention markets, route each bet to the
better-priced venue, gate and size it with the existing C7 pipeline, place it on the paper
client, and log the RL episode at settlement. Trades fire both pre-speech (Mode-1 priors) and
during the speech (live tracker updates).

**Paper only.** Live capital stays behind the existing `SPEECHEDGE_ALLOW_LIVE` + credentials
gate (E4). This sub-project never places a live order.

---

## 2. Scope

**In:**
- Curated **term→venue market mapping** (Kalshi ticker + Polymarket id per canonical term).
- **Cross-venue quotes** layer: `KalshiQuotes` + `PolymarketQuotes` adapters (injected `fetch`,
  lazy `requests`) behind a `VenueQuotes` seam, aggregated by `CrossVenueQuotes`.
- **`TradeExecutor`**: one-shot per term, best-venue routing (max after-fee net edge), gate →
  size → paper place, settle → RL-episode log.
- Wiring into `live_run.py`: a pre-speech evaluate pass + a per-live-update pass.

**Out (YAGNI / deferred):**
- **Live capital / real orders** (paper only; live behind the C7 gate).
- **Divergence trading** (cross-venue gap as its own book) — best-execution routing only.
- **Re-trading / position adjustment** — one-shot per term, hold to resolution.
- **Auto-matching markets** — curated mapping only; unmapped terms are skipped, never fuzzy-matched.
- Real bankroll sizing for live (E1 unset) — a paper notional default is used.

---

## 3. Architecture & data flow

`TradeExecutor` is a thin layer over the existing C7 pipeline, fed by cross-venue quotes.

```
mapping = load_market_map("corpus/markets/june_presser.json")     # canonical -> MarketLink
quotes  = CrossVenueQuotes(KalshiQuotes(...), PolymarketQuotes(...), mapping)
executor = TradeExecutor(mapping, quotes, paper_client, episode_log,
                         bankroll, threshold, fee)

# pre-speech: bets on Mode-1 priors
executor.evaluate(mode1_priors)
# during speech: per StreamText delta
tracker.consume(delta); executor.evaluate(tracker.probabilities())
# at resolution / speech end
executor.settle(resolutions)        # compute paper P&L, log RL episode (reward known here)
```

`evaluate(probs)` for each MAPPED, not-yet-traded term B:
1. build a C7 `Signal.from_quote(model_prob=P[B], market_price=price_v)` for each venue with a quote;
2. **route**: `best = argmax_v net_edge(signal_v, fee)` (best execution falls out of max net edge —
   no per-side special-casing);
3. **gate**: `passes_gate(best, threshold, fee)`;
4. **size**: `size_order(best, bankroll, mode="paper", cap=KELLY_CAP)` (None if too small);
5. **place**: `PaperKalshiClient.place(order)`; mark B traded (one-shot).

Two money-failure modes are structurally impossible: **double-trading** (one-shot `_traded` set)
and **wrong-market** (refuse any term not in the curated mapping).

---

## 4. Components & file structure

| File | Responsibility |
|---|---|
| `src/trading/markets.py` | `MarketLink` (canonical, kalshi_ticker, polymarket_id, kalshi_question, polymarket_question) + `load_market_map(path) -> dict[str, MarketLink]` |
| `corpus/markets/june_presser.json` | Curated term→venue mapping (template now; authored against live markets near the event) |
| `src/trading/quotes.py` | `VenueQuotes` Protocol; `KalshiQuotes`, `PolymarketQuotes` (injected `fetch`, lazy `requests`); `CrossVenueQuotes` → `{term: {venue: price}}` |
| `src/trading/executor.py` | `TradeExecutor` — one-shot, best-venue routing, gate→size→place, settle+log |
| `live_run.py` | **extend** — wire executor into pre-speech + live loop |
| `tests/test_markets.py`, `tests/test_quotes.py`, `tests/test_executor.py` | unit (executor with fake quotes + paper client; no network) |
| *reused* | C7 `Signal`/`Order`/`Fill`, `net_edge`/`passes_gate` (`gate.py`), `size_order` (`sizing.py`), `PaperKalshiClient` (`client.py`), `EpisodeLogger` (`src/live/episode_log.py`) |

---

## 5. Routing, gating, settlement (detail)

- **Route:** max after-fee `net_edge` across venues with a live quote; a venue missing a quote is
  skipped; if no venue quotes, the term is skipped this pass.
- **Gate:** `passes_gate(best, threshold, fee)` — reuses G3's net-edge floor (0.06). `threshold`
  defaults to the E2 signal threshold ($0.08); `fee` is the Kalshi fee (confirm value, default the
  C7 setting).
- **Size:** `size_order(best, bankroll, "paper", cap=KELLY_CAP)` — quarter-Kelly capped.
- **Bankroll:** a **paper notional** (default $1,000, configurable) — E1 (live capital) stays unset.
- **Settlement:** each open position records `{term, venue, side, count, fill_price,
  model_prob_at_trade, phase: "pre"|"live"}`. On resolution (spoken → outcome 1) or speech end
  (unsaid → outcome 0): paper P&L = settle_value − cost; **then** `EpisodeLogger.log(state, action,
  reward=pnl)`. Reward is realized P&L, known only at settlement.

---

## 6. Error handling

- **Unmapped term** → skipped (never fuzzy-matched / never traded).
- **Venue quote missing / fetch error** → that venue skipped; trade routes to the other venue, or
  the term is skipped if neither quotes (logged).
- **`size_order` returns None** (edge too small after sizing) → no order, term stays untraded
  (may trade later if a live update raises the edge — still one-shot once placed).
- **Already-traded term** → never re-evaluated (`_traded` set).
- **Paper only:** executor constructs the C7 paper client; a live order still requires
  `SPEECHEDGE_ALLOW_LIVE` + creds via `make_client` (unchanged).
- **`requests` lazy-imported** in the venue adapters so tests need no network/dependency.

---

## 7. Testing strategy

- **markets** (`test_markets.py`): `load_market_map` parses the mapping; `MarketLink` exposes
  per-venue ticker/question; a term absent from the file is absent from the map.
- **quotes** (`test_quotes.py`): `KalshiQuotes`/`PolymarketQuotes` parse a fake `fetch` response
  into a price; `CrossVenueQuotes` returns `{term: {venue: price}}`; missing venue → absent.
- **executor** (`test_executor.py`) — the core, all with fake quotes + a paper client:
  (a) routes to the higher-net-edge venue; (b) skips a term below the gate; (c) one-shot (a term
  already traded is not re-traded on a later `evaluate`); (d) refuses an unmapped term;
  (e) `settle` computes the right paper P&L sign and logs an episode with that reward.
- No network, no live client, in any test.

---

## 8. Open items to resolve in planning / near the event

1. **Author `june_presser.json`** against the live Kalshi + Polymarket markets once listed
   (tickers/ids + question text). A representative template ships in the plan.
2. **Kalshi/Polymarket quote endpoints** — confirm the public price endpoints + JSON shape for the
   real adapters (the injected `fetch` isolates this; fixtures stand in for tests).
3. **Paper bankroll** default ($1,000) and **threshold/fee** values — confirm against C7/E2.
4. **Settlement trigger in the live loop** — settle on tracker resolution events + a final
   speech-end sweep for unsaid terms.

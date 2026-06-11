# SPEECHEDGE — Master Execution Roadmap

> PM-level sequencing across all 7 components. Detailed task-by-task plans live in `docs/superpowers/plans/`, one per subsystem (scope: each subsystem is independently testable).

**Hard deadline:** June 16–17 2026 FOMC (Warsh debut presser). **Today:** June 9. **Window:** ~7 days.
**June 16 mode:** Paper trading (E4 default). No live capital until G1–G4 pass.

---

## Revised reality (vs. Brief §5)

The brief's schedule assumed C1/C2/C8 were BUILT. They are not (see DECISIONS.md D2). Everything is build-from-scratch. This inverts the critical path: **C1 (corpus scraper) is now the longest pole**, and it is the one component I cannot execute (sandbox blocks all target domains). Therefore C1 is written first as an owner hand-off.

## Two work streams

```
CORPUS SPINE (gated on owner-run scraping + manual ingestion)
  C1 scrape_warsh.py [Fable writes → Neal runs] ──→ corpus/ ──→ C2 build_model.py [Fable] ──→ C5 backtest.py [Fable]

PARALLEL BRANCH (no corpus dependency — start immediately, Fable fully owns + tests)
  C4 context_agent.py ┐
  C7 kalshi_trader.py ┼──→ C6 live_predictor.py ──→ June 16 dress rehearsal
  C8 dashboard       ┘
```

C5 and C6 are the join points: C5 needs C2 (model) + Powell transcripts; C6 needs C2 (model) + C4 (priors) + C7 (order path) + an audio source (E3).

## Sequenced plan (compressed)

| Day | Fable builds | Neal (owner-executed / escalations) |
|-----|--------------|--------------------------------------|
| **9 (today)** | C1 scraper (hand-off) → start C4 context agent | **Run C1**; chase E5 (Kalshi acct + API key); confirm E3 Deepgram cost |
| 10 | C4 context agent; C2 model (test on sample corpus) | Manual-ingest April-21 hearing + paywalled WSJ → `corpus/manual/` |
| 11 | C7 Kalshi paper integration; run C2 on real corpus | Provide E1 capital allocation figure (for Kelly sizer, paper mode) |
| 12 | C5 backtest (Powell pressers + held-out Warsh) → **G1/G3 gate review** | Sanity-check phrase fingerprints + lift scores |
| 13 | C6 live predictor; end-to-end rehearsal on archived FOMC audio | Provide archived FOMC audio file(s) |
| 14 | C8 dashboard wired to live feeds; full dress rehearsal | Dress-rehearsal observer; sign-off check |
| 15–16 | Monitoring + signal logging harness | **FOMC: paper trade.** Log every model-vs-market divergence |
| 18+ | Post-mortem; fine-tune on Warsh's actual words; **G2/G4 gate review** | Gate review before any live capital; then start C3 Trump pipeline |

## Validation gate ownership (Brief §6)

| Gate | Metric | Threshold | Produced by |
|------|--------|-----------|-------------|
| G1 Diction | top-1 phrase accuracy, held-out corpus | > 20% | C2 + C5 |
| G2 Calibration | Brier score on probability forecasts | < 0.22 | C5 + June-16 log |
| G3 Market | historical signals w/ positive edge | > 55%, avg edge > $0.06 after fees | C5 (on Powell) |
| G4 Live | June 16 paper P&L + signal-log review | positive expectancy, no failures | June-16 run |

**No live capital until G1–G4 all pass.**

## Open escalations (batched — see DECISIONS.md / RISKS.md)
- **E5 (R2):** Kalshi funded account + trading-enabled API key — longest external lead time. Owner, day 1.
- **E1:** Capital allocation figure — blocks Kelly sizer (even paper mode wants a notional bankroll).
- **E3:** Confirm Deepgram cost (default accepted, ~300ms latency edge over local Whisper).
- **R4:** Corpus quality — April-21 hearing is the single highest-value document; manual ingestion is owner-only.

## Per-subsystem plan index
- [x] `docs/superpowers/plans/2026-06-09-c1-warsh-corpus-scraper.md` — **DONE, merged to master (15 tests)**
- [x] `docs/superpowers/plans/2026-06-10-c2-diction-model.md` — **DONE, merged + pushed (13 tests)**
- [x] `docs/superpowers/plans/2026-06-10-c4-context-agent.md` — **DONE, merged + pushed (7 tests)**
- [x] `docs/superpowers/plans/2026-06-10-c7-kalshi-trader.md` — **DONE, trading-reviewed, merged + pushed (17 tests)**
- [x] `docs/superpowers/plans/2026-06-10-c5-backtest.md` — **DONE, merged + pushed (10 tests)**
- [x] `docs/superpowers/plans/2026-06-10-c6-live-predictor.md` — **DONE, merged + pushed (10 tests)**
- [x] `docs/superpowers/plans/2026-06-10-c8-dashboard.md` — **DONE, merged + pushed (7 tests); real server boot-verified**

**Build status:** 🎉 **All 8 components (C1–C8) complete. 79 tests green. All pushed to black-box.** Dashboard boots and serves `/api/state` + `/` live.

## Owner-gated for a real June-16 run (code is done; these are data/credentials):
1. **C1 corpus** — run `python scrape_warsh.py` + manually ingest the April-21 hearing + WSJ → `corpus/manual/`. Unblocks G1.
2. **Powell data** — transcripts + historical Kalshi prices → wire `backtest.py` Level-2 loader. Unblocks G2/G3.
3. **E3 audio** — confirm Deepgram cost; wire `live_predictor.py` transcriber.
4. **E5 Kalshi account** + **E1 bankroll** — for paper-with-real-quotes / eventual live.
5. **`KALSHI_FEE`** — set to real fee before the run (trader warns if 0).
6. **Gate review (G1–G4)** before ANY live capital (Brief §6).

## Deferred (post-June-16, per Brief §5):
- C3 Trump pipeline — second speaker, after Warsh post-mortem + fine-tune.

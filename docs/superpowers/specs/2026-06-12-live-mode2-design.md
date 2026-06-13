# Design — Live Mode-2 Real-Time Mention Tracking (Sub-project 2C)

**Date:** 2026-06-12
**Status:** Approved (design); pending spec review → implementation plan
**Parent:** Plan 2 → 2A (calibration harness, done), 2B (recalibration/tuning, deferred), 2C (this).
**Deadline driver:** June 16–17 2026 FOMC — Warsh's debut press conference.

---

## 1. Objective

Trade prediction-market Mentions during the speech itself: stream the live transcript, resolve
each market term the instant Warsh utters it, and continuously update the probability of the
**unsaid** terms via Bayesian log-odds propagation — feeding a paper trader and logging RL
episodes for future offline training. Seeds from the validated Mode-1 priors (Plan 1).

**No LLM in the hot path.** Updates are deterministic resolution + statistical Bayesian
conditioning. This is the A.3-compliant "adapts as he speaks" behavior: the posterior moves, the
model weights never do.

---

## 2. Scope

**In:**
- `StreamText` caption client (the Fed's official real-time CART feed) behind the existing
  `Transcriber` seam, plus a `ReplayStream` that feeds a stored transcript as chunks for tests.
- Precomputed, frozen corpus **co-occurrence weights** `w(A→B)`.
- `LiveMentionTracker`: per-term log-odds state, **resolve-then-propagate** per caption delta,
  emits live P(mention) + resolution events.
- **RL-episode logger**: append `(state, action, reward)` tuples to a replay buffer
  (collection only — no training, no weight updates).
- A live **runner** wiring feed → tracker → paper trading (C7) → episode log.
- **Hearing-replay** end-to-end validation (April-21 transcript streamed through `ReplayStream`).

**Out (YAGNI / deferred):**
- **Real-time LLM forecasting** of unsaid terms (the "+ LLM" option was not chosen).
- **Clock/hazard (survival) decay** — real signal, but needs intra-document positional data the
  whole-document corpus only weakly supports; deferred, noted as a future evidence source.
- **Live capital.** Paper mode by default (Authority Matrix E4); real orders require the existing
  explicit `SPEECHEDGE_ALLOW_LIVE` + credentials gate (C7) and passing validation gates.
- **Self-run STT fallback** (Deepgram/ElevenLabs) — the interface allows it, but the
  implementation is deferred; StreamText is primary (see §6).
- Offline RL training itself (2C only *collects* episodes).

---

## 3. Architecture — the streaming loop

Seed each term's log-odds from its Mode-1 prior, then per caption delta: resolve, propagate, emit.

```
state[B] = to_logodds(p_prior[B])  for every market term B          # seed once (from Mode-1)

loop over caption deltas from the Transcriber (StreamText live, or ReplayStream in tests):
    running_text += delta
    # 1. RESOLVE (deterministic, certain)
    for each unsaid term B:
        if B.mentioned_in(running_text): P[B] = 1.0; mark resolved; retire B
    # 2. PROPAGATE (Bayesian, forward-looking) — only on NEW evidence in this delta
    for each evidence term A first-seen in this delta:
        for each unsaid term B:
            state[B] += w(A -> B)                                   # frozen, clamped
    # 3. EMIT
    P[B] = from_logodds(state[B]) for unsaid B
    -> paper trading signal (edge vs market price), resolution events
    -> RL-episode log (state, action, reward)
```

The **resolver** handles certainty (he said it → P=1 → settle); the **propagator** handles the
shifting odds on everything unsaid. They are deliberately separate so a real-money settlement
event never depends on the model being right — only the forward-looking bets do.

Reuses the existing `src/live/` primitives (`to_logodds`, `from_logodds`, `bayesian_update`,
`diction_loglikelihood`, the `Transcriber` Protocol) rather than rebuilding them.

---

## 4. Components & file structure

| File | Responsibility |
|---|---|
| `src/live/streamtext.py` | `StreamTextClient` (polls the JSON caption endpoint with a moving `last` cursor) + `ReplayStream` (yields a stored transcript in chunks). Both satisfy the existing `Transcriber` seam |
| `src/live/cooccurrence.py` | Precompute frozen weights `w(A→B) = clamp(log(P(B|A)/P(B)), ±W_MAX)` from the corpus |
| `src/live/tracker.py` | `LiveMentionTracker` — per-term log-odds state; `consume(delta)` does resolve-then-propagate; emits live P + resolution events; fires each evidence term once |
| `src/live/episode_log.py` | `EpisodeLogger` — append `(state, action, reward)` JSONL to a replay buffer (injectable path; collection only) |
| `live_run.py` | Runner: wires Transcriber + tracker + paper trading (C7) + episode log into the loop |
| `tests/test_streamtext.py`, `tests/test_cooccurrence.py`, `tests/test_tracker.py`, `tests/test_episode_log.py`, `tests/test_live_replay.py` | Unit + hearing-replay end-to-end |
| *reused* | `src/live/inference.py`, `src/live/types.py` (`Transcriber`, `MarketState`), `src/mentions` (`predict_event_priors`, `MarketTerm`), `src/trading` (C7 paper client/sizing) |

---

## 5. Propagation math

- **State** per term B: a log-odds value, seeded `state[B] = to_logodds(p_prior[B])`.
- **Resolve:** `B.mentioned_in(running_text)` → `P[B]=1.0`, retire B (no further updates, settle
  its paper position).
- **Propagate (fire-once):** when an evidence term A appears in the transcript **for the first
  time**, add `w(A→B)` to every unsaid B's log-odds. A term he repeats does NOT re-fire — repetition
  is a counting artifact, not new evidence.
- **Weight:** `w(A→B) = clamp(log(P(B|A)/P(B)), -W_MAX, +W_MAX)`, estimated from corpus
  co-occurrence (doc-level), bounded by `W_MAX` (default ~2.0 in log-odds). The clamp is the
  guardrail against a flukey pairing (two terms sharing 3 of 29 docs look perfectly correlated)
  yanking a position — the live analogue of the Mode-1 multiplier bounds.
- **Emit:** `P[B] = from_logodds(state[B])` for unsaid B.

Evidence terms A range over the curated lexicon (the same `fed_macro_terms.json` pool), so the
propagator listens to far more than the ~18 tradeable market terms.

---

## 6. Error handling & resilience

- **StreamText feed drops / wrong event ID** → the tracker keeps its current state and holds the
  last good P (degrades to Mode-1-prior trading, never to garbage). Self-run STT is the intended
  fallback behind the same `Transcriber` seam but is deferred (§2).
- **Resolver always runs** even if propagation is skipped — settlement of spoken terms is the
  high-value, model-independent path and must never be blocked.
- **Clamped weights** (`W_MAX`) bound any single correlation's effect.
- **Paper by default (E4):** `live_run.py` uses the C7 paper client; a real order requires the
  existing explicit `SPEECHEDGE_ALLOW_LIVE` + credentials gate.
- **Idempotent resolution:** a term already resolved is never re-settled or re-propagated.

---

## 7. Testing strategy

- **Pure tracker** (`test_tracker.py`): seed from fixed priors, feed scripted deltas, assert
  (a) a term resolves to P=1 exactly when its pattern first appears, (b) an unsaid correlated
  term's P moves in the right direction on evidence, (c) fire-once (a repeated evidence term does
  not move P twice), (d) resolved terms are retired and never re-updated.
- **Co-occurrence** (`test_cooccurrence.py`): on a tiny synthetic corpus with known co-occurrence,
  assert `w(A→B)` has the right sign and is clamped at `±W_MAX`.
- **StreamText/Replay** (`test_streamtext.py`): `ReplayStream` yields the expected chunks; the
  `StreamTextClient` JSON parsing + cursor advance tested against a fake HTTP response (no network).
- **Hearing replay end-to-end** (`test_live_replay.py`): stream the cleaned April-21 transcript
  through `ReplayStream` → assert the tracker resolves the known-said terms (rate cut, trump,
  inflation, independence, QE) by end, and never resolves the unsaid ones (recession, stagflation,
  soft landing). This is the offline proof the live loop works before June 16.
- Frozen weights computed once from the corpus; tests never hit the network or an LLM.

---

## 8. Open items to resolve in planning

1. **StreamText event ID** for Warsh's debut presser — confirm when the June broadcast page
   publishes (Powell's was `CFI-FRB`); the `StreamTextClient` takes the event id as config.
2. **`W_MAX` clamp** default (~2.0 log-odds) — set conservatively; revisit against the
   hearing-replay behavior.
3. **Trade cadence** — emit a signal per caption delta vs. on a fixed interval; default per-delta,
   with the paper trader de-duping unchanged signals.
4. **Episode `(state, action, reward)` schema** — finalize the exact tuple fields for the replay
   buffer (state = per-term P + elapsed; action = paper trade; reward = realized resolution).
5. **Co-occurrence recency** — whether `P(B|A)` uses recency-weighted doc counts (consistent with
   Mode-1's `as_of`); default unweighted for the frozen matrix, revisit in 2B.

# Design — Warsh Mention-Prediction Model

**Date:** 2026-06-11
**Status:** Approved (design); pending spec review → implementation plan
**Deadline driver:** June 16–17 2026 FOMC — Warsh's debut press conference

---

## 1. Objective

Build a **reusable model that predicts which words/phrases Kevin Warsh will utter** during
a speaking event, expressed as a calibrated probability **P(mention)** per candidate term.
The predictions feed trading on **prediction-market "Mentions" markets** (Kalshi
`/category/mentions`, Polymarket), where each sub-market resolves YES/NO on whether a
specific term is spoken during the event.

We are **not** trading policy-outcome markets ("Warsh cuts rates at first meeting"). The
edge is purely diction: our P(mention) vs. the market's implied price on each term.

### Two prediction moments
- **Before the speech (Mode 1):** a *prior* P(mention) per term — the pre-event bet.
- **During the speech (Mode 2):** a *live* P(mention) that updates in real time as the
  transcript streams, resolving terms YES the instant they are spoken.

### Reuse scope
**Same speaker (Warsh), any event.** The speaker model (his corpus + lexicon) is fixed.
What changes per event is the **term list** (from the market), the **news context**, and the
**live transcript**. The engine takes `(corpus, term_list, news, transcript) → predictions`
so a new Warsh event is new config + new inputs, not a rewrite. Generalizing to other
speakers (Powell, Trump) is explicitly **out of scope** for now but must not be designed out:
all Warsh-specific data stays behind config (corpus path, lexicon), never hardcoded in logic.

---

## 2. Inputs and Outputs

### Inputs (per event run)
1. **Past speeches** — the fixed Warsh corpus (`corpus/warsh_corpus.jsonl`, 29 docs /
   ~117k words). Supplies historical base rates.
2. **News context (deep)** — far more than headlines. A *context pack* assembled before the
   event from: full financial-press articles (WSJ / Bloomberg / Reuters / FT), other FOMC
   members' recent speeches + the FOMC statement + minutes, market-implied rate path (CME
   FedWatch / rate futures), the Trump–Fed political backdrop, and any CPI/jobs prints in the
   meeting window.
3. **Live transcript** — streaming text of the event as it happens (Mode 2 only). Primary
   source is the Fed's official real-time caption feed (see §5).

### Term vocabulary
**Supplied by the market.** We scrape the Kalshi / Polymarket Mentions market's published
sub-market terms for the event (e.g. the 18 terms for the June presser: "Rate Cut / Cut Rate",
"Trump", …). Each market term expands to a set of **surface patterns / phrasing variants**
("rate cut" ⇄ "cut rate" ⇄ "lower the policy rate") so a mention is detected however he
phrases it. The model does **not** invent its own terms (discovery is out of scope).

### Outputs
- **Mode 1:** `{term → P_prior(mention)}` for every market term.
- **Mode 2:** `{term → P_live(mention)}`, updated continuously; resolved terms pinned at 1.0.
- Both feed: **edge = |P_model − market_price|**, gated and Kelly-sized by the existing
  trading layer (C7), in **paper mode** by default (Authority Matrix E4 — live capital needs
  explicit owner sign-off and passing gates).
- **Target the mispriced terms, not uniform coverage.** Prior art (SOTU mention markets) shows
  the obvious high-frequency terms are efficiently priced (~90%+) — no edge there. The trading
  layer **ranks terms by `|P_model − price|`** and concentrates on the uncertain middle
  (~0.3–0.7) and news-sensitive tail, where our news-conditioning can beat stale base-rate
  pricing. A correct prediction on a term the market already prices at 0.95 earns ~nothing.

---

## 3. Architecture — the Hybrid engine

The model fuses the three inputs with a **statistical backbone + bounded LLM reasoning**.
The LLM is used where it is irreplaceable (reasoning about news, forecasting upcoming
content) and statistics are used where speed, determinism, and calibration matter.

### 3.0 Phase decomposition (prepared statement vs. Q&A)

A market term resolves YES if it is spoken **anywhere** in the event — opening statement *or*
Q&A. So phase is **not** a market distinction; it is an internal decomposition of how we
estimate the probability, because the two phases have very different dynamics:

```
P_event(term) = 1 − (1 − P_prep(term)) · (1 − P_qa(term))
```

- **P_prep — prepared statement.** *More* predictable, not less. Fed opening statements are
  heavily templated (mandate → current conditions → policy stance → forward guidance), so this
  is a high-confidence estimate from corpus base-rate + the recurring structural skeleton of
  his/Fed opening remarks. Edge here is **early** (before the market converges) and on subtle
  terms; once the statement is released/read, these terms resolve deterministically.
- **P_qa — question & answer.** The genuinely uncertain, unscripted part — driven by his
  diction habits + the day's news, and the focus of live updating. This is where the durable
  forecasting alpha lives (Q&A can't be front-run from a leaked script).

Both phases are predicted and combined into the single `P_event` the market resolves on. In
live Mode 2, once the prepared statement is delivered, its terms collapse to resolved {0,1} and
the live updater's remaining work is almost entirely **Q&A terms**.

### Mode 1 — before the speech (prior)
```
corpus ─► Base-rate estimator ─► P_base(term)        (statistical, smoothed)
news   ─► News adjuster (LLM)  ─► multiplier(term)    (Opus 4.8, bounded)
                                  P_prior = clip(P_base × multiplier, 0, 1)
```

- **Base-rate estimator (statistical).** For each term, `P_base` = smoothed fraction of
  Warsh's prior comparable events that contain the term. **Add-k (Laplace) smoothing gives a
  nonzero floor** so a term he has never said on record is not trapped at 0 (the bounded
  multiplier can then lift it). Phrasing variants are folded together before counting.
  Produces **phase-conditioned** rates `P_prep` (from prepared remarks / opening-statement
  structure) and `P_qa` (from Q&A-style sources), combined per §3.0.
- **News adjuster (LLM, Opus 4.8).** Claude reads the deep news pack + term list and emits a
  **bounded multiplier per term** with a one-line rationale. **Bound: 0.25× – 4×.** The LLM can
  say "today's news makes 'independence' 4× more likely than baseline" but can **never invent a
  probability from nothing** — it only scales an existing (smoothed) base rate. This guardrail
  means a hallucination cannot blow up a position. Latency is irrelevant here (runs once,
  pre-event), so the strongest model is used.

### Mode 2 — during the speech (live, real-time LLM)
```
live caption line ─► [deterministic] term already said? ─► resolve YES, pin P=1, stop trading it
                  └► [LLM, fast]      P_live(unsaid term) ← forecast of upcoming speech
                                       prior + transcript-so-far → updated P per remaining term
```

- **Two-tier resolution.** Detecting a term was *already spoken* is a deterministic regex
  match on the incoming caption line — instant, no LLM, no cost. The LLM's job is purely
  **forward-looking**: P(an unsaid term still comes up), which requires it to predict the next
  stretch of his speech (the "next N words" intuition) given context + what he's said so far.
- **Real-time LLM (Haiku 4.5 / Sonnet 4.6).** Fast model in the loop, called per new caption
  line (or every few seconds). **Prompt caching is what makes this affordable:** the stable
  prefix — corpus context + news pack + term list + Mode-1 prior — is cached, so each live call
  pays only for the *new transcript delta*. A ~50-min presser ⇒ a few hundred cached calls.
- **Inference-only (brief A.3).** Repeated LLM *inference* as evidence arrives is permitted;
  model **weights are never updated mid-speech**. Frozen weights, streaming evidence.

---

## 4. Component mapping (mostly reframe, not rebuild)

| Existing | Role in this design | Change required |
|---|---|---|
| **C1 corpus** (`scrape_warsh.py`, `src/warsh/`) | Speaker history input | None (done; hearing cleaned) |
| **C2 diction model** (`src/warsh/model.py`) | **Base-rate estimator** | Add `mention_base_rate(term)` w/ add-k smoothing + variant folding |
| **C4 context agent** (`src/agents/`) | **News adjuster** | Expand sources to deep pack; output bounded per-term multipliers |
| **C6 live inference** (`src/live/`) | **Mode-2 live updater** | Reframe from next-token → mention-tracking; add StreamText poller + real-time LLM call + deterministic resolver |
| **C5 backtest** (`src/backtest/`) | **Validation** | Replace broken next-token G1 with mention-calibration (Brier) gate |
| **C7 trading** (`src/trading/`) | Signal/sizing on mentions markets | Map per-term P → per-sub-market signal; scrape market terms+prices |
| **C8 dashboard** (`src/dashboard/`) | Display | Show per-term P(mention), prior vs live, resolution state |

New modules (event/term plumbing): a **term-vocabulary loader** (scrape market →
terms + variant patterns) and a **StreamText caption client** (live transcript).

---

## 5. Live transcript acquisition

**Primary — Fed StreamText CART feed (human captioner).** The Fed contracts real-time
captioning (event `CFI-FRB`). StreamText exposes a documented **Realtime Caption Pull API**:

```
GET https://www.streamtext.net/captions?event=CFI-FRB&last={cursor}&length={N}&language=en
→ {"content": "...", "lastPosition": 1234, "languageCode": "en"}
```

Poll with a moving `last` cursor; returns new caption lines as JSON and handles its own
corrections/backspaces. Verified live 2026-06-11: returns the documented JSON shape (404 +
empty content only because no event is currently broadcasting). Human stenographer ⇒ ~99%+
accuracy, low latency, no API key, free to pull. **This is near-ground-truth input** and
removes the STT word-error risk the brief assumed.

**Fallback — self-run STT** on the Fed live audio (`federalreserve.gov/live-broadcast.htm`
or the Fed YouTube live) via **Deepgram Nova-3** (~300 ms, $0.0077/min, ~6.8% WER) or
**ElevenLabs Scribe v2 Realtime** (~150 ms). Used only if the event ID changes for Warsh's
debut or the StreamText feed is unavailable. The Mode-2 input is an injected interface so
either source plugs in behind the same contract.

**Risk to confirm closer to the event:** the StreamText event ID for *Warsh's* debut presser
may differ from Powell's `CFI-FRB`; verify when the June broadcast page goes up, with STT as
the standing fallback.

---

## 6. Validation — mention calibration (replaces broken G1)

The shipped G1 metric (`top1_accuracy`) measures generic **next-token** accuracy and is the
wrong test for a mentions model (it scores predicting "the"/"of", not whether a term appears).
Spec drift: ROADMAP / agent gate table / C5 plan all say "top-1 **phrase** accuracy."

**New gate — G1′ Mention Calibration.** For a vocabulary of candidate terms, the Mode-1 model
predicts `P(mention)` for a held-out event; we compare to the realized YES/NO outcomes.

- **Ground truth in hand:** the **April-21 2026 confirmation hearing is a resolved Mentions
  market**, and we hold the cleaned Warsh-only transcript. We can predict its terms from
  prior-only corpus data (date-split, no leakage) and score against actual resolutions.
  (Sanity-checked 2026-06-11: the transcript correctly resolves "Rate Cut/Cut Rate"=YES,
  "Trump"=YES, etc.)
- **Metric:** **Brier score** on per-term mention prediction (calibration — required for Kelly
  sizing to be sane), plus mention hit-rate / AUC for ranking. Threshold to be set during
  implementation against the Powell-presser + hearing baselines (the legacy ">20% next-token"
  number does not transfer).
- **G3 (market mechanics)** on Powell pressers stays as-is. **G2** (Brier) effectively merges
  into G1′ since the mentions objective *is* a calibration objective.
- **No live capital until gates pass** (brief §6).

---

## 7. Data flow (per event)

```
PRE-EVENT (Mode 1):
  scrape market terms ──┐
  corpus ──► base rates ─┼─► P_prior ──► trade prior (paper)
  news pack ──► LLM × ───┘

DURING EVENT (Mode 2), loop:
  poll StreamText ──► caption delta
        ├─ deterministic resolver: any term said? → pin P=1, retire it
        └─ fast LLM (cached prefix + delta): P_live for unsaid terms
                                   └─► trade live (paper)

POST-EVENT:
  realized resolutions ──► calibration report (Brier) ──► gate review
```

---

## 8. Error handling & resilience

- **StreamText drops / wrong event ID** → fall back to STT source behind the same interface;
  if both fail, Mode 2 degrades to *holding the Mode-1 prior* (no live updates) rather than
  trading on stale/garbage text.
- **LLM call slow/fails in live loop** → skip the update for that tick (keep last good P);
  never block the resolver (deterministic path always runs). Bounded retry, then hold.
- **Multiplier hallucination** → contained by the 0.25×–4× clip and smoothed base-rate floor.
- **Live mode is paper by default** (E4); real orders require the existing explicit
  `SPEECHEDGE_ALLOW_LIVE` + credentials gate (C7).
- **Exact Anthropic API params** (prompt-cache headers, `thinking`/`effort` config, model IDs
  `claude-opus-4-8` / `claude-sonnet-4-6` / `claude-haiku-4-5`) to be validated against the
  claude-api skill at implementation time.

---

## 9. Testing strategy

- **Pure cores, offline:** base-rate estimator (incl. smoothing floor + variant folding),
  deterministic resolver, calibration/Brier metric — unit-tested on synthetic + small fixtures.
- **Injected interfaces:** LLM (news adjuster, live updater), StreamText client, market client
  — behind protocols, tested with fakes so the pipeline runs with no network/API.
- **End-to-end replay:** feed the April-21 hearing transcript through Mode 2 as a simulated
  live stream → confirm it resolves the known terms in order and the calibration gate runs.
- Doc-level (never sentence-level) train/test split preserved for any held-out scoring.

---

## 10. Out of scope (YAGNI)

- Other speakers (Powell/Trump) — interfaces stay speaker-agnostic, but no second corpus now.
- Model-discovered terms — trade only market-published terms.
- Policy-outcome markets.
- Runtime weight updates / fine-tuning (forbidden by A.3).

---

## 11. Open items to resolve in planning

1. Calibration gate threshold (set from Powell + hearing baselines).
2. Mode-2 LLM call cadence (per caption line vs fixed interval) and which fast model.
3. Term-variant pattern authoring (manual list vs LLM-assisted expansion, reviewed).
4. Confirm Warsh-debut StreamText event ID when the June broadcast page publishes.
5. **P_prep for a first-time Chair.** Warsh has *no* prior FOMC-presser opening statements
   (he was a Governor, never Chair — see R3). Estimate `P_prep` from the **institutional**
   opening-statement template (Powell's presser structure) for *what topics get covered*,
   conditioned on **Warsh's diction** for *word choice* — rather than from his own
   (nonexistent) presser history. Resolve the exact blend in planning.

# DECISIONS.md — SPEECHEDGE

Running log of decisions. Each entry: date · decision · rationale · authority (Autonomous / Escalated-default-accepted / Owner-decided).

Format reference: Authority Matrix is §7 of the Program Brief. Defaults accepted are logged here per Rule 2.

---

## 2026-06-09

### D1 — Workspace: new dedicated repo `~/Downloads/SpeechEdge`
- **Decision:** SPEECHEDGE lives in its own git repo, separate from the insider-trades project (`HudgeFundPoliticianSourceScraper`).
- **Rationale:** Unrelated projects; mixing histories couples deployment, deps, and risk. Clean separation.
- **Authority:** Owner-decided (confirmed 2026-06-09).

### D2 — All components are NOT BUILT; build from scratch
- **Decision:** Treat C1, C2, and C8 as NOT BUILT. The brief's "BUILT" status column was inaccurate — none of the named files exist on this machine.
- **Rationale:** Verified against filesystem: no `scrape_warsh.py`, `build_model.py`, or React dashboard anywhere under `~/Downloads`. The only SPEECHEDGE artifact found was `Free_App/logos/kalshi.png`.
- **Impact:** Critical path changes. C1 (scraper) must be *written* before Neal can run it; the corpus spine (C1→C2→C5) does not start until scraper code exists. See RISKS.md R1.
- **Authority:** Owner-decided (confirmed 2026-06-09).

---

## 2026-06-10

### D3 — `phrase_signals` returns `list[dict]`, not `dict[str,float]`
- **Decision:** The diction model's `phrase_signals(text)` returns `[{"phrase","axis","weight"}, ...]` (matched signal phrases with their axis + weight), superseding the original INTERFACES §2 `dict[str,float]` (phrase→lift) sketch.
- **Rationale:** The richer form carries the signal **axis** (hawkish/dovish/independence/qt) and **weight**, which C5/C6 need to map diction → market signal. Lift scores live separately in the fingerprints module. Implemented + tested in C2.
- **Authority:** Autonomous (code interface; §7). INTERFACES.md §2 updated to match.

### D4 — Foreground-only builds; consistent commit+push to black-box
- **Decision:** Build components in the **foreground** (background subagents lack Bash here). Commit + push to `origin` (black-box) after every component. Hard rules recorded in CLAUDE.md.
- **Authority:** Owner-directed (2026-06-10).

---

### D5 — Corpus scraping works from this environment; brief's "sandbox blocks domains" claim is FALSE
- **Decision/finding:** Empirically verified (2026-06-11) that Bash/`requests` and WebFetch reach `federalreserve.gov` and `hoover.org` (HTTP 200). The brief's §7 claim that "the agent network sandbox blocks every target domain" is **not true here** — so C1 scraping is NOT owner-blocked. Fable ran it directly.
- **Done:** Pulled all **18 Warsh Board-of-Governors speeches (2006–2010)** from federalreserve.gov via `scrape_warsh.py` (verified URLs now in `SOURCES`), plus 3 recent Hoover docs (Commanding Heights 2025 lecture, Cogan–Warsh 2022 essay, "Inflation Is A Choice") into `corpus/manual/`. Corpus = **21 docs / 80,387 words**. Model now predicts `inflation is a` → `choice` (0.75) and the §4.1 signal phrases are present.
- **Impact:** R1/R4 substantially de-risked — the corpus critical path is unblocked without waiting on Neal. Remaining gap is recency/coverage (see RISKS R4), not access.
- **Authority:** Owner-directed ("absolutely necessary"); finding logged.

## 2026-06-11

### D6 — Mentions-market reframe; G1' (Brier calibration) replaces next-token G1
- **Decision:** SPEECHEDGE trades prediction-market *Mentions* markets (Kalshi/Polymarket) — P(Warsh utters term T during an event) vs. market price — not policy-outcome markets. The diction model is validated by **mention calibration (Brier)**, not the mis-specified next-token "top-1 accuracy" (which scored predicting function words). See spec `docs/superpowers/specs/2026-06-11-mention-prediction-model-design.md`.
- **First real number (G1'):** Mode-1 predictor (phase-decomposed add-k base rate x bounded news multiplier) backtested on the **resolved April-21 hearing** (leakage-free date split, 28 train docs): **Brier = 0.1528, PASS** (< 0.25 placeholder threshold). The legacy next-token G1 read 17.3% and is retired. Dominant error: `recession` (predicted 0.814, not said) — argues for recency-weighting in Plan 2.
- **Scope:** Plan 1 (Mode-1 predictor + calibration gate) shipped via subagent-driven TDD. Mode-2 live (StreamText + real-time LLM) = Plan 2; trading + dashboard = Plan 3.
- **Authority:** Owner-directed (mentions reframe) + Autonomous (metric correction, logged).

---

## 2026-06-12

### D7 — Level-1 calibration harness shipped (2A); first multi-event read exposes empty-phase overconfidence
- **Decision:** Mention-model validation is now multi-event: leave-one-event-out over Warsh's corpus against a curated 40-term Fed/macro lexicon, scored as a pooled (prediction, outcome) set with Brier, log-loss, AUC, and a reliability diagram. Speaker-agnostic via CorpusLoader/VocabularySource seams (Powell/SOTU real market term-lists plug in later). Spec: `docs/superpowers/specs/2026-06-12-backtest-harness-design.md`.
- **Result (Warsh LOO, recency 4y):** 26 events scored / 3 skipped, 1040 (pred,outcome) pairs, base rate 0.327. **Brier 0.2658, log-loss 0.7321, AUC 0.8123.** The rosy n=1 hearing (Brier 0.1375) was misleading, as predicted.
- **Finding (drives 2B):** Strong ranking (AUC 0.81 — selection works) but badly miscalibrated sizing. Root cause, confirmed by the reliability diagram + a per-term probe: when a speaker has ZERO prior docs in a phase, that phase's add-k rate defaults to 0.5 (the n=0 Jeffreys prior), and the noisy-OR then floors EVERY term's event probability at >=0.5 — inflating rare terms into the 0.5-0.6 band (37% of predictions, ~10% observed). Fix in 2B: an unobserved phase must contribute ~0, not 0.5; then recalibrate.
- **Scope:** 2A done. 2B's first task is the empty-phase fix + recalibration. 2C is live Mode-2. Level-2 P&L remains out (no historical price data).
- **Authority:** Owner-directed.

---

## Accepted defaults (from Authority Matrix §7)
Logged per Rule 2 — accepted unless evidence says otherwise.

| ID | Decision | Default accepted | Status |
|----|----------|------------------|--------|
| E2 | Signal threshold | $0.08 (recalibrate from C5 evidence; confirm changes) | Accepted, pending C5 |
| E3 | Audio source | Deepgram API (~300ms) over local Whisper (~1.5s) | Accepted, **cost confirmation pending from Neal** |
| E4 | June 16 mode | Paper trading | Accepted (going live needs explicit owner sign-off) |

### Still blocking — no default
- **E1 Capital allocation:** No default. Kelly sizing blocked until Neal provides.
- **E5 Kalshi account:** Funded account + trading-enabled API key. Owner action, longest external lead time — chase day 1.

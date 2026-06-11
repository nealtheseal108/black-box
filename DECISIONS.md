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

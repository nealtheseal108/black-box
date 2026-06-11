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

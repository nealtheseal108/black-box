# CLAUDE.md — SPEECHEDGE

You are **Fable, the program manager** for SPEECHEDGE (Brief §0). This file is read automatically every session so the PM context survives across days without re-pasting.

## ⛔ Working directory (HARD RULE — never violate)
- **All work happens in `/Users/nealt1/Downloads/SpeechEdge`.** Every read, write, edit, test run, and commit.
- **Never touch `/Users/nealt1/Downloads/HudgeFundPoliticianSourceScraper`** (the unrelated insider-trades repo / FreeportMarkets). It has nothing to do with SpeechEdge. Do not read, write, branch, worktree, or commit there — ever.
- This session may have launched from the HedgeFund directory; that does not matter. Operate on SpeechEdge by **absolute path** and use **`git -C /Users/nealt1/Downloads/SpeechEdge`** for every git command. Confirm `pwd`/target before any file or git operation if unsure.

## 🔁 Commit cadence (HARD RULE)
- **Commit + push to `origin` (`github.com/nealtheseal108/black-box`) consistently** — after every completed task or component, and before ending a turn with new work. Don't let work sit uncommitted.
- `git -C /Users/nealt1/Downloads/SpeechEdge add -A && git -C ... commit -m "..." && git -C ... push origin master`
- Commit messages end with the `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` trailer.
- **Do not use background subagents for builds** — they lack Bash permission in this environment and can't run tests/commits. Build in the foreground.

## Source of truth
- `SPEECHEDGE_BRIEF.md` (Neal will place it in the repo root) is the **single source of truth**. Read it before any work.
- **Do not change the product intent** (Brief §1). Optimize execution, not strategy.
- Check the **Authority Matrix (Brief §7)** before every decision. Log accepted defaults.

## Standing rules
1. Maintain `DECISIONS.md` and `RISKS.md`. Surface escalations in **batches**, not one-at-a-time, unless deadline-critical.
2. **OWNER-EXECUTED** work runs on Neal's machine, not yours: all scraping (C1/C3), manual corpus ingestion, account creation, funding, credentials. Prepare/verify/unblock — never execute.
3. **June 16–17 2026 FOMC is immovable.** Paper trading by default (E4). **No live capital until validation gates G1–G4 pass** (Brief §6).
4. Kelly sizing is **blocked** until Neal provides E1 capital allocation.

## Where things are
- `docs/ROADMAP.md` — master execution roadmap (revised critical path).
- `docs/INTERFACES.md` — shared data contracts; all components build against these.
- `docs/superpowers/plans/` — per-subsystem implementation plans.
- `src/warsh/` — corpus + diction model code. `corpus/` — scraped output (gitignored). `corpus/manual/` — owner-dropped paywalled docs.

## Conventions
- Python 3.11+, pytest, TDD. Pure transforms separated from I/O (network/file).
- **Secrets never committed.** API keys via env / `.env` (gitignored): `KALSHI_API_KEY`, `DEEPGRAM_API_KEY`, `ANTHROPIC_API_KEY`.
- Trading code (`kalshi_trader.py`, anything that places orders) gets reviewed by the `trading-code-reviewer` agent before merge — bugs there cost real money.
- Model changes get validated by the `backtest-validator` agent against gates G1–G4 before they inform any sizing.

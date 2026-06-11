# RISKS.md — SPEECHEDGE

Living risk register. Severity: 🔴 critical (deadline/capital-threatening) · 🟠 high · 🟡 medium.
Each: ID · description · severity · owner · mitigation · status.

---

## 🔴 R1 — "BUILT" deliverables do not exist; full build from scratch under a compressed deadline
- **Severity:** 🔴 critical
- **Detail:** Brief §2–3 claimed C1/C2/C8 were BUILT. Verified false — nothing on disk. The original 5-week plan assumed corpus + model already existed; we have ~7 days (June 9 → June 16 FOMC) to build all seven components from zero.
- **Impact:** Corpus spine (C1→C2→C5) cannot start until scraper code is written. C1 execution is owner-only (sandbox). Compresses every downstream gate (G1–G4).
- **Mitigation:** (1) Front-load the right-hand parallel branch (C4, C7, C8) which has no corpus dependency. (2) Write C1 scraper code immediately so Neal can run it day 1. (3) Re-scope to a *paper-trading* June 16 target (E4 default) — no live capital regardless, which removes the capital-loss tail risk while we validate.
- **Owner:** Fable (build) + Neal (run C1)
- **Status:** OPEN — top priority.

## 🔴 R2 — E5 Kalshi account is the longest external lead time
- **Severity:** 🔴 critical (to live mode only; paper mode unaffected)
- **Detail:** Funded account + trading-enabled API key require KYC/approval latency outside our control. No engineering compresses it.
- **Mitigation:** Escalated to Neal day 1. C7 builds against Kalshi's public/paper endpoints first so code is ready the moment credentials land. Paper-trade June 16 does not strictly require a funded account (read-only market data + simulated fills).
- **Owner:** Neal
- **Status:** OPEN — chase immediately.

## 🟠 R3 — Warsh has zero presser history → market-level (Level-2) backtest impossible pre-debut
- **Severity:** 🟠 high
- **Detail:** Settled in Appendix A.4. We can only validate the *pipeline* on Powell's last 10 pressers; Warsh's diction model is validated on held-out corpus (G1). First 3 Warsh meetings are live validation at minimal size.
- **Mitigation:** Two-level backtest (§6): diction accuracy on held-out Warsh corpus (G1) + market mechanics on Powell (G3). Accept residual model-transfer risk; size minimally for first meetings.
- **Owner:** Fable
- **Status:** ACCEPTED (per Appendix A.4) — design backtest accordingly.

## 🟠 R4 — Corpus quality/coverage gates everything (garbage-in)
- **Severity:** 🟠 high
- **Detail:** Highest-value single document (April 21 2026 confirmation hearing) and paywalled WSJ pieces require manual ingestion by Neal. If corpus is thin or skewed (e.g. only old FRASER speeches), the diction model won't reflect Warsh's *current* triangulated stance.
- **Mitigation:** Scraper code prioritizes recency-weighted, high-signal sources; corpus/manual/ ingestion path for paywalled material; G1 held-out accuracy gate catches a weak model before any capital.
- **Owner:** Neal (ingest) + Fable (scraper + model)
- **Status:** 🟢 LARGELY MITIGATED (2026-06-11). Fable pulled a 25-doc / 128,622-word corpus directly (sandbox doesn't block the domains — see DECISIONS D5): all 18 Fed Governor speeches (2006–2010) + the **April-21-2026 confirmation hearing** (27k-word Rev transcript — the brief's highest-value doc) + 6 Hoover essays/lectures/interviews (incl. Commanding Heights, "Inflation Is A Choice"). Both eras covered; model predicts `inflation is a`→`choice`. **Remaining gaps:** (1) WSJ op-eds (paywalled; archive.org CDX timed out) — non-blocking since the §4.1 signal lexicon is hardcoded and detects phrases live regardless of corpus frequency; (2) the hearing transcript mixes senators' words with Warsh's — a future refinement is to extract Warsh-only turns for cleaner diction stats; (3) C2 fingerprint-lift baseline is too small (noisy stopword lift) — model is fine, the lift *report* needs a real baseline.

## 🟡 R5 — Live latency budget (STT → inference → order) is tight
- **Severity:** 🟡 medium
- **Detail:** Edge depends on firing before the market reprices. E3 default Deepgram (~300ms) chosen over local Whisper (~1.5s). Inference must stay inference-only (Appendix A.3) — no runtime weight updates.
- **Mitigation:** Bayesian inference-only updating; pre-computed priors from Mode 1; profile end-to-end on archived FOMC audio (C6 rehearsal) before June 16.
- **Owner:** Fable
- **Status:** OPEN — measure during C6.

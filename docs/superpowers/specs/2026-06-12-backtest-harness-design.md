# Design — Level-1 Calibration Backtest Harness (Sub-project 2A)

**Date:** 2026-06-12
**Status:** Approved (design); pending spec review → implementation plan
**Parent:** "Plan 2" decomposed into 2A (this), 2B (hyperparameter tuning), 2C (live Mode-2).
**Predecessor:** Plan 1 (Mode-1 mention predictor) — `docs/superpowers/specs/2026-06-11-mention-prediction-model-design.md`

---

## 1. Objective

Turn mention-model validation from **n=1** (the single resolved April-21 hearing) into a
statistically meaningful sample, so we can (a) trust the Mode-1 priors we would paper-trade on
June 16, and (b) tune the model's hyperparameters honestly (2B) and, much later, feed any
offline RL (2C). This is a **Level-1 calibration** harness: it scores the model's *probabilities*
against realized mention outcomes — not trade P&L.

**Why calibration is the money metric, not an academic one:** miscalibrated probabilities are
how Kelly sizing goes broke. If the model says 0.7 but 0.7-predictions only hit 50%, Kelly
oversizes every bet and the edge inverts. The reliability diagram is the curve sizing depends on.

---

## 2. Scope

**In:**
- Leave-one-event-out (LOO) calibration on Warsh's corpus (his 29 dated docs → ~25 scored events
  after the cold-start guard).
- A curated Fed/macro **lexicon** as the candidate-term vocabulary for Warsh's events (his past
  speeches predate Kalshi/Polymarket mention markets, so no real market terms exist for them).
- Aggregate calibration metrics over a flat pool of `(predicted_probability, binary_outcome)`
  pairs: Brier, log-loss, AUC, and a binned reliability diagram.
- Two **injected seams** so cross-speaker data plugs in later without a redesign:
  `CorpusLoader` (docs per speaker) and `VocabularySource` (terms per event).

**Out (YAGNI / later sub-projects):**
- **Level-2 P&L backtest** and any historical market *price* data (deferred entirely — not even
  a hook; the user chose Level-1-only).
- **Cross-speaker data sourcing** (Powell presser scraping, historical Kalshi/Polymarket
  *term-list* retrieval). The harness exposes the seams; the `MarketVocabulary` /
  Powell `CorpusLoader` implementations are a later build. For Powell/SOTU we will pull real
  historical mention-market **term lists** (not prices) when that work is scheduled.
- Hyperparameter tuning (2B), live Mode-2 (2C), RL.

---

## 3. Architecture — LOO pool-reduction

The harness collapses "many events × many terms" into one flat pool of
`(predicted_probability, binary_outcome)` pairs, then scores that pool once. Every metric is a
different read of the same pool, which is what keeps it speaker-agnostic — Warsh-LOO, and later
Powell/SOTU, all just *add pairs to the pool*.

```
for each held-out event E (one of the speaker's dated docs):
    train_docs = corpus_loader.docs_for(speaker), kept strictly BEFORE E.date   # temporal, no leakage
    if len(train_docs) < min_prior_docs: skip and log                            # cold-start guard
    terms    = vocabulary_source.terms_for(E)                                    # SEAM 1
    preds    = predict_event_priors(train_docs, terms, news_context="",
                                    adjuster=NullAdjuster(), as_of=E.date)        # Plan-1 reuse
    outcomes = resolve_outcomes(terms, E.text)                                   # 1 if term in E.text else 0
    for term: pool.append( (preds[term]["p_prior"], outcomes[term]) )

summary = calibration_summary(pool)   # Brier, log-loss, AUC, reliability bins, base rate, n_scored
```

### Injected seams
- **`CorpusLoader`** — `docs_for(speaker: str) -> list[dict]` (dated docs). `JsonlCorpusLoader`
  wraps `corpus/warsh_corpus.jsonl`. Later: a Powell loader over scraped pressers.
- **`VocabularySource`** — `terms_for(event: Event) -> list[MarketTerm]`. `LexiconVocabulary`
  loads the curated lexicon now. Later: `MarketVocabulary` returns the real historical
  Kalshi/Polymarket term list for that event. This is the curated-for-Warsh /
  market-terms-for-others split as two implementations of one interface.

---

## 4. Components & file structure

| File | Responsibility |
|---|---|
| `src/backtest/events.py` | `Event` dataclass (`speaker, date, text, context_type`); `CorpusLoader` Protocol; `JsonlCorpusLoader` (loads Warsh jsonl → events, sorted by date) |
| `src/backtest/vocabulary.py` | `VocabularySource` Protocol; `LexiconVocabulary` (loads curated lexicon JSON → `list[MarketTerm]` via Plan-1 `load_terms`) |
| `corpus/lexicon/fed_macro_terms.json` | Curated ~80-term Fed/macro pool (hawkish/dovish/independence/QT lexicon + observed market terms), in the Plan-1 `MarketTerm` JSON format (`{canonical, patterns}`) |
| `src/backtest/loo.py` | `leave_one_out(loader, speaker, vocab, min_prior_docs, as_of_recency) -> list[(pred, outcome)]` + per-event detail for reporting |
| `src/backtest/calibration.py` | **extend** (keep existing `brier_score`, `calibration_report`, `split_before`): add `log_loss`, `auc`, `reliability_diagram`, `calibration_summary` |
| `backtest_harness.py` | Runner: Warsh LOO with `LexiconVocabulary` → accumulate → print Brier/log-loss/AUC + reliability table + n_scored/n_skipped + base rate |
| `tests/test_events.py`, `tests/test_vocabulary.py`, `tests/test_loo.py`, `tests/test_calibration_metrics.py`, `tests/test_backtest_harness.py` | Unit + end-to-end tests |

Plan-1 reuse (no reimplementation): `MarketTerm`, `load_terms`, `predict_event_priors`,
`resolve_outcomes`, `event_base_rate` (recency-aware via `as_of`), existing `calibration.py`.

---

## 5. Metrics (and the money question each answers)

| Metric | Answers | Money meaning |
|---|---|---|
| **Reliability diagram** (deciles: mean_pred vs observed_freq, count per bin) | Does 0.7 mean 0.7? | Sizing safety — the curve Kelly depends on; off-diagonal ⇒ oversizing |
| **Log-loss** (cross-entropy) | Overall calibration (proper scoring rule) | The tuning objective for 2B (and any future RLVR) |
| **AUC** | Can it rank said-terms above unsaid? | Selection — which terms are worth betting at all |
| **Brier** | Combined calibration + sharpness | Headline gate; continuity with Plan-1's 0.1375 |

Both AUC and calibration are reported because they answer *different* money questions: AUC =
"do we have any edge to bet on," calibration = "can we size without blowing up." A model can ace
one and fail the other.

### Honesty guards
- **`min_prior_docs` cold-start guard** (default 3): skip held-out events whose speaker has too
  few prior docs, so we don't score predictions made off a near-empty (smoothing-floor) base
  rate. Report `n_scored` vs `n_skipped`.
- **Report base rate** (fraction of positive outcomes in the pool) alongside every metric — a
  reliability diagram is uninterpretable without knowing negatives dominate (which is the point
  of including them; avoids the survivorship trap).

---

## 6. Error handling

- Event with `< min_prior_docs` prior docs → skipped, counted in `n_skipped`.
- Empty vocabulary or empty event text → skip with a logged warning (never silently score zero).
- Missing/invalid `date` on a doc → excluded from both train and held-out sets, logged (current
  corpus has valid ISO dates on all 29).
- `log_loss` clamps predictions to `[eps, 1-eps]` (eps=1e-15) to avoid `log(0)` blowups.
- `auc` with only one outcome class present (all-0 or all-1 pool) → return `None` and note it
  (AUC undefined); the run still reports Brier/log-loss.

---

## 7. Testing strategy

- **Pure metrics** (`test_calibration_metrics.py`): `log_loss`, `auc`, `reliability_diagram` on
  synthetic pools with known answers (perfect/worst/random predictors; a hand-checkable AUC; a
  reliability pool where bin frequencies are obvious).
- **LOO engine** (`test_loo.py`): a tiny synthetic single-speaker corpus (4–5 dated docs) +
  a 2-term fake vocab → assert the temporal split excludes the held-out and later docs, the
  cold-start guard skips early events, and the pool has the expected `(pred, outcome)` shape.
- **Seams** (`test_events.py`, `test_vocabulary.py`): `JsonlCorpusLoader` and `LexiconVocabulary`
  against small fixtures; confirm they satisfy their Protocols.
- **End-to-end** (`test_backtest_harness.py`): run on the real Warsh corpus + curated lexicon;
  assert `n_scored` in a sane range, base rate in (0,1), all metrics in valid ranges, and that
  the summary dict has the documented keys. Record the headline numbers.
- Doc-level (never sentence-level) temporal split preserved throughout.

---

## 8. Open items to resolve in planning

1. **Lexicon contents.** Author the ~80-term curated Fed/macro pool (sources: the hawkish/dovish/
   independence/QT signal lexicon already referenced in the project + the 8 observed market terms
   from the hearing fixture). Manual list, reviewed — patterns reuse the Plan-1 corrected regex
   style (e.g. the multi-word "rate cut" form).
2. **`min_prior_docs` default.** Start at 3; revisit once we see how many Warsh events survive.
3. **Recency in LOO.** Pass `as_of=E.date` so each held-out event is predicted with recency
   weighting to its own date (consistent with how live prediction will work). The harness is also
   how 2B will A/B the half-life, so expose it as a parameter.
4. **AUC implementation.** Use the rank-based (Mann–Whitney) formula to avoid a sklearn dependency
   (project has no ML deps); unit-test against a hand-computed case.

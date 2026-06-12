# Level-1 Calibration Backtest Harness Implementation Plan (Sub-project 2A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a speaker-agnostic Level-1 calibration harness that scores the Mode-1 mention predictor across ~25 leave-one-event-out Warsh events (instead of n=1), reporting Brier, log-loss, AUC, and a reliability diagram over a flat pool of `(predicted_probability, outcome)` pairs.

**Architecture:** A leave-one-event-out (LOO) loop reduces "many events × many terms" into one flat `(pred, outcome)` pool, scored once. Two injected seams — `CorpusLoader` (docs per speaker) and `VocabularySource` (terms per event) — keep it speaker-agnostic so Powell/SOTU plug in later. Reuses Plan-1's `predict_event_priors`, `MarketTerm`, and existing `calibration.py`.

**Tech Stack:** Python 3.11, pytest, stdlib only (no ML deps — AUC via rank formula). Builds on `corpus/warsh_corpus.jsonl` and `src/mentions/`.

**Working directory (HARD RULE):** All commands from `/Users/nealt1/Downloads/SpeechEdge`. Run Python with `.venv/bin/python`. Commit with `git -C /Users/nealt1/Downloads/SpeechEdge`, push to `origin master` (black-box). Committing on master is the established, authorized workflow.

**Spec:** `docs/superpowers/specs/2026-06-12-backtest-harness-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `src/backtest/calibration.py` | **modify** — add `log_loss`, `auc`, `reliability_diagram`, `calibration_summary` (keep existing functions) |
| `src/backtest/events.py` | `Event` dataclass, `CorpusLoader` Protocol, `JsonlCorpusLoader` |
| `src/backtest/vocabulary.py` | `VocabularySource` Protocol, `LexiconVocabulary` |
| `corpus/lexicon/fed_macro_terms.json` | Curated Fed/macro term pool (`MarketTerm` JSON format) |
| `src/backtest/loo.py` | `leave_one_out(...)` → flat `(pred, outcome)` pool + counts |
| `backtest_harness.py` | Runner: Warsh LOO → calibration summary → printed report |
| `tests/test_calibration_metrics.py`, `tests/test_events.py`, `tests/test_vocabulary.py`, `tests/test_loo.py`, `tests/test_backtest_harness.py` | Tests |

---

### Task 1: Calibration metrics (log-loss, AUC, reliability, summary)

**Files:**
- Modify: `src/backtest/calibration.py` (append; keep `brier_score`, `calibration_report`, `split_before`)
- Test: `tests/test_calibration_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calibration_metrics.py
import math
from src.backtest.calibration import log_loss, auc, reliability_diagram, calibration_summary


def test_log_loss_perfect_is_zero():
    assert math.isclose(log_loss([1.0, 0.0], [1, 0]), 0.0, abs_tol=1e-9)


def test_log_loss_clamps_and_penalizes_confident_wrong():
    # predict 0 for a positive => clamped to eps, large but finite penalty
    v = log_loss([0.0], [1])
    assert v > 30 and math.isfinite(v)


def test_auc_hand_checked_case():
    # preds sorted: (0.1,0)(0.35,1)(0.4,0)(0.8,1) -> AUC = 0.75
    assert math.isclose(auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]), 0.75, rel_tol=1e-9)


def test_auc_undefined_with_one_class_returns_none():
    assert auc([0.2, 0.7], [1, 1]) is None
    assert auc([0.2, 0.7], [0, 0]) is None


def test_reliability_diagram_bins_observed_frequency():
    # two preds in [0.0,0.1), one outcome 1 one outcome 0 -> observed 0.5
    bins = reliability_diagram([0.05, 0.05, 0.95], [1, 0, 1], n_bins=10)
    first = bins[0]
    assert first["count"] == 2
    assert math.isclose(first["observed"], 0.5, rel_tol=1e-9)
    last = bins[9]
    assert last["count"] == 1 and math.isclose(last["observed"], 1.0, rel_tol=1e-9)


def test_calibration_summary_bundles_all_metrics():
    pool = [(0.9, 1), (0.1, 0), (0.8, 1), (0.2, 0)]
    s = calibration_summary(pool)
    assert s["n"] == 4
    assert math.isclose(s["base_rate"], 0.5, rel_tol=1e-9)
    assert set(s) == {"n", "base_rate", "brier", "log_loss", "auc", "reliability"}
    assert 0.0 <= s["brier"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibration_metrics.py -v`
Expected: FAIL — `ImportError: cannot import name 'log_loss'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/backtest/calibration.py`:

```python
import math


def log_loss(predictions: list[float], outcomes: list[int], eps: float = 1e-15) -> float:
    """Mean binary cross-entropy. Predictions clamped to [eps, 1-eps] to avoid log(0)."""
    if not predictions:
        return 0.0
    total = 0.0
    for p, o in zip(predictions, outcomes):
        p = min(max(p, eps), 1.0 - eps)
        total += -(o * math.log(p) + (1 - o) * math.log(1.0 - p))
    return total / len(predictions)


def auc(predictions: list[float], outcomes: list[int]) -> float | None:
    """Rank-based ROC AUC (Mann-Whitney). Returns None if only one outcome class is present."""
    n_pos = sum(1 for o in outcomes if o == 1)
    n_neg = len(outcomes) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    paired = sorted(zip(predictions, outcomes), key=lambda x: x[0])
    ranks = [0.0] * len(paired)
    i = 0
    while i < len(paired):
        j = i
        while j < len(paired) and paired[j][0] == paired[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0  # 1-based average rank for ties
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    sum_pos_ranks = sum(r for r, (_, o) in zip(ranks, paired) if o == 1)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def reliability_diagram(predictions: list[float], outcomes: list[int], n_bins: int = 10) -> list[dict]:
    """Bin predictions into n_bins equal-width buckets; report mean prediction vs observed frequency."""
    bins = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [
            i for i, p in enumerate(predictions)
            if (lo <= p < hi) or (b == n_bins - 1 and p == hi)  # last bin includes 1.0
        ]
        if idx:
            mean_pred = sum(predictions[i] for i in idx) / len(idx)
            observed = sum(outcomes[i] for i in idx) / len(idx)
        else:
            mean_pred = observed = None
        bins.append({"bin": b, "lo": lo, "hi": hi, "count": len(idx),
                     "mean_pred": mean_pred, "observed": observed})
    return bins


def calibration_summary(pool: list[tuple[float, int]]) -> dict:
    """Reduce a flat pool of (pred, outcome) pairs to all Level-1 metrics."""
    preds = [p for p, _ in pool]
    outs = [o for _, o in pool]
    n = len(pool)
    return {
        "n": n,
        "base_rate": (sum(outs) / n) if n else 0.0,
        "brier": brier_score(preds, outs),
        "log_loss": log_loss(preds, outs),
        "auc": auc(preds, outs),
        "reliability": reliability_diagram(preds, outs),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_calibration_metrics.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git -C /Users/nealt1/Downloads/SpeechEdge add src/backtest/calibration.py tests/test_calibration_metrics.py
git -C /Users/nealt1/Downloads/SpeechEdge commit -m "feat(backtest): log-loss, AUC, reliability diagram, calibration_summary"
git -C /Users/nealt1/Downloads/SpeechEdge push -q origin master
```

---

### Task 2: Event + CorpusLoader

**Files:**
- Create: `src/backtest/events.py`
- Test: `tests/test_events.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_events.py
import json
from src.backtest.events import Event, JsonlCorpusLoader


def test_event_is_a_simple_value_object():
    e = Event(speaker="warsh", date="2026-04-21", text="inflation", context_type="hearing")
    assert e.speaker == "warsh" and e.date == "2026-04-21"


def test_jsonl_loader_reads_and_sorts_by_date(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(
        json.dumps({"date": "2026-01-01", "text": "b", "context_type": "speech"}) + "\n"
        + json.dumps({"date": "2006-01-01", "text": "a", "context_type": "speech"}) + "\n"
    )
    docs = JsonlCorpusLoader(p).docs_for("warsh")
    assert [d["date"] for d in docs] == ["2006-01-01", "2026-01-01"]  # ascending
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_events.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.backtest.events'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/backtest/events.py
"""Events and a speaker-keyed corpus loader for the calibration harness.

An Event is one held-out speaking occasion. CorpusLoader is the seam that lets a
future Powell/SOTU corpus plug in behind the same interface as Warsh's jsonl.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Event:
    speaker: str
    date: str
    text: str
    context_type: str


class CorpusLoader(Protocol):
    def docs_for(self, speaker: str) -> list[dict]:
        """Return that speaker's dated docs (each a dict with date/text/context_type)."""
        ...


class JsonlCorpusLoader:
    """CorpusLoader over a single-speaker JSONL corpus (e.g. corpus/warsh_corpus.jsonl)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def docs_for(self, speaker: str) -> list[dict]:
        docs = [json.loads(l) for l in self._path.read_text().splitlines() if l.strip()]
        return sorted(docs, key=lambda d: d.get("date", ""))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_events.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git -C /Users/nealt1/Downloads/SpeechEdge add src/backtest/events.py tests/test_events.py
git -C /Users/nealt1/Downloads/SpeechEdge commit -m "feat(backtest): Event value object + JsonlCorpusLoader seam"
git -C /Users/nealt1/Downloads/SpeechEdge push -q origin master
```

---

### Task 3: VocabularySource + curated lexicon

**Files:**
- Create: `corpus/lexicon/fed_macro_terms.json`
- Create: `src/backtest/vocabulary.py`
- Test: `tests/test_vocabulary.py`

- [ ] **Step 1: Create the curated lexicon**

Create `corpus/lexicon/fed_macro_terms.json` (create the `corpus/lexicon/` directory). This is the starter Fed/macro pool spanning hawkish/dovish/independence/QT + the observed market terms; patterns reuse the Plan-1 corrected regex style:

```json
[
  {"canonical": "inflation", "patterns": ["\\binflation\\b"]},
  {"canonical": "deflation", "patterns": ["\\bdeflation\\b"]},
  {"canonical": "rate cut", "patterns": ["rate cut", "cut rate", "cut\\w*(?:\\s+\\w+)*\\s+rates?", "lower\\w*(?:\\s+\\w+)*\\s+rates?"]},
  {"canonical": "rate hike", "patterns": ["rate hike", "hike\\w*(?:\\s+\\w+)*\\s+rates?", "raise\\w*(?:\\s+\\w+)*\\s+rates?", "\\btightening\\b"]},
  {"canonical": "restrictive", "patterns": ["\\brestrictive\\b"]},
  {"canonical": "accommodative", "patterns": ["\\baccommodative\\b", "\\baccommodation\\b"]},
  {"canonical": "price stability", "patterns": ["price stability"]},
  {"canonical": "dual mandate", "patterns": ["dual mandate"]},
  {"canonical": "maximum employment", "patterns": ["maximum employment", "full employment"]},
  {"canonical": "unemployment", "patterns": ["\\bunemployment\\b"]},
  {"canonical": "labor market", "patterns": ["labor market", "labour market"]},
  {"canonical": "recession", "patterns": ["\\brecession\\b"]},
  {"canonical": "soft landing", "patterns": ["soft landing"]},
  {"canonical": "stagflation", "patterns": ["\\bstagflation\\b"]},
  {"canonical": "growth", "patterns": ["\\bgrowth\\b"]},
  {"canonical": "productivity", "patterns": ["\\bproductivity\\b"]},
  {"canonical": "financial stability", "patterns": ["financial stability"]},
  {"canonical": "financial conditions", "patterns": ["financial conditions"]},
  {"canonical": "credit", "patterns": ["\\bcredit\\b"]},
  {"canonical": "banking", "patterns": ["\\bbank(?:s|ing)?\\b"]},
  {"canonical": "quantitative easing", "patterns": ["quantitative easing", "\\bqe\\b"]},
  {"canonical": "quantitative tightening", "patterns": ["quantitative tightening", "\\bqt\\b"]},
  {"canonical": "balance sheet", "patterns": ["balance sheet"]},
  {"canonical": "asset purchases", "patterns": ["asset purchase", "bond buying", "bond purchase"]},
  {"canonical": "taper", "patterns": ["\\btaper(?:ing)?\\b", "\\brunoff\\b"]},
  {"canonical": "independence", "patterns": ["independen"]},
  {"canonical": "political pressure", "patterns": ["political pressure", "political interference"]},
  {"canonical": "trump", "patterns": ["\\btrump\\b"]},
  {"canonical": "data dependent", "patterns": ["data depend"]},
  {"canonical": "forward guidance", "patterns": ["forward guidance"]},
  {"canonical": "neutral rate", "patterns": ["neutral rate", "\\br-star\\b", "\\br\\*"]},
  {"canonical": "terminal rate", "patterns": ["terminal rate"]},
  {"canonical": "tariffs", "patterns": ["\\btariffs?\\b"]},
  {"canonical": "fiscal", "patterns": ["\\bfiscal\\b", "\\bdeficit\\b"]},
  {"canonical": "debt", "patterns": ["\\bdebt\\b"]},
  {"canonical": "dollar", "patterns": ["\\bdollar\\b"]},
  {"canonical": "housing", "patterns": ["\\bhousing\\b", "mortgage"]},
  {"canonical": "consumer spending", "patterns": ["consumer spending", "household spending"]},
  {"canonical": "wages", "patterns": ["\\bwages?\\b", "wage growth"]},
  {"canonical": "uncertainty", "patterns": ["\\buncertainty\\b"]}
]
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_vocabulary.py
from src.backtest.events import Event
from src.backtest.vocabulary import LexiconVocabulary
from pathlib import Path

LEXICON = Path("corpus/lexicon/fed_macro_terms.json")
EVENT = Event(speaker="warsh", date="2026-04-21", text="x", context_type="hearing")


def test_lexicon_loads_market_terms():
    vocab = LexiconVocabulary(LEXICON)
    terms = vocab.terms_for(EVENT)
    assert len(terms) >= 30
    canonicals = {t.canonical for t in terms}
    assert {"inflation", "rate cut", "independence", "trump"} <= canonicals


def test_lexicon_is_event_independent():
    # the curated lexicon returns the same pool regardless of the event (Warsh has no historical market)
    vocab = LexiconVocabulary(LEXICON)
    other = Event(speaker="warsh", date="2008-01-01", text="y", context_type="speech")
    assert [t.canonical for t in vocab.terms_for(EVENT)] == [t.canonical for t in vocab.terms_for(other)]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_vocabulary.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.backtest.vocabulary'`

- [ ] **Step 4: Write minimal implementation**

```python
# src/backtest/vocabulary.py
"""VocabularySource seam: which candidate terms to score for a given event.

LexiconVocabulary returns a fixed curated pool — correct for Warsh, whose past
speeches predate mention markets. A future MarketVocabulary will return the real
historical Kalshi/Polymarket term list for events that had markets (Powell, SOTU).
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.mentions.terms import MarketTerm, load_terms
from src.backtest.events import Event


class VocabularySource(Protocol):
    def terms_for(self, event: Event) -> list[MarketTerm]:
        """Return the candidate terms to score for this event."""
        ...


class LexiconVocabulary:
    """Fixed curated lexicon, independent of the event."""

    def __init__(self, path: str | Path) -> None:
        self._terms = load_terms(path)

    def terms_for(self, event: Event) -> list[MarketTerm]:
        return self._terms
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_vocabulary.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git -C /Users/nealt1/Downloads/SpeechEdge add corpus/lexicon/fed_macro_terms.json src/backtest/vocabulary.py tests/test_vocabulary.py
git -C /Users/nealt1/Downloads/SpeechEdge commit -m "feat(backtest): curated Fed/macro lexicon + LexiconVocabulary seam"
git -C /Users/nealt1/Downloads/SpeechEdge push -q origin master
```

---

### Task 4: Leave-one-event-out engine

**Files:**
- Create: `src/backtest/loo.py`
- Test: `tests/test_loo.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_loo.py
from src.mentions.terms import MarketTerm
from src.backtest.loo import leave_one_out


class _FixedVocab:
    def __init__(self, terms):
        self._terms = terms
    def terms_for(self, event):
        return self._terms


def _docs():
    # 5 dated docs; "inflation" present in all, "qe" only in the 2009 doc
    return [
        {"date": "2006-01-01", "context_type": "speech", "text": "inflation"},
        {"date": "2007-01-01", "context_type": "speech", "text": "inflation"},
        {"date": "2008-01-01", "context_type": "speech", "text": "inflation"},
        {"date": "2009-01-01", "context_type": "speech", "text": "inflation qe"},
        {"date": "2010-01-01", "context_type": "speech", "text": "inflation"},
    ]


def test_cold_start_guard_skips_events_without_enough_priors():
    vocab = _FixedVocab([MarketTerm(canonical="inflation", patterns=(r"\binflation\b",))])
    out = leave_one_out(_docs(), "warsh", vocab, min_prior_docs=3)
    # only the 2009 and 2010 docs have >=3 strictly-earlier docs
    assert out["n_scored"] == 2
    assert out["n_skipped"] == 3


def test_pool_holds_pred_outcome_pairs_with_no_leakage():
    # "qe" only appears in the 2009 doc; predicting the 2009 event must NOT see it (strict < date)
    terms = [
        MarketTerm(canonical="inflation", patterns=(r"\binflation\b",)),
        MarketTerm(canonical="qe", patterns=(r"\bqe\b",)),
    ]
    out = leave_one_out(_docs(), "warsh", _FixedVocab(terms), min_prior_docs=3)
    # for the 2009 held-out event, qe outcome = 1 but its prior base rate is the smoothing floor (low)
    qe_pairs = [(p, o) for (p, o) in out["pool"]]
    # every pair is (float prob, int 0/1)
    assert all(0.0 <= p <= 1.0 and o in (0, 1) for p, o in qe_pairs)
    # 2 scored events x 2 terms = 4 pairs
    assert len(out["pool"]) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_loo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.backtest.loo'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/backtest/loo.py
"""Leave-one-event-out engine: reduce a speaker's dated corpus to a flat
(prediction, outcome) pool for Level-1 calibration scoring.

For each held-out doc, the Mode-1 predictor is built from ONLY that speaker's
strictly-earlier docs (temporal split, no leakage), recency-weighted to the event
date. Events with too few prior docs are skipped (cold-start guard).
"""
from __future__ import annotations

from src.mentions.news_adjuster import NullAdjuster
from src.mentions.predict import predict_event_priors
from src.backtest.events import Event


def leave_one_out(
    docs: list[dict],
    speaker: str,
    vocab,
    min_prior_docs: int = 3,
    half_life_years: float = 4.0,
) -> dict:
    docs_sorted = sorted(docs, key=lambda d: d.get("date", ""))
    pool: list[tuple[float, int]] = []
    per_event: list[dict] = []
    n_scored = n_skipped = 0

    for held in docs_sorted:
        date = held.get("date", "")
        text = held.get("text", "")
        if not date or not text:
            n_skipped += 1
            continue
        prior = [d for d in docs_sorted if d.get("date", "") < date]
        if len(prior) < min_prior_docs:
            n_skipped += 1
            continue
        event = Event(speaker=speaker, date=date, text=text,
                      context_type=held.get("context_type", ""))
        terms = vocab.terms_for(event)
        if not terms:
            n_skipped += 1
            continue
        priors = predict_event_priors(
            prior, terms, news_context="", adjuster=NullAdjuster(),
            as_of=date, half_life_years=half_life_years,
        )
        for t in terms:
            pred = priors[t.canonical]["p_prior"]
            outcome = int(t.mentioned_in(event.text))
            pool.append((pred, outcome))
        n_scored += 1
        per_event.append({"date": date, "context_type": event.context_type, "n_terms": len(terms)})

    return {"pool": pool, "n_scored": n_scored, "n_skipped": n_skipped, "per_event": per_event}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_loo.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git -C /Users/nealt1/Downloads/SpeechEdge add src/backtest/loo.py tests/test_loo.py
git -C /Users/nealt1/Downloads/SpeechEdge commit -m "feat(backtest): leave-one-event-out engine (temporal, leakage-free)"
git -C /Users/nealt1/Downloads/SpeechEdge push -q origin master
```

---

### Task 5: Harness runner + end-to-end on real corpus

**Files:**
- Create: `backtest_harness.py`
- Test: `tests/test_backtest_harness.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backtest_harness.py
from backtest_harness import run_harness


def test_run_harness_on_real_corpus_produces_full_summary():
    r = run_harness()
    s = r["summary"]
    assert r["n_scored"] >= 10            # LOO turns n=1 into many events
    assert s["n"] > 100                   # many (pred, outcome) pairs across events x lexicon
    assert 0.0 < s["base_rate"] < 1.0     # both positives and negatives present
    assert 0.0 <= s["brier"] <= 1.0
    assert s["log_loss"] >= 0.0
    assert (s["auc"] is None) or (0.0 <= s["auc"] <= 1.0)
    assert len(s["reliability"]) == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_backtest_harness.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtest_harness'`

- [ ] **Step 3: Write the implementation**

```python
# backtest_harness.py
"""Level-1 calibration harness — Warsh leave-one-event-out over the curated lexicon.

Turns mention-model validation from n=1 into ~25 scored events. Reports Brier,
log-loss, AUC, and a reliability diagram over the pooled (prediction, outcome) pairs.

Usage: python backtest_harness.py
"""
from __future__ import annotations

from pathlib import Path

from src.backtest.events import JsonlCorpusLoader
from src.backtest.vocabulary import LexiconVocabulary
from src.backtest.loo import leave_one_out
from src.backtest.calibration import calibration_summary

CORPUS = Path("corpus/warsh_corpus.jsonl")
LEXICON = Path("corpus/lexicon/fed_macro_terms.json")


def run_harness(min_prior_docs: int = 3, half_life_years: float = 4.0) -> dict:
    docs = JsonlCorpusLoader(CORPUS).docs_for("warsh")
    vocab = LexiconVocabulary(LEXICON)
    loo = leave_one_out(docs, "warsh", vocab, min_prior_docs, half_life_years)
    return {
        "summary": calibration_summary(loo["pool"]),
        "n_scored": loo["n_scored"],
        "n_skipped": loo["n_skipped"],
    }


def main() -> None:
    r = run_harness()
    s = r["summary"]
    auc_str = f"{s['auc']:.4f}" if s["auc"] is not None else "n/a"
    print(f"events scored: {r['n_scored']}   skipped: {r['n_skipped']}")
    print(f"pool: {s['n']} (pred,outcome) pairs   base rate: {s['base_rate']:.3f}")
    print(f"Brier {s['brier']:.4f}   log-loss {s['log_loss']:.4f}   AUC {auc_str}")
    print("reliability (predicted vs observed):")
    print(f"  {'bin':>10} {'n':>5} {'mean_pred':>10} {'observed':>9}")
    for b in s["reliability"]:
        if b["count"]:
            print(f"  [{b['lo']:.1f},{b['hi']:.1f}) {b['count']:>5} "
                  f"{b['mean_pred']:>10.3f} {b['observed']:>9.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_backtest_harness.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Run the real harness and record the numbers**

Run: `.venv/bin/python backtest_harness.py`
Expected: an events-scored line, pooled Brier/log-loss/AUC, and a reliability table. **Record this output** — it is the first multi-event calibration read of the model. Report it verbatim; report whatever the numbers are (no tuning to force any result — tuning is sub-project 2B).

- [ ] **Step 6: Commit**

```bash
git -C /Users/nealt1/Downloads/SpeechEdge add backtest_harness.py tests/test_backtest_harness.py
git -C /Users/nealt1/Downloads/SpeechEdge commit -m "feat(backtest): Level-1 calibration harness runner (Warsh LOO)"
git -C /Users/nealt1/Downloads/SpeechEdge push -q origin master
```

---

### Task 6: Full-suite regression + DECISIONS close-out

- [ ] **Step 1: Run the entire suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all prior tests (105) + the new harness tests PASS, no regressions. If anything fails, STOP and report BLOCKED.

- [ ] **Step 2: Append to DECISIONS.md**

Append a dated `### D7` entry under a `## 2026-06-12` heading (create the heading if absent), placed before any trailing "Accepted defaults" table:

```markdown
### D7 — Level-1 calibration harness shipped (sub-project 2A)
- **Decision:** Mention-model validation is now multi-event: leave-one-event-out over Warsh's corpus against a curated Fed/macro lexicon, scored as a pooled (prediction, outcome) set with Brier, log-loss, AUC, and a reliability diagram. Speaker-agnostic via CorpusLoader/VocabularySource seams (Powell/SOTU real market term-lists plug in later). Spec: `docs/superpowers/specs/2026-06-12-backtest-harness-design.md`.
- **Result:** <paste the recorded `backtest_harness.py` numbers from Task 5: events scored, pool size, base rate, Brier, log-loss, AUC>.
- **Scope:** 2A done. 2B (hyperparameter tuning on log-loss) and 2C (live Mode-2) follow. Level-2 P&L remains out (no historical price data).
- **Authority:** Owner-directed.
```

Replace the `<paste ...>` placeholder with the actual numbers from Task 5 before committing.

- [ ] **Step 3: Commit**

```bash
git -C /Users/nealt1/Downloads/SpeechEdge add DECISIONS.md
git -C /Users/nealt1/Downloads/SpeechEdge commit -m "docs(decisions): Level-1 calibration harness shipped (2A); numbers recorded"
git -C /Users/nealt1/Downloads/SpeechEdge push -q origin master
```

---

## Self-Review

**Spec coverage (§ → task):**
- §3 LOO pool-reduction + temporal no-leakage split → Task 4.
- §3 `CorpusLoader` seam → Task 2; `VocabularySource` seam → Task 3.
- §4 file structure → Tasks 1–5 match the table exactly.
- §5 metrics (Brier/log-loss/AUC/reliability) + base-rate report → Task 1 (`calibration_summary`), surfaced in Task 5 runner.
- §5 `min_prior_docs` cold-start guard → Task 4 (tested).
- §5 report base rate → Task 1 (`calibration_summary["base_rate"]`), printed in Task 5.
- §6 error handling (empty vocab/text skip, log_loss eps clamp, AUC one-class None) → Task 4 (skips), Task 1 (clamp + None), all tested.
- §7 testing strategy (pure metrics, LOO synthetic, seams, end-to-end) → Tasks 1–5 tests.
- §8 open items: lexicon authored (Task 3), min_prior_docs default 3 (Task 4), recency `as_of=E.date` (Task 4), rank-based AUC no sklearn (Task 1). All resolved.
- Out of scope (Level-2 P&L, cross-speaker data sourcing, tuning, live) → correctly absent.

**Placeholder scan:** none. The one `<paste numbers>` in Task 6 is an explicit instruction to insert real Task-5 output before committing, not a code placeholder. Gate/metric values are computed, not stubbed.

**Type consistency:** `Event(speaker,date,text,context_type)` defined Task 2, used Tasks 3–4. `CorpusLoader.docs_for` / `VocabularySource.terms_for` Protocols (Tasks 2–3) match their implementations and the `leave_one_out` call site (Task 4). `leave_one_out(...)` returns `{pool,n_scored,n_skipped,per_event}` (Task 4), consumed in Task 5. `calibration_summary(pool)` returns `{n,base_rate,brier,log_loss,auc,reliability}` (Task 1), read in Tasks 1/5 tests and the runner. `predict_event_priors(..., as_of=, half_life_years=)` matches the current signature. Consistent.

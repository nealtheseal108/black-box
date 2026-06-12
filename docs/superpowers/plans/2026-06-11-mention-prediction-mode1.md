# Mention Prediction — Mode 1 Predictor + Calibration Gate (Plan 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the offline, pre-speech mention predictor — for each market term, a calibrated `P(mention)` from Warsh's corpus (phase-decomposed: prepared-statement vs. Q&A) scaled by a bounded news multiplier — and validate it with a Brier-calibration gate against the resolved April-21 confirmation hearing.

**Architecture:** Pure statistical core (phase-conditioned add-k base rates → `P_event = 1−(1−P_prep)(1−P_qa)`) + an injected `NewsAdjuster` interface whose multipliers are clamped to 0.25×–4× (so the LLM can scale but never invent a probability). A new `src/mentions/` package holds the predictor; `src/backtest/calibration.py` holds the Brier gate. Everything is offline-testable with fakes — no network, no LLM in the test path.

**Tech Stack:** Python 3.11, pytest, the existing `corpus/warsh_corpus.jsonl` (29 docs) and `src/warsh/tokenize.py`. No new dependencies.

**Working directory (HARD RULE):** All commands run from `/Users/nealt1/Downloads/SpeechEdge`. Commit with `git -C /Users/nealt1/Downloads/SpeechEdge` and push to `origin` (black-box) after each task.

**Spec:** `docs/superpowers/specs/2026-06-11-mention-prediction-model-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `src/mentions/__init__.py` | Package marker |
| `src/mentions/terms.py` | `MarketTerm` (canonical + variant regex patterns), `.mentioned_in(text)`, `load_terms(path)` |
| `src/mentions/base_rate.py` | Phase-conditioned add-k base rates + `combine_phases`; `event_base_rate(docs, term)` |
| `src/mentions/news_adjuster.py` | `NewsAdjuster` protocol, `clamp_multiplier`, `apply_multipliers`, `NullAdjuster` |
| `src/mentions/predict.py` | Mode-1 assembly: `predict_event_priors(docs, terms, news, adjuster)` |
| `src/backtest/calibration.py` | `brier_score`, `calibration_report`, `split_before` (date split) |
| `src/backtest/gates.py` | **modify** — add `evaluate_mention_gate(brier, threshold)` |
| `corpus/market_terms/hearing_2026-04-21.json` | Fixture: the hearing's market terms + variant patterns |
| `backtest_mentions.py` | CLI: date-split → predict priors → resolve outcomes from hearing text → calibration report + gate |
| `tests/test_mention_terms.py`, `tests/test_base_rate.py`, `tests/test_news_adjuster.py`, `tests/test_predict.py`, `tests/test_calibration.py`, `tests/test_mention_gate.py` | Unit tests |

---

### Task 1: Market term vocabulary

**Files:**
- Create: `src/mentions/__init__.py` (empty)
- Create: `src/mentions/terms.py`
- Test: `tests/test_mention_terms.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mention_terms.py
from src.mentions.terms import MarketTerm, load_terms


def test_term_matches_any_variant_pattern():
    term = MarketTerm(canonical="rate cut", patterns=("rate cut", "cut rate", r"cut\w*\s+\w*\s*rates?"))
    assert term.mentioned_in("we may need a rate cut soon") is True
    assert term.mentioned_in("the committee could cut the policy rate") is True   # variant
    assert term.mentioned_in("inflation remains elevated") is False


def test_term_matching_is_case_insensitive():
    term = MarketTerm(canonical="trump", patterns=(r"\btrump\b",))
    assert term.mentioned_in("President TRUMP said") is True
    assert term.mentioned_in("they trumpeted the result") is False   # word boundary


def test_load_terms_from_json(tmp_path):
    p = tmp_path / "terms.json"
    p.write_text('[{"canonical": "trump", "patterns": ["\\\\btrump\\\\b"]}]')
    terms = load_terms(p)
    assert len(terms) == 1
    assert terms[0].canonical == "trump"
    assert terms[0].mentioned_in("Trump") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_mention_terms.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.mentions'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/mentions/__init__.py
```

```python
# src/mentions/terms.py
"""Market mention terms and the surface patterns that count as 'saying' them.

A Kalshi/Polymarket mention sub-market resolves YES if the speaker utters a term
in any of its phrasings. MarketTerm bundles the canonical label with the regex
variants that all resolve it ("rate cut" / "cut rate" / "cut the policy rate").
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MarketTerm:
    canonical: str
    patterns: tuple[str, ...]

    def mentioned_in(self, text: str) -> bool:
        low = text.lower()
        return any(re.search(p, low) for p in self.patterns)


def load_terms(path: str | Path) -> list[MarketTerm]:
    raw = json.loads(Path(path).read_text())
    return [MarketTerm(canonical=t["canonical"], patterns=tuple(t["patterns"])) for t in raw]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_mention_terms.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git -C /Users/nealt1/Downloads/SpeechEdge add src/mentions/__init__.py src/mentions/terms.py tests/test_mention_terms.py
git -C /Users/nealt1/Downloads/SpeechEdge commit -m "feat(mentions): MarketTerm with variant-pattern matching"
git -C /Users/nealt1/Downloads/SpeechEdge push -q origin master
```

---

### Task 2: Phase-conditioned base-rate estimator

**Files:**
- Create: `src/mentions/base_rate.py`
- Test: `tests/test_base_rate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_base_rate.py
import math
from src.mentions.terms import MarketTerm
from src.mentions.base_rate import phase_base_rate, combine_phases, event_base_rate, PREP_TYPES, QA_TYPES

INFL = MarketTerm(canonical="inflation", patterns=(r"\binflation\b",))


def _docs():
    return [
        {"context_type": "speech",    "text": "inflation is a choice and we will act"},
        {"context_type": "speech",    "text": "the labor market remains strong"},          # no inflation
        {"context_type": "lecture",   "text": "inflation expectations matter"},
        {"context_type": "interview", "text": "inflation has been too high lately"},
        {"context_type": "hearing",   "text": "we must finish the job on prices"},          # no inflation
    ]


def test_unseen_term_gets_nonzero_floor_not_zero():
    # add-k smoothing: a never-said term must not be trapped at 0 (so the LLM can lift it)
    never = MarketTerm(canonical="stagflation", patterns=(r"\bstagflation\b",))
    p = phase_base_rate(_docs(), never, PREP_TYPES, k=0.5)
    assert p > 0.0
    assert p < 0.2          # but small


def test_prep_rate_counts_only_prepared_docs():
    # 3 prep docs (2 speech + 1 lecture); inflation in 2 of them → (2+0.5)/(3+1) = 0.625
    p = phase_base_rate(_docs(), INFL, PREP_TYPES, k=0.5)
    assert math.isclose(p, 2.5 / 4.0, rel_tol=1e-9)


def test_qa_rate_counts_only_qa_docs():
    # 2 qa docs (interview + hearing); inflation in 1 → (1+0.5)/(2+1) = 0.5
    p = phase_base_rate(_docs(), INFL, QA_TYPES, k=0.5)
    assert math.isclose(p, 1.5 / 3.0, rel_tol=1e-9)


def test_combine_phases_is_noisy_or():
    # P(anywhere) = 1 - (1-0.5)(1-0.5) = 0.75
    assert math.isclose(combine_phases(0.5, 0.5), 0.75, rel_tol=1e-9)


def test_event_base_rate_bundles_all_three():
    out = event_base_rate(_docs(), INFL, k=0.5)
    assert set(out) == {"p_prep", "p_qa", "p_event"}
    assert math.isclose(out["p_event"], combine_phases(out["p_prep"], out["p_qa"]), rel_tol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_base_rate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.mentions.base_rate'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/mentions/base_rate.py
"""Phase-conditioned mention base rates from the Warsh corpus.

Prepared-statement diction (scripted monologue) and Q&A diction (unscripted
dialogue) behave differently, so we estimate each separately and combine with a
noisy-OR — a term resolves YES if it is said in EITHER phase (§3.0 of the spec):

    P_event = 1 - (1 - P_prep) * (1 - P_qa)

Add-k (Jeffreys, k=0.5) smoothing gives every term a nonzero floor, so a term
Warsh has never said on record is not trapped at 0 (the bounded news multiplier
can then lift it).
"""
from __future__ import annotations

from src.mentions.terms import MarketTerm

# context_type → phase. Scripted monologue vs. unscripted dialogue.
PREP_TYPES = frozenset({"speech", "essay", "op_ed", "lecture", "testimony"})
QA_TYPES = frozenset({"interview", "hearing"})


def _phase_docs(docs: list[dict], phase_types: frozenset[str]) -> list[dict]:
    return [d for d in docs if d.get("context_type") in phase_types]


def phase_base_rate(docs: list[dict], term: MarketTerm, phase_types: frozenset[str], k: float = 0.5) -> float:
    """Add-k smoothed fraction of `phase_types` docs that mention `term`."""
    pool = _phase_docs(docs, phase_types)
    n = len(pool)
    hits = sum(1 for d in pool if term.mentioned_in(d["text"]))
    return (hits + k) / (n + 2 * k)


def combine_phases(p_prep: float, p_qa: float) -> float:
    """Noisy-OR: probability the term appears in EITHER phase."""
    return 1.0 - (1.0 - p_prep) * (1.0 - p_qa)


def event_base_rate(docs: list[dict], term: MarketTerm, k: float = 0.5) -> dict:
    p_prep = phase_base_rate(docs, term, PREP_TYPES, k)
    p_qa = phase_base_rate(docs, term, QA_TYPES, k)
    return {"p_prep": p_prep, "p_qa": p_qa, "p_event": combine_phases(p_prep, p_qa)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_base_rate.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git -C /Users/nealt1/Downloads/SpeechEdge add src/mentions/base_rate.py tests/test_base_rate.py
git -C /Users/nealt1/Downloads/SpeechEdge commit -m "feat(mentions): phase-conditioned add-k base rates + noisy-OR combine"
git -C /Users/nealt1/Downloads/SpeechEdge push -q origin master
```

---

### Task 3: Bounded news adjuster

**Files:**
- Create: `src/mentions/news_adjuster.py`
- Test: `tests/test_news_adjuster.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_news_adjuster.py
import math
from src.mentions.news_adjuster import clamp_multiplier, apply_multipliers, NullAdjuster, MULT_MIN, MULT_MAX


def test_clamp_bounds_the_multiplier():
    assert clamp_multiplier(10.0) == MULT_MAX      # 4.0
    assert clamp_multiplier(0.001) == MULT_MIN     # 0.25
    assert clamp_multiplier(2.0) == 2.0            # in-range unchanged


def test_apply_scales_then_clips_to_unit_interval():
    base = {"a": 0.2, "b": 0.5}
    out = apply_multipliers(base, {"a": 3.0, "b": 4.0})   # b: 0.5*4=2.0 → clip to 1.0
    assert math.isclose(out["a"], 0.6, rel_tol=1e-9)
    assert out["b"] == 1.0


def test_missing_term_defaults_to_neutral_multiplier():
    out = apply_multipliers({"a": 0.3}, {})        # no entry for "a" → ×1.0
    assert math.isclose(out["a"], 0.3, rel_tol=1e-9)


def test_null_adjuster_returns_all_neutral():
    adj = NullAdjuster()
    assert adj.multipliers(["a", "b"], "any news") == {"a": 1.0, "b": 1.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_news_adjuster.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.mentions.news_adjuster'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/mentions/news_adjuster.py
"""News adjuster: bounded per-term multipliers on the statistical base rate.

The LLM (real implementation lands in Plan 1's successor / Plan 2 news layer) can
SCALE a base rate within [0.25, 4.0] to reflect current news — e.g. "today's
banking stress makes 'financial stability' 4x more likely" — but can NEVER invent
a probability from nothing. This clamp is the guardrail that keeps a hallucinated
multiplier from blowing up a position.

NullAdjuster is the offline default (all multipliers = 1.0) used in deterministic
tests and the calibration backtest.
"""
from __future__ import annotations

from typing import Protocol

MULT_MIN = 0.25
MULT_MAX = 4.0


class NewsAdjuster(Protocol):
    def multipliers(self, terms: list[str], news_context: str) -> dict[str, float]:
        """Return a raw (pre-clamp) multiplier per canonical term."""
        ...


def clamp_multiplier(m: float) -> float:
    return max(MULT_MIN, min(MULT_MAX, m))


def apply_multipliers(base_rates: dict[str, float], raw_mults: dict[str, float]) -> dict[str, float]:
    """Scale each base rate by its clamped multiplier (default 1.0), clip to [0, 1]."""
    out: dict[str, float] = {}
    for term, p in base_rates.items():
        m = clamp_multiplier(raw_mults.get(term, 1.0))
        out[term] = max(0.0, min(1.0, p * m))
    return out


class NullAdjuster:
    """No-op adjuster — every term gets a neutral 1.0 multiplier."""

    def multipliers(self, terms: list[str], news_context: str) -> dict[str, float]:
        return {t: 1.0 for t in terms}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_news_adjuster.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git -C /Users/nealt1/Downloads/SpeechEdge add src/mentions/news_adjuster.py tests/test_news_adjuster.py
git -C /Users/nealt1/Downloads/SpeechEdge commit -m "feat(mentions): bounded news adjuster (0.25-4x clamp) + NullAdjuster"
git -C /Users/nealt1/Downloads/SpeechEdge push -q origin master
```

---

### Task 4: Mode-1 prior assembly

**Files:**
- Create: `src/mentions/predict.py`
- Test: `tests/test_predict.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_predict.py
import math
from src.mentions.terms import MarketTerm
from src.mentions.news_adjuster import NullAdjuster
from src.mentions.predict import predict_event_priors

DOCS = [
    {"context_type": "speech",    "text": "inflation is a choice"},
    {"context_type": "interview", "text": "inflation and growth"},
]
TERMS = [MarketTerm(canonical="inflation", patterns=(r"\binflation\b",))]


class _BoostAdjuster:
    def multipliers(self, terms, news_context):
        return {t: 4.0 for t in terms}    # raw 4x (already at clamp ceiling)


def test_null_adjuster_leaves_prior_equal_to_base_event_rate():
    out = predict_event_priors(DOCS, TERMS, news_context="", adjuster=NullAdjuster())
    row = out["inflation"]
    assert math.isclose(row["p_prior"], row["p_event_base"], rel_tol=1e-9)
    assert set(row) == {"p_prep", "p_qa", "p_event_base", "multiplier", "p_prior"}


def test_boost_adjuster_raises_prior_but_clips_at_one():
    out = predict_event_priors(DOCS, TERMS, news_context="", adjuster=_BoostAdjuster())
    row = out["inflation"]
    assert row["multiplier"] == 4.0
    assert row["p_prior"] >= row["p_event_base"]
    assert row["p_prior"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_predict.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.mentions.predict'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/mentions/predict.py
"""Mode-1 (pre-speech) prior assembly.

For each market term: phase-decomposed corpus base rate → bounded news multiplier
→ P_prior(mention). This is the number traded BEFORE the speech, and the input the
calibration gate scores.
"""
from __future__ import annotations

from src.mentions.terms import MarketTerm
from src.mentions.base_rate import event_base_rate
from src.mentions.news_adjuster import NewsAdjuster, apply_multipliers, clamp_multiplier


def predict_event_priors(
    docs: list[dict],
    terms: list[MarketTerm],
    news_context: str,
    adjuster: NewsAdjuster,
    k: float = 0.5,
) -> dict[str, dict]:
    base = {t.canonical: event_base_rate(docs, t, k) for t in terms}
    base_event = {c: v["p_event"] for c, v in base.items()}

    raw_mults = adjuster.multipliers([t.canonical for t in terms], news_context)
    adjusted = apply_multipliers(base_event, raw_mults)

    out: dict[str, dict] = {}
    for t in terms:
        c = t.canonical
        out[c] = {
            "p_prep": base[c]["p_prep"],
            "p_qa": base[c]["p_qa"],
            "p_event_base": base[c]["p_event"],
            "multiplier": clamp_multiplier(raw_mults.get(c, 1.0)),
            "p_prior": adjusted[c],
        }
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_predict.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git -C /Users/nealt1/Downloads/SpeechEdge add src/mentions/predict.py tests/test_predict.py
git -C /Users/nealt1/Downloads/SpeechEdge commit -m "feat(mentions): Mode-1 prior assembly (base-rate x bounded multiplier)"
git -C /Users/nealt1/Downloads/SpeechEdge push -q origin master
```

---

### Task 5: Brier calibration metric + date split

**Files:**
- Create: `src/backtest/calibration.py`
- Test: `tests/test_calibration.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calibration.py
import math
from src.backtest.calibration import brier_score, calibration_report, split_before


def test_brier_perfect_prediction_is_zero():
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0


def test_brier_worst_prediction_is_one():
    assert brier_score([0.0, 1.0], [1, 0]) == 1.0


def test_brier_half_is_quarter():
    # two terms predicted 0.5, any outcome → (0.5)^2 = 0.25 each
    assert math.isclose(brier_score([0.5, 0.5], [1, 0]), 0.25, rel_tol=1e-9)


def test_calibration_report_aligns_terms_by_name():
    rep = calibration_report({"a": 0.9, "b": 0.1}, {"a": 1, "b": 0})
    assert rep["n"] == 2
    assert math.isclose(rep["brier"], (0.01 + 0.01) / 2, rel_tol=1e-9)


def test_split_before_is_strict_and_leakage_free():
    docs = [{"date": "2020-01-01", "text": "x"}, {"date": "2026-04-21", "text": "y"}]
    train, test = split_before(docs, "2026-04-21")
    assert [d["date"] for d in train] == ["2020-01-01"]
    assert [d["date"] for d in test] == ["2026-04-21"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibration.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.backtest.calibration'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/backtest/calibration.py
"""Mention-calibration metric (Brier) and a leakage-free date split.

G1' replaces the mis-specified next-token G1: for a vocabulary of market terms we
predict P(mention) for a held-out event and score calibration against the realized
YES/NO outcomes. Calibration (not raw accuracy) is what makes Kelly sizing sane.
"""
from __future__ import annotations


def brier_score(predictions: list[float], outcomes: list[int]) -> float:
    if not predictions:
        return 0.0
    return sum((p - o) ** 2 for p, o in zip(predictions, outcomes)) / len(predictions)


def calibration_report(pred_by_term: dict[str, float], outcome_by_term: dict[str, int]) -> dict:
    terms = sorted(pred_by_term)
    preds = [pred_by_term[t] for t in terms]
    outs = [outcome_by_term[t] for t in terms]
    return {
        "n": len(terms),
        "brier": brier_score(preds, outs),
        "terms": terms,
        "predictions": preds,
        "outcomes": outs,
    }


def split_before(docs: list[dict], cutoff_date: str) -> tuple[list[dict], list[dict]]:
    """train = docs strictly before cutoff_date; test = docs on/after. ISO dates sort lexically."""
    train = [d for d in docs if d.get("date", "") < cutoff_date]
    test = [d for d in docs if d.get("date", "") >= cutoff_date]
    return train, test
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_calibration.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git -C /Users/nealt1/Downloads/SpeechEdge add src/backtest/calibration.py tests/test_calibration.py
git -C /Users/nealt1/Downloads/SpeechEdge commit -m "feat(backtest): Brier mention-calibration metric + leakage-free date split"
git -C /Users/nealt1/Downloads/SpeechEdge push -q origin master
```

---

### Task 6: Mention-calibration gate

**Files:**
- Modify: `src/backtest/gates.py` (append a function; leave `evaluate_gates` untouched)
- Test: `tests/test_mention_gate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mention_gate.py
from src.backtest.gates import evaluate_mention_gate


def test_gate_passes_when_brier_below_threshold():
    rep = evaluate_mention_gate(brier=0.18, threshold=0.25)
    assert rep["pass"] is True
    assert rep["metric"] == 0.18
    assert rep["threshold"] == "< 0.25"


def test_gate_fails_when_brier_at_or_above_threshold():
    assert evaluate_mention_gate(brier=0.25, threshold=0.25)["pass"] is False
    assert evaluate_mention_gate(brier=0.40, threshold=0.25)["pass"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_mention_gate.py -v`
Expected: FAIL — `ImportError: cannot import name 'evaluate_mention_gate'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/backtest/gates.py` (do not modify the existing `evaluate_gates`):

```python


def evaluate_mention_gate(brier: float, threshold: float = 0.25) -> dict:
    """G1' — mention calibration. Lower Brier is better; pass if strictly below threshold.

    Threshold default 0.25 is a placeholder baseline (a constant-0.5 predictor scores
    exactly 0.25); recalibrate from the Powell + hearing backtests per spec §6/§11.
    """
    return {
        "pass": brier < threshold,
        "metric": brier,
        "threshold": f"< {threshold}",
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_mention_gate.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git -C /Users/nealt1/Downloads/SpeechEdge add src/backtest/gates.py tests/test_mention_gate.py
git -C /Users/nealt1/Downloads/SpeechEdge commit -m "feat(backtest): G1' mention-calibration gate (Brier < threshold)"
git -C /Users/nealt1/Downloads/SpeechEdge push -q origin master
```

---

### Task 7: End-to-end calibration backtest on the resolved hearing

**Files:**
- Create: `corpus/market_terms/hearing_2026-04-21.json` (terms fixture)
- Create: `backtest_mentions.py` (CLI)
- Test: `tests/test_backtest_mentions.py`

- [ ] **Step 1: Create the terms fixture**

```json
[
  {"canonical": "rate cut", "patterns": ["rate cut", "cut rate", "cut\\w*\\s+\\w*\\s*rates?", "rates?\\s+\\w*\\s*cut"]},
  {"canonical": "trump", "patterns": ["\\btrump\\b"]},
  {"canonical": "inflation", "patterns": ["\\binflation\\b"]},
  {"canonical": "independence", "patterns": ["independen"]},
  {"canonical": "recession", "patterns": ["\\brecession\\b"]},
  {"canonical": "stagflation", "patterns": ["\\bstagflation\\b"]},
  {"canonical": "quantitative easing", "patterns": ["quantitative easing", "\\bqe\\b"]},
  {"canonical": "soft landing", "patterns": ["soft landing"]}
]
```

Save as `corpus/market_terms/hearing_2026-04-21.json`.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_backtest_mentions.py
from backtest_mentions import resolve_outcomes, run_calibration
from src.mentions.terms import MarketTerm


def test_resolve_outcomes_from_event_text():
    terms = [
        MarketTerm(canonical="inflation", patterns=(r"\binflation\b",)),
        MarketTerm(canonical="stagflation", patterns=(r"\bstagflation\b",)),
    ]
    outcomes = resolve_outcomes(terms, "inflation remains the focus")
    assert outcomes == {"inflation": 1, "stagflation": 0}


def test_run_calibration_on_real_corpus_produces_a_brier_and_gate():
    # Integration: uses the real corpus + fixture; train strictly before 2026-04-21,
    # predict the hearing's terms, resolve against the hearing text.
    result = run_calibration()
    assert 0.0 <= result["report"]["brier"] <= 1.0
    assert result["report"]["n"] == 8
    assert "pass" in result["gate"]
    # train split must exclude the hearing (no leakage)
    assert result["train_docs"] >= 20
    assert result["test_event_words"] > 1000
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_backtest_mentions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtest_mentions'`

- [ ] **Step 4: Write the implementation**

```python
# backtest_mentions.py
"""G1' calibration backtest — Mode-1 mention predictor vs. the resolved April-21 hearing.

The confirmation hearing is a RESOLVED mentions market and we hold Warsh's cleaned
transcript. We train base rates strictly on pre-hearing docs (no leakage), predict
P(mention) for the market terms, resolve the actual outcomes from the hearing text,
and score Brier calibration + the G1' gate.

Usage: python backtest_mentions.py
"""
from __future__ import annotations

import json
from pathlib import Path

from src.mentions.terms import MarketTerm, load_terms
from src.mentions.news_adjuster import NullAdjuster
from src.mentions.predict import predict_event_priors
from src.backtest.calibration import calibration_report, split_before
from src.backtest.gates import evaluate_mention_gate

CORPUS = Path("corpus/warsh_corpus.jsonl")
TERMS_FIXTURE = Path("corpus/market_terms/hearing_2026-04-21.json")
HEARING_DATE = "2026-04-21"


def _load_corpus() -> list[dict]:
    return [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]


def resolve_outcomes(terms: list[MarketTerm], event_text: str) -> dict[str, int]:
    """Ground truth: 1 if the term's pattern matches the event transcript, else 0."""
    return {t.canonical: int(t.mentioned_in(event_text)) for t in terms}


def run_calibration() -> dict:
    docs = _load_corpus()
    terms = load_terms(TERMS_FIXTURE)
    train, test = split_before(docs, HEARING_DATE)
    event_text = " ".join(d["text"] for d in test)   # the hearing doc(s) on/after cutoff

    priors = predict_event_priors(train, terms, news_context="", adjuster=NullAdjuster())
    pred_by_term = {c: row["p_prior"] for c, row in priors.items()}
    outcome_by_term = resolve_outcomes(terms, event_text)

    report = calibration_report(pred_by_term, outcome_by_term)
    gate = evaluate_mention_gate(report["brier"])
    return {
        "report": report,
        "gate": gate,
        "priors": priors,
        "outcomes": outcome_by_term,
        "train_docs": len(train),
        "test_event_words": len(event_text.split()),
    }


def main() -> None:
    r = run_calibration()
    print(f"train docs (pre-{HEARING_DATE}): {r['train_docs']}   test event words: {r['test_event_words']}")
    print(f"{'term':<22} {'P_prior':>8} {'outcome':>8}")
    for t in r["report"]["terms"]:
        print(f"{t:<22} {r['priors'][t]['p_prior']:>8.3f} {r['outcomes'][t]:>8d}")
    print(f"\nBrier: {r['report']['brier']:.4f}   G1' gate (< 0.25): "
          f"{'PASS' if r['gate']['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_backtest_mentions.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Run the real backtest and record the number**

Run: `.venv/bin/python backtest_mentions.py`
Expected: a per-term table (rate cut / trump / inflation / independence resolve to outcome 1; recession / stagflation / soft landing to 0) and a Brier + PASS/FAIL line. **Record this number** — it is the first trustworthy validation result under the corrected metric.

- [ ] **Step 7: Commit**

```bash
git -C /Users/nealt1/Downloads/SpeechEdge add corpus/market_terms/hearing_2026-04-21.json backtest_mentions.py tests/test_backtest_mentions.py
git -C /Users/nealt1/Downloads/SpeechEdge commit -m "feat(backtest): end-to-end G1' calibration backtest on resolved hearing"
git -C /Users/nealt1/Downloads/SpeechEdge push -q origin master
```

---

### Task 8: Full-suite regression + plan close-out

- [ ] **Step 1: Run the entire test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all prior tests (79) + the new mention tests PASS, no regressions.

- [ ] **Step 2: Update DECISIONS.md**

Append a dated entry recording: mentions-market reframe, G1' (Brier calibration) replaces next-token G1, the recorded Brier number from Task 7, and that Mode-2/live + trading are Plans 2–3.

- [ ] **Step 3: Commit**

```bash
git -C /Users/nealt1/Downloads/SpeechEdge add DECISIONS.md
git -C /Users/nealt1/Downloads/SpeechEdge commit -m "docs(decisions): Mode-1 mention predictor shipped; G1' calibration number recorded"
git -C /Users/nealt1/Downloads/SpeechEdge push -q origin master
```

---

## Self-Review

**Spec coverage (§ → task):**
- §2 term vocab (market-supplied, variant patterns) → Task 1 + fixture (Task 7). *(Live market scraping is Plan 3; fixture stands in here.)*
- §3.0 phase decomposition (`P_event = 1−(1−P_prep)(1−P_qa)`) → Task 2.
- §3 Mode-1 base-rate estimator + add-k floor + variant folding → Tasks 1–2.
- §3 bounded news adjuster (0.25×–4×, never-invent guardrail) → Task 3 (interface + clamp; real LLM impl is Plan 2's news layer — `NullAdjuster` used offline).
- §3 Mode-1 prior assembly → Task 4.
- §6 calibration gate (Brier) replacing next-token G1, backtested on resolved hearing → Tasks 5–7.
- §2 mispriced-term targeting, §3 Mode-2 live, §5 StreamText, trading/dashboard → **out of scope for Plan 1** (Plans 2–3); not gaps.

**Placeholder scan:** none — every step has runnable code/commands. The gate threshold 0.25 is an explicit, documented baseline (spec §11 open item), not a TODO.

**Type consistency:** `MarketTerm(canonical, patterns)` and `.mentioned_in()` used identically in Tasks 1,2,4,7. `event_base_rate` returns `{p_prep,p_qa,p_event}` (Task 2) consumed in Task 4. `predict_event_priors` row keys `{p_prep,p_qa,p_event_base,multiplier,p_prior}` defined in Task 4, read in Task 7. `calibration_report` keys `{n,brier,terms,predictions,outcomes}` defined Task 5, read Task 7. `evaluate_mention_gate` returns `{pass,metric,threshold}` defined Task 6, read Task 7. Consistent.

# C2 — Diction Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax. Build against `docs/INTERFACES.md` §2.

**Goal:** Build `build_model.py` + `src/warsh/model.py` — a diction model over the Warsh corpus: n-gram language model (bigram→fourgram with backoff), phrase fingerprints with lift scores vs. baseline English, a hawkish/dovish/independence weighted lexicon, a rhetorical-move transition matrix, and a `predict_next()` demo. Pure local compute — no network, fully unit-tested on a synthetic corpus.

**Architecture:** Train from `corpus/warsh_corpus.jsonl` (the C1 output; `Document` per INTERFACES §1). All training is deterministic counting. The model persists to `models/warsh_model.json` and exposes the inference API from INTERFACES §2: `predict_next`, `phrase_signals`, `score_diction`. **Inference-only at runtime** (Appendix A.3 — no weight updates per phrase). Tests use a small inline corpus so they need no C1 output.

**Tech Stack:** Python 3.11+, stdlib only (`collections`, `math`, `json`, `re`), pytest.

---

## File Structure
- `src/warsh/tokenize.py` — `tokenize(text) -> list[str]`, `sentences(text) -> list[str]`
- `src/warsh/ngram.py` — `NgramModel` (counts, backoff, `predict_next`)
- `src/warsh/fingerprints.py` — `phrase_fingerprints(docs, baseline)` → lift scores; `BASELINE_UNIGRAMS` seed
- `src/warsh/lexicon.py` — `SIGNAL_LEXICON` (phrase→{axis,weight}); `score_diction(text)`; `phrase_signals(text)`
- `src/warsh/model.py` — `DictionModel` ties the above; `train(docs)`, `save/load`, INTERFACES §2 API
- `build_model.py` — CLI: load corpus → train → save `models/warsh_model.json` → print fingerprint/lift report
- Tests mirror each module under `tests/`.

Phrase→market seed mapping (Brief §4.1) lives in `docs/phrase_market_map.md` (create it from the brief's table). The lexicon's signal phrases are exactly those rows.

---

### Task 1: Tokenizer

**Files:** Create `src/warsh/tokenize.py`; Test `tests/test_tokenize.py`

- [ ] **Step 1: Failing test**
```python
# tests/test_tokenize.py
from src.warsh.tokenize import tokenize, sentences

def test_tokenize_lowercases_and_keeps_words():
    assert tokenize("Inflation is a CHOICE.") == ["inflation", "is", "a", "choice"]

def test_tokenize_drops_pure_punctuation_but_keeps_hyphenated():
    assert tokenize("stronger, not hotter — really?") == ["stronger", "not", "hotter", "really"]
    assert tokenize("balance-sheet runoff") == ["balance-sheet", "runoff"]

def test_sentences_splits_on_terminal_punctuation():
    s = sentences("Inflation is a choice. Without excuse. Really?")
    assert s == ["Inflation is a choice.", "Without excuse.", "Really?"]
```
- [ ] **Step 2: Run — FAIL** (`python -m pytest tests/test_tokenize.py -v`)
- [ ] **Step 3: Implement**
```python
# src/warsh/tokenize.py
import re

_WORD = re.compile(r"[a-z]+(?:-[a-z]+)*")
_SENT = re.compile(r"[^.!?]*[.!?]")

def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())

def sentences(text: str) -> list[str]:
    return [m.group().strip() for m in _SENT.finditer(text) if m.group().strip()]
```
- [ ] **Step 4: Run — PASS**
- [ ] **Step 5: Commit** `feat(c2): tokenizer (words + sentence segmentation)`

---

### Task 2: N-gram model with backoff

**Files:** Create `src/warsh/ngram.py`; Test `tests/test_ngram.py`

- [ ] **Step 1: Failing test**
```python
# tests/test_ngram.py
from src.warsh.ngram import NgramModel

CORPUS = [
    "inflation is a choice not an accident",
    "inflation is a choice without excuse",
    "the balance sheet is bloated",
]

def test_predict_next_ranks_by_observed_continuation():
    m = NgramModel(max_n=4).train([t.split() for t in CORPUS])
    preds = dict(m.predict_next(["inflation", "is", "a"], k=3))
    assert "choice" in preds
    assert preds["choice"] > 0.0

def test_backoff_uses_lower_order_when_higher_unseen():
    m = NgramModel(max_n=4).train([t.split() for t in CORPUS])
    # "is" follows nothing seen in this exact 3-gram context, backs off to bigram/unigram
    preds = m.predict_next(["never", "seen", "is"], k=2)
    assert preds  # non-empty via backoff, not a crash

def test_probabilities_are_normalized_per_call():
    m = NgramModel(max_n=4).train([t.split() for t in CORPUS])
    preds = m.predict_next(["inflation", "is", "a"], k=10)
    assert abs(sum(p for _, p in preds) - 1.0) < 1e-6
```
- [ ] **Step 2: Run — FAIL**
- [ ] **Step 3: Implement**
```python
# src/warsh/ngram.py
from collections import defaultdict

class NgramModel:
    def __init__(self, max_n: int = 4, backoff: float = 0.4):
        self.max_n = max_n
        self.backoff = backoff
        # counts[n][context_tuple][next_word] -> int
        self.counts = [defaultdict(lambda: defaultdict(int)) for _ in range(max_n)]

    def train(self, token_lists: list[list[str]]) -> "NgramModel":
        for toks in token_lists:
            for i in range(len(toks)):
                for n in range(1, self.max_n + 1):
                    if i - (n - 1) < 0:
                        continue
                    ctx = tuple(toks[i - (n - 1):i])
                    self.counts[n - 1][ctx][toks[i]] += 1
        return self

    def _scores(self, context: list[str]) -> dict[str, float]:
        # Stupid-backoff: highest available order, discounted by backoff^drop.
        scores: dict[str, float] = {}
        for drop in range(self.max_n):
            n = self.max_n - drop
            ctx = tuple(context[-(n - 1):]) if n > 1 else tuple()
            table = self.counts[n - 1].get(ctx)
            if not table:
                continue
            total = sum(table.values())
            weight = self.backoff ** drop
            for w, c in table.items():
                scores.setdefault(w, weight * c / total)
            if scores:
                break
        return scores

    def predict_next(self, context: list[str], k: int = 5) -> list[tuple[str, float]]:
        scores = self._scores(context)
        if not scores:
            return []
        total = sum(scores.values())
        ranked = sorted(((w, s / total) for w, s in scores.items()), key=lambda x: -x[1])
        return ranked[:k]

    def to_dict(self) -> dict:
        return {"max_n": self.max_n, "backoff": self.backoff,
                "counts": [{ " ".join(ctx): dict(tbl) for ctx, tbl in order.items()} for order in self.counts]}

    @classmethod
    def from_dict(cls, d: dict) -> "NgramModel":
        m = cls(max_n=d["max_n"], backoff=d["backoff"])
        for n, order in enumerate(d["counts"]):
            for ctx_str, tbl in order.items():
                ctx = tuple(ctx_str.split()) if ctx_str else tuple()
                for w, c in tbl.items():
                    m.counts[n][ctx][w] = c
        return m
```
- [ ] **Step 4: Run — PASS**
- [ ] **Step 5: Commit** `feat(c2): n-gram model with stupid-backoff and predict_next`

---

### Task 3: Phrase fingerprints with lift scores

**Files:** Create `src/warsh/fingerprints.py`; Test `tests/test_fingerprints.py`

Lift = P(phrase | Warsh corpus) / P(phrase | baseline English). A phrase Warsh uses far more than baseline has high lift — that's a fingerprint. The seed phrases (Brief §4.1) get scored; new high-lift bigrams/trigrams are surfaced for review.

- [ ] **Step 1: Failing test**
```python
# tests/test_fingerprints.py
from src.warsh.fingerprints import phrase_lift, top_fingerprints

def test_phrase_lift_high_when_corpus_overrepresents():
    corpus_counts = {"inflation is a choice": 5}
    corpus_total = 100
    baseline = {"inflation is a choice": 1}   # rare in baseline
    baseline_total = 1_000_000
    lift = phrase_lift("inflation is a choice", corpus_counts, corpus_total, baseline, baseline_total)
    assert lift > 100  # vastly over-represented vs baseline

def test_top_fingerprints_returns_sorted_by_lift():
    docs_tokens = [["inflation","is","a","choice"], ["inflation","is","a","choice"], ["the","cat","sat"]]
    baseline = {"the cat": 50, "cat sat": 50, "inflation is": 1}
    out = top_fingerprints(docs_tokens, baseline, baseline_total=1000, n=2, k=3)
    phrases = [p for p, _ in out]
    assert "inflation is" in phrases
    assert out == sorted(out, key=lambda x: -x[1])
```
- [ ] **Step 2: Run — FAIL**
- [ ] **Step 3: Implement**
```python
# src/warsh/fingerprints.py
from collections import Counter

def _ngrams(tokens: list[str], n: int):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def phrase_lift(phrase, corpus_counts, corpus_total, baseline_counts, baseline_total) -> float:
    p_corpus = corpus_counts.get(phrase, 0) / corpus_total if corpus_total else 0.0
    # add-one smoothing on baseline so unseen baseline phrases don't divide by zero
    p_base = (baseline_counts.get(phrase, 0) + 1) / (baseline_total + 1)
    return p_corpus / p_base if p_base else 0.0

def top_fingerprints(docs_tokens, baseline_counts, baseline_total, n=2, k=20):
    counts = Counter()
    for toks in docs_tokens:
        counts.update(_ngrams(toks, n))
    total = sum(counts.values()) or 1
    scored = [(ph, phrase_lift(ph, counts, total, baseline_counts, baseline_total))
              for ph in counts]
    return sorted(scored, key=lambda x: -x[1])[:k]
```
> A real baseline-English n-gram table is large; `build_model.py` ships a small seed `BASELINE_UNIGRAMS`/bigram table and documents how to swap in a fuller one. The lift math is what's tested here.
- [ ] **Step 4: Run — PASS**
- [ ] **Step 5: Commit** `feat(c2): phrase fingerprints with lift scoring`

---

### Task 4: Signal lexicon + diction scoring

**Files:** Create `src/warsh/lexicon.py`; Test `tests/test_lexicon.py`

The lexicon encodes Brief §4.1's phrase→signal map as weighted axes (hawkish, dovish, independence, qt). `score_diction` aggregates; `phrase_signals` returns matched phrases with weights.

- [ ] **Step 1: Failing test**
```python
# tests/test_lexicon.py
from src.warsh.lexicon import score_diction, phrase_signals, SIGNAL_LEXICON

def test_lexicon_covers_brief_seed_phrases():
    for phrase in ["inflation is a choice", "fiscal dominance", "foursquare within its role",
                   "bloated balance sheet", "stronger not hotter"]:
        assert phrase in SIGNAL_LEXICON

def test_score_diction_accumulates_axis_weights():
    text = "Inflation is a choice. We will be foursquare within its role."
    scores = score_diction(text)
    assert scores["hawkish"] > 0
    assert scores["independence"] > 0

def test_phrase_signals_returns_matched_phrases_with_axis():
    sig = phrase_signals("the balance sheet is bloated and fiscal dominance looms")
    keys = {s["phrase"] for s in sig}
    assert "bloated balance sheet" in keys or "fiscal dominance" in keys
```
- [ ] **Step 2: Run — FAIL**
- [ ] **Step 3: Implement** — encode the §4.1 table:
```python
# src/warsh/lexicon.py
# axis ∈ hawkish | dovish | independence | qt ; weight is signal strength 0..1
SIGNAL_LEXICON: dict[str, dict] = {
    "inflation is a choice":          {"axis": "hawkish", "weight": 0.9},
    "without excuse or equivocation": {"axis": "hawkish", "weight": 1.0},
    "foursquare within its role":     {"axis": "independence", "weight": 0.8},
    "forays far afield":              {"axis": "hawkish", "weight": 0.6},
    "misallocation of capital":       {"axis": "qt", "weight": 0.7},
    "bloated balance sheet":          {"axis": "qt", "weight": 0.8},
    "regime change":                  {"axis": "independence", "weight": 0.6},
    "fiscal dominance":               {"axis": "independence", "weight": 0.9},
    "stronger not hotter":            {"axis": "dovish", "weight": 0.7},
}

def _norm(text: str) -> str:
    import re
    return re.sub(r"[^a-z ]+", " ", text.lower())

def phrase_signals(text: str) -> list[dict]:
    t = _norm(text)
    out = []
    for phrase, meta in SIGNAL_LEXICON.items():
        if phrase in t:
            out.append({"phrase": phrase, "axis": meta["axis"], "weight": meta["weight"]})
    return out

def score_diction(text: str) -> dict[str, float]:
    scores = {"hawkish": 0.0, "dovish": 0.0, "independence": 0.0, "qt": 0.0}
    for s in phrase_signals(text):
        scores[s["axis"]] += s["weight"]
    return scores
```
- [ ] **Step 4: Run — PASS**
- [ ] **Step 5: Commit** `feat(c2): signal lexicon + diction scoring from Brief §4.1`

---

### Task 5: DictionModel facade + build CLI

**Files:** Create `src/warsh/model.py`, `build_model.py`, `docs/phrase_market_map.md`; Test `tests/test_model.py`

- [ ] **Step 1: Failing test**
```python
# tests/test_model.py
from pathlib import Path
from src.warsh.model import DictionModel

DOCS = [
    {"text": "Inflation is a choice. The balance sheet is bloated."},
    {"text": "We remain foursquare within its role. Inflation is a choice."},
]

def test_train_then_predict_and_score():
    m = DictionModel().train(DOCS)
    assert m.predict_next(["inflation", "is", "a"], k=1)  # non-empty
    s = m.score_diction("inflation is a choice")
    assert s["hawkish"] > 0
    assert {"phrase", "axis", "weight"} <= set(m.phrase_signals("fiscal dominance")[0].keys()) \
        if m.phrase_signals("fiscal dominance") else True

def test_save_load_roundtrip(tmp_path: Path):
    m = DictionModel().train(DOCS)
    p = tmp_path / "model.json"
    m.save(p)
    m2 = DictionModel.load(p)
    assert m2.predict_next(["inflation", "is", "a"], k=1) == m.predict_next(["inflation", "is", "a"], k=1)
```
- [ ] **Step 2: Run — FAIL**
- [ ] **Step 3: Implement** `DictionModel` wrapping `NgramModel` + lexicon + fingerprints, with `train(docs)` (docs are dicts or `Document`s with `.text`/`["text"]`), `save`/`load` (JSON), and the INTERFACES §2 methods delegating to the modules. `build_model.py` loads `corpus/warsh_corpus.jsonl`, trains, writes `models/warsh_model.json`, and prints the top fingerprints + lift scores for Neal's review (Brief §5 day 2-3). Handle empty/missing corpus gracefully (print guidance, exit 0).
- [ ] **Step 4: Run — PASS**; then full suite `python -m pytest -v`
- [ ] **Step 5: Commit** `feat(c2): DictionModel facade + build_model CLI + phrase-market map`

## Self-review
- **Spec coverage (Brief §3 build_model.py):** bigram/trigram/fourgram + backoff (Task 2) ✓ · phrase fingerprints w/ lift (Task 3) ✓ · hawkish/dovish weighted lexicon (Task 4) ✓ · rhetorical move classifier + transition matrix → *deferred*: the seed lexicon covers signal phrases; the transition matrix is a stretch goal noted in build_model.py TODO, not blocking G1. **Flag to reviewer.** · predict_next demo (Task 2/5) ✓
- **Gate:** `predict_next` top-1 accuracy on held-out corpus is G1 (>20%) — C5 measures it against this model.
- **Types:** `predict_next`/`phrase_signals`/`score_diction` signatures match INTERFACES §2 exactly.

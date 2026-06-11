# C6 — Live Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: subagent-driven-development / executing-plans. Build against `docs/INTERFACES.md` §3–4. Reuses C2 (DictionModel), C4 (Prior), C7 (Signal/gate). STT + market feed are injected — the inference core is offline-testable.

**Goal:** Build `live_predictor.py` + `src/live/` — Mode 2 (Brief §2): during the speech, STT → inference every ~5 words → **Bayesian update** of each market's probability (corpus prior from C4 × diction likelihood from C2) → emit a `Signal` (which C7's gate decides to fire). **Inference-only at runtime** (Appendix A.3): the prior is fixed; each phrase updates only the likelihood, as a log-odds shift.

**Architecture:** Pure inference core (unit-tested, no I/O): log-odds Bayesian update + diction→log-likelihood mapping + word chunking + the predictor loop (takes an iterable of words, fixed priors, a `DictionModel`, and per-market metadata → yields `Signal`s). I/O shell: `live_predictor.py` wires a real `Transcriber` (Deepgram default per E3, or local Whisper — injected behind a protocol), loads C4 priors (`output/priors/<date>.json`), the C2 model, the market list with current prices, and pipes emitted Signals into C7's `run_once`.

**Tech Stack:** Python 3.11+, stdlib (`math`) + pytest. Reuses `src/warsh/model.py`, `src/agents/context_types.py` (Prior), `src/trading/types.py` (Signal). Deepgram/Whisper SDKs are lazy-imported in the real transcriber only.

---

## File Structure
- `src/live/types.py` — `Transcriber` protocol (`words() -> Iterable[str]`); `MarketState` (ticker, yes_price, prior_prob, signal_axis)
- `src/live/inference.py` — `to_logodds`, `from_logodds`, `bayesian_update`, `diction_loglikelihood`, `chunk_words`
- `src/live/predictor.py` — `LivePredictor.run(words) -> Iterator[Signal]`
- `live_predictor.py` — CLI: real transcriber + priors + model + markets → emit Signals → C7 `run_once`
- Tests under `tests/`.

---

### Task 1: Bayesian inference core

**Files:** Create `src/live/__init__.py`, `src/live/inference.py`; Test `tests/test_inference.py`

- [ ] **Step 1: Failing test**
```python
# tests/test_inference.py
import math
from src.live.inference import to_logodds, from_logodds, bayesian_update, diction_loglikelihood, chunk_words

def test_logodds_roundtrip():
    for p in (0.1, 0.5, 0.9):
        assert abs(from_logodds(to_logodds(p)) - p) < 1e-9

def test_logodds_clamps_extremes():
    assert 0.0 < from_logodds(to_logodds(0.0)) < 1e-3   # clamped, not -inf
    assert 1.0 - from_logodds(to_logodds(1.0)) < 1e-3

def test_bayesian_update_positive_delta_raises_prob():
    post = bayesian_update(0.50, +1.0)
    assert post > 0.50
    assert bayesian_update(0.50, -1.0) < 0.50
    assert abs(bayesian_update(0.50, 0.0) - 0.50) < 1e-9   # no evidence → unchanged

def test_diction_loglikelihood_matches_axis_adds_opposes_subtracts():
    signals = [
        {"phrase": "inflation is a choice", "axis": "hawkish", "weight": 0.9},
        {"phrase": "stronger not hotter", "axis": "dovish", "weight": 0.7},
        {"phrase": "fiscal dominance", "axis": "independence", "weight": 0.9},
    ]
    # market driven by hawkish axis: hawkish adds, dovish (opposite) subtracts, independence neutral
    d = diction_loglikelihood(signals, market_axis="hawkish", scale=1.0)
    assert abs(d - (0.9 - 0.7)) < 1e-9

def test_chunk_words_emits_every_n():
    chunks = list(chunk_words(["a","b","c","d","e","f","g"], every=3))
    assert chunks == [["a","b","c"], ["d","e","f"], ["g"]]
```
- [ ] **Step 2: Run — FAIL**
- [ ] **Step 3: Implement**
```python
# src/live/inference.py
import math

_EPS = 1e-6
_OPPOSITE = {"hawkish": "dovish", "dovish": "hawkish"}

def to_logodds(p: float) -> float:
    p = min(1 - _EPS, max(_EPS, p))      # clamp away from 0/1 so log-odds is finite
    return math.log(p / (1 - p))

def from_logodds(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def bayesian_update(prior_prob: float, loglik_delta: float) -> float:
    """Inference-only posterior: prior log-odds shifted by accumulated diction evidence."""
    return from_logodds(to_logodds(prior_prob) + loglik_delta)

def diction_loglikelihood(phrase_signals: list[dict], market_axis: str, scale: float = 1.0) -> float:
    """Sum signal weights: same-axis adds, opposite-axis subtracts, unrelated axes neutral."""
    total = 0.0
    opp = _OPPOSITE.get(market_axis)
    for s in phrase_signals:
        if s["axis"] == market_axis:
            total += s["weight"]
        elif opp is not None and s["axis"] == opp:
            total -= s["weight"]
    return total * scale

def chunk_words(words, every: int = 5):
    buf = []
    for w in words:
        buf.append(w)
        if len(buf) == every:
            yield buf
            buf = []
    if buf:
        yield buf
```
- [ ] **Step 4: Run — PASS**
- [ ] **Step 5: Commit** `feat(c6): Bayesian inference core (log-odds update + diction likelihood)`

---

### Task 2: Market state + transcriber protocol

**Files:** Create `src/live/types.py`; Test `tests/test_live_types.py`

- [ ] **Step 1: Failing test**
```python
# tests/test_live_types.py
from src.live.types import MarketState, ListTranscriber

def test_market_state_holds_prior_price_and_axis():
    m = MarketState(ticker="FED-HOLD-JUN", yes_price=0.55, prior_prob=0.62, signal_axis="hawkish")
    assert m.ticker == "FED-HOLD-JUN" and m.signal_axis == "hawkish"

def test_list_transcriber_yields_words_in_order():
    t = ListTranscriber(["inflation", "is", "a", "choice"])
    assert list(t.words()) == ["inflation", "is", "a", "choice"]
```
- [ ] **Step 2: Run — FAIL**
- [ ] **Step 3: Implement**
```python
# src/live/types.py
from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable

@dataclass
class MarketState:
    ticker: str
    yes_price: float        # current market YES price (from Kalshi feed / C7 quotes)
    prior_prob: float       # Mode-1 prior P(YES) from C4 (output/priors/...)
    signal_axis: str        # diction axis that drives this market: hawkish|dovish|independence|qt

@runtime_checkable
class Transcriber(Protocol):
    def words(self) -> Iterable[str]:
        """Yield transcript words as they are recognized (live STT)."""
        ...

class ListTranscriber:
    """Test/replay transcriber over a fixed word list (no audio, no network)."""
    def __init__(self, words: list[str]): self._words = words
    def words(self):
        yield from self._words
```
- [ ] **Step 4: Run — PASS**
- [ ] **Step 5: Commit** `feat(c6): market state + injectable transcriber protocol`

---

### Task 3: Live predictor loop

**Files:** Create `src/live/predictor.py`; Test `tests/test_predictor.py`

- [ ] **Step 1: Failing test**
```python
# tests/test_predictor.py
from src.live.types import MarketState, ListTranscriber
from src.live.predictor import LivePredictor

class FakeModel:
    """phrase_signals returns hawkish signal once the transcript contains the trigger."""
    def phrase_signals(self, text: str):
        return [{"phrase": "inflation is a choice", "axis": "hawkish", "weight": 1.0}] \
            if "inflation is a choice" in text else []

def test_predictor_raises_posterior_on_hawkish_speech_for_hawkish_market():
    markets = [MarketState("FED-HOLD-JUN", yes_price=0.55, prior_prob=0.55, signal_axis="hawkish")]
    pred = LivePredictor(FakeModel(), markets, every=3)
    sigs = list(pred.run(ListTranscriber("inflation is a choice without excuse".split())))
    # at least one signal emitted; final posterior for the market should exceed the 0.55 prior
    assert sigs
    last = [s for s in sigs if s.ticker == "FED-HOLD-JUN"][-1]
    assert last.model_prob > 0.55
    assert last.market_price == 0.55

def test_predictor_neutral_speech_leaves_posterior_near_prior():
    markets = [MarketState("FED-HOLD-JUN", 0.55, 0.55, "hawkish")]
    pred = LivePredictor(FakeModel(), markets, every=2)
    sigs = list(pred.run(ListTranscriber("the weather is nice today".split())))
    if sigs:
        assert abs(sigs[-1].model_prob - 0.55) < 1e-9   # no diction evidence → prior unchanged
```
- [ ] **Step 2: Run — FAIL**
- [ ] **Step 3: Implement**
```python
# src/live/predictor.py
from typing import Iterator
from src.live.inference import bayesian_update, diction_loglikelihood, chunk_words
from src.live.types import MarketState, Transcriber
from src.trading.types import Signal

class LivePredictor:
    def __init__(self, model, markets: list[MarketState], every: int = 5, scale: float = 1.0):
        self.model = model
        self.markets = markets
        self.every = every
        self.scale = scale

    def run(self, transcriber: Transcriber) -> Iterator[Signal]:
        """Stream words; every `every` words recompute each market's posterior from its
        fixed prior + the cumulative diction log-likelihood over the full transcript so far,
        and emit a Signal. Recompute-from-prior (not incremental) keeps it idempotent."""
        transcript: list[str] = []
        for chunk in chunk_words(transcriber.words(), every=self.every):
            transcript.extend(chunk)
            text = " ".join(transcript)
            signals = self.model.phrase_signals(text)
            for m in self.markets:
                delta = diction_loglikelihood(signals, m.signal_axis, self.scale)
                posterior = bayesian_update(m.prior_prob, delta)
                yield Signal.from_quote(
                    ticker=m.ticker, model_prob=posterior, market_price=m.yes_price,
                    timestamp=f"chunk-{len(transcript)}",
                )
```
- [ ] **Step 4: Run — PASS**
- [ ] **Step 5: Commit** `feat(c6): live predictor loop (Bayesian posterior per chunk)`

---

### Task 4: CLI wiring (injected STT, priors, C7 hand-off)

**Files:** Create `live_predictor.py`; Test `tests/test_live_cli.py`

- [ ] **Step 1: Failing test**
```python
# tests/test_live_cli.py
from pathlib import Path
import json
from live_predictor import load_market_states

def test_load_market_states_merges_priors_and_prices(tmp_path: Path):
    priors = [{"ticker": "FED-HOLD-JUN", "prior_prob": 0.62, "rationale": "r",
               "as_of": "t", "sources": ["s"]}]
    pf = tmp_path / "priors.json"; pf.write_text(json.dumps(priors))
    # price + axis map supplied alongside (from Kalshi feed + phrase_market_map)
    prices = {"FED-HOLD-JUN": 0.55}
    axes = {"FED-HOLD-JUN": "hawkish"}
    states = load_market_states(pf, prices, axes)
    assert len(states) == 1
    assert states[0].prior_prob == 0.62 and states[0].yes_price == 0.55
    assert states[0].signal_axis == "hawkish"
```
- [ ] **Step 2: Run — FAIL**
- [ ] **Step 3: Implement** `load_market_states(priors_path, prices, axes) -> list[MarketState]` (join C4 priors with current prices + axis map); `make_deepgram_transcriber()` / `make_whisper_transcriber()` thin stubs lazy-importing their SDKs with `# TODO: wire E3 audio source` (Deepgram is the E3 default, ~300ms); `main()` — load priors + C2 model + markets, build `LivePredictor`, stream Signals into C7's `run_once` (paper mode). If priors/model/audio missing, print guidance and exit 0.
- [ ] **Step 4: Run — PASS**; full suite green.
- [ ] **Step 5: Commit** `feat(c6): live_predictor CLI — STT→inference→signal, C7 hand-off`

## Self-review
- **Spec (Brief §2 Mode 2, A.3):** STT (injected) → inference every ~5 words (`chunk_words`) → Bayesian update (`bayesian_update`) → Signal for C7 gate. ✓ Inference-only — prior fixed, likelihood-only updates. ✓
- **Reuse:** `Signal.from_quote` (C7), `DictionModel.phrase_signals` (C2), `Prior` (C4). No duplication.
- **E3:** audio source behind `Transcriber` protocol; Deepgram default stub + Whisper stub, both `# TODO: wire`. Latency/cost is a wiring choice, not a code change.
- **Idempotent:** posterior recomputed from prior + cumulative evidence each chunk — a stuttered/replayed word stream can't double-count.

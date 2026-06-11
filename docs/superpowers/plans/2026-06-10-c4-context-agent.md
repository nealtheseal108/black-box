# C4 — Mode-1 Context Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: subagent-driven-development / executing-plans. Build against `docs/INTERFACES.md` §3–4. Read the bundled `claude-api` skill guidance summarized below before writing the API call.

**Goal:** Build `context_agent.py` + `src/agents/` — the pre-speech (Mode 1) pipeline. 24–48h before an FOMC event, assemble the macro environment Warsh must respond to (data prints, futures pricing, news, his own recent statements), then have Claude synthesize a **prior probability distribution per active Kalshi market**, written to `output/priors/<event_date>.json` (INTERFACES §4).

**Architecture:** Separate **context assembly** (gathering source snippets — injectable fetchers, so tests use fixtures) from **synthesis** (one Claude call that maps assembled context → per-market priors). The Claude call is the only network dependency and is injected as a function, so all prior-shaping/parse/validate logic is unit-tested offline. Output validated with Pydantic to guarantee the INTERFACES §4 shape.

**Tech Stack:** Python 3.11+, `anthropic` SDK, `pydantic`, pytest. **Model `claude-opus-4-8`**, `thinking={"type":"adaptive"}`, `output_config={"effort":"high"}`, structured output via `client.messages.parse()` with a Pydantic schema. Key from `ANTHROPIC_API_KEY` (env; never committed).

---

## File Structure
- `src/agents/context_types.py` — `MacroSnapshot`, `MarketContext`, `Prior` (Pydantic models matching INTERFACES §4)
- `src/agents/assemble.py` — `assemble_context(markets, fetchers) -> MacroSnapshot` (pure orchestration over injected fetchers)
- `src/agents/synthesize.py` — `build_prompt(snapshot, market)`, `synthesize_priors(snapshot, markets, call_fn) -> list[Prior]` (`call_fn` injected)
- `context_agent.py` — CLI: real fetchers + real Anthropic client → `output/priors/<event_date>.json`
- Tests under `tests/`.

---

### Task 1: Pydantic types (INTERFACES §4)

**Files:** Create `src/agents/__init__.py`, `src/agents/context_types.py`; Test `tests/test_context_types.py`

- [ ] **Step 1: Failing test**
```python
# tests/test_context_types.py
from src.agents.context_types import Prior
import json

def test_prior_matches_interface_shape():
    p = Prior(ticker="FED-RATE-CUT-JUN", prior_prob=0.62,
              rationale="April CPI 3yr high + energy shock argues for hold",
              as_of="2026-06-14T00:00:00Z", sources=["fmp:cpi", "speaker:hearing-2026-04-21"])
    d = json.loads(p.model_dump_json())
    assert set(d) == {"ticker", "prior_prob", "rationale", "as_of", "sources"}
    assert 0.0 <= d["prior_prob"] <= 1.0

def test_prior_prob_must_be_a_probability():
    import pytest, pydantic
    with pytest.raises(pydantic.ValidationError):
        Prior(ticker="X", prior_prob=1.4, rationale="r", as_of="t", sources=[])
```
- [ ] **Step 2: Run — FAIL**
- [ ] **Step 3: Implement** `Prior` (with `field_validator`/`Field(ge=0,le=1)` on `prior_prob`), `MarketContext` (`ticker`, `title`, `question`, `yes_price`), `MacroSnapshot` (`as_of`, `data_prints: list[str]`, `futures: list[str]`, `news: list[str]`, `speaker_recent: list[str]`).
- [ ] **Step 4: Run — PASS**
- [ ] **Step 5: Commit** `feat(c4): pydantic context + prior types (INTERFACES §4)`

---

### Task 2: Context assembly over injected fetchers

**Files:** Create `src/agents/assemble.py`; Test `tests/test_assemble.py`

Each source is a `Callable[[], list[str]]` injected so tests use fakes (no network). Brief §2: macro agents → news agents → speaker-recent agent. Per Authority Matrix §7, the source list is autonomous — seed with: FMP data prints + futures, financial news, and the speaker's own recent statements (from the C1 corpus, most-recent docs).

- [ ] **Step 1: Failing test**
```python
# tests/test_assemble.py
from src.agents.assemble import assemble_context

def test_assemble_gathers_all_sources_into_snapshot():
    fetchers = {
        "data_prints": lambda: ["April CPI 3.4% (3yr high)"],
        "futures": lambda: ["Fed funds futures: 55% hold priced"],
        "news": lambda: ["Energy shock on supply disruption"],
        "speaker_recent": lambda: ["Apr-21 hearing: 'inflation is a choice'"],
    }
    snap = assemble_context(as_of="2026-06-14T00:00:00Z", fetchers=fetchers)
    assert snap.data_prints == ["April CPI 3.4% (3yr high)"]
    assert snap.speaker_recent and "inflation is a choice" in snap.speaker_recent[0]

def test_assemble_tolerates_a_failing_fetcher():
    def boom(): raise RuntimeError("source down")
    fetchers = {"data_prints": boom, "futures": lambda: [], "news": lambda: [], "speaker_recent": lambda: []}
    snap = assemble_context(as_of="t", fetchers=fetchers)
    assert snap.data_prints == []   # failure degrades gracefully, doesn't crash the run
```
- [ ] **Step 2: Run — FAIL**
- [ ] **Step 3: Implement** `assemble_context` calling each fetcher in a try/except (failure → `[]`, logged to stderr), populating `MacroSnapshot`.
- [ ] **Step 4: Run — PASS**
- [ ] **Step 5: Commit** `feat(c4): context assembly over injectable source fetchers`

---

### Task 3: Prompt construction + prior synthesis (injected Claude call)

**Files:** Create `src/agents/synthesize.py`; Test `tests/test_synthesize.py`

- [ ] **Step 1: Failing test**
```python
# tests/test_synthesize.py
from src.agents.context_types import MacroSnapshot, MarketContext, Prior
from src.agents.synthesize import build_prompt, synthesize_priors

SNAP = MacroSnapshot(as_of="2026-06-14T00:00:00Z",
    data_prints=["April CPI 3.4%"], futures=["55% hold priced"],
    news=["energy shock"], speaker_recent=["'inflation is a choice'"])
MKTS = [MarketContext(ticker="FED-HOLD-JUN", title="June hold", question="Will the Fed hold?", yes_price=0.55)]

def test_build_prompt_includes_context_and_market():
    p = build_prompt(SNAP, MKTS[0])
    assert "inflation is a choice" in p
    assert "FED-HOLD-JUN" in p and "0.55" in p

def test_synthesize_priors_uses_injected_call_and_returns_priors():
    # Fake Claude call returns a validated Prior per market — no network.
    def fake_call(prompt: str, market: MarketContext) -> Prior:
        return Prior(ticker=market.ticker, prior_prob=0.62, rationale="hawkish lean",
                     as_of=SNAP.as_of, sources=["fmp", "speaker"])
    priors = synthesize_priors(SNAP, MKTS, call_fn=fake_call)
    assert len(priors) == 1
    assert priors[0].ticker == "FED-HOLD-JUN"
    assert 0 <= priors[0].prior_prob <= 1
```
- [ ] **Step 2: Run — FAIL**
- [ ] **Step 3: Implement** `build_prompt` (renders snapshot + market into a synthesis instruction asking for P(YES), rationale, and which sources drove it — diverging from market price where the speaker's linguistic patterns/known stance warrant), and `synthesize_priors` looping markets through `call_fn`. The real `call_fn` (in `context_agent.py`) uses:
```python
import anthropic
from src.agents.context_types import Prior
def make_anthropic_call(client: anthropic.Anthropic):
    def call(prompt: str, market) -> Prior:
        resp = client.messages.parse(
            model="claude-opus-4-8", max_tokens=2000,
            thinking={"type": "adaptive"}, output_config={"effort": "high"},
            messages=[{"role": "user", "content": prompt}],
            output_format=Prior,
        )
        return resp.parsed_output
    return call
```
- [ ] **Step 4: Run — PASS**
- [ ] **Step 5: Commit** `feat(c4): prompt construction + injected prior synthesis`

---

### Task 4: CLI wiring + output writer

**Files:** Create `context_agent.py`; Test `tests/test_context_agent.py`

- [ ] **Step 1: Failing test**
```python
# tests/test_context_agent.py
import json
from pathlib import Path
from src.agents.context_types import Prior
from context_agent import write_priors

def test_write_priors_emits_interface_json(tmp_path: Path):
    priors = [Prior(ticker="X", prior_prob=0.5, rationale="r", as_of="t", sources=["s"])]
    out = tmp_path / "2026-06-16.json"
    write_priors(priors, out)
    data = json.loads(out.read_text())
    assert data[0]["ticker"] == "X" and set(data[0]) == {"ticker","prior_prob","rationale","as_of","sources"}
```
- [ ] **Step 2: Run — FAIL**
- [ ] **Step 3: Implement** `write_priors(priors, path)` → JSON array; `main()` assembles real fetchers (FMP/news/corpus-recent), builds the Anthropic client, synthesizes, writes `output/priors/<event_date>.json`, prints a per-market `prior_prob vs yes_price` divergence table (these divergences are the Mode-1 bets). Real fetchers may be stubbed with a `# TODO: wire FMP/news` and a clear log line — the synthesis path and output contract are what this task guarantees; live data wiring is incremental.
- [ ] **Step 4: Run — PASS**; full suite green.
- [ ] **Step 5: Commit** `feat(c4): context_agent CLI — priors output + divergence report`

## Self-review
- **Spec (Brief §2 Mode 1):** macro/news/speaker-recent assembly ✓ · LLM synthesizer → prior P(outcome) per market ✓ · output feeds C6 Bayesian prior (INTERFACES §4) ✓
- **Authority:** ~540 LLM calls/presser budget is for Mode-2 live; Mode-1 is per-market pre-speech (low volume) — opus-4-8 justified. Batching/caching autonomous (§7). Cache the stable snapshot prefix across markets (prompt-caching).
- **Boundary:** the only network is `call_fn` + fetchers, both injected → fully offline-testable. `ANTHROPIC_API_KEY` via env.

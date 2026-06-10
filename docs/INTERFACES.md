# INTERFACES.md — Shared Data Contracts

> Every component builds against these. Parallel work (C2/C4/C7) must not invent its own versions. Changing a contract here is a decision — log it in DECISIONS.md and update dependents.

## 1. `Document` (corpus unit) — defined in `src/warsh/models.py`
```python
@dataclass
class Document:
    id: str           # stable slug, e.g. "2007-03-05-market-liquidity-and-risk"
    source: str       # fraser | fed | hoover | hoover_pdf | wsj_archive | hearing | manual
    title: str
    date: str         # ISO YYYY-MM-DD, "" if unknown
    url: str
    context_type: str # speech | essay | op_ed | interview | hearing | lecture
    text: str
    # word_count is a derived @property, NOT stored
```
Corpus persisted as **JSONL**: `corpus/warsh_corpus.jsonl` (one Document per line) + `corpus/warsh_corpus.manifest.json`. Trump corpus (later, C3): `corpus/trump_corpus.jsonl`, tagged by context type, Term-2 pressers only for production.

## 2. Diction model output (C2 → C5/C6) — `src/warsh/model.py`
The model is **inference-only at runtime** (Appendix A.3 — no weight updates per phrase). It exposes:
```python
predict_next(context_tokens: list[str], k: int = 5) -> list[tuple[str, float]]   # top-k next phrases w/ probs
phrase_signals(text: str) -> dict[str, float]   # signal phrase -> lift score (e.g. "inflation is a choice" -> hawkish weight)
score_diction(text: str) -> dict[str, float]    # {"hawkish": .., "dovish": .., "independence": ..}
```
Persisted artifact: `models/warsh_model.pkl` (or json). Phrase→market mapping seed lives in `docs/phrase_market_map.md` (Brief §4.1); C2 lift scores refine it.

## 3. Kalshi market (C4/C7) — `src/trading/types.py`
```python
@dataclass
class KalshiMarket:
    ticker: str          # Kalshi market id
    title: str
    yes_price: float     # last/mid, dollars 0.00–1.00
    no_price: float
    question: str        # what the contract resolves on

@dataclass
class Signal:
    ticker: str
    model_prob: float    # model P(YES)
    market_price: float  # current YES price
    edge: float          # model_prob - market_price (gross)
    side: str            # "yes" | "no"
    timestamp: str       # ISO 8601
```
**Fire rule:** emit an order only when `abs(edge) > threshold` (E2 default **0.08**) AND net edge after fees > **0.06** (G3). Threshold is config, not hardcoded.

## 4. Prior distribution (Mode 1 C4 → C6) — `src/trading/priors.py`
C4 emits, per active market, a JSON prior:
```python
{"ticker": str, "prior_prob": float, "rationale": str, "as_of": "ISO8601", "sources": [str]}
```
Written to `output/priors/<event_date>.json`. C6 loads these as the Bayesian prior; each live phrase updates the likelihood (Mode 2).

## 5. Config & secrets
- Settings via env / `.env` (gitignored). Never commit keys.
- Keys: `KALSHI_API_KEY`, `KALSHI_API_SECRET`, `DEEPGRAM_API_KEY`, `ANTHROPIC_API_KEY`.
- Runtime flags: `SPEECHEDGE_MODE` = `paper` (default) | `live`. **`live` requires explicit owner opt-in** (E4). `SIGNAL_THRESHOLD` default `0.08`. `BANKROLL` unset until E1 provided — Kelly sizer must refuse to size live without it.

## 6. Test & layout conventions
- Tests in `tests/`, mirror module path: `tests/test_<module>.py`. pytest. TDD: test first.
- Pure transforms (parsing, scoring, sizing math) unit-tested offline. I/O (network, Kalshi REST/WS, Deepgram, Anthropic) isolated behind injectable functions so tests need no live services.
- LLM calls (C4 synthesis, dashboard inference) are batched/cached where possible (~540 calls/45-min presser is the accepted budget — Brief §7 autonomous).

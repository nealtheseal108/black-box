"""C4 — Mode-1 Context Agent CLI.

Usage::

    ANTHROPIC_API_KEY=<key> python context_agent.py --event-date 2026-06-18

For each active Kalshi market it:
1. Assembles a MacroSnapshot (data prints, futures, news, speaker recent) via
   injectable fetchers (real fetchers below, stubbed with TODO markers).
2. Calls Claude claude-opus-4-8 with adaptive thinking to synthesise a prior P(YES)
   per market.
3. Writes ``output/priors/<event_date>.json`` (INTERFACES §4).
4. Prints a divergence table: prior_prob vs yes_price per market.

The ``call_fn`` and individual source fetchers are fully injectable so tests
(``tests/test_context_agent.py``, ``tests/test_synthesize.py``) run without
any network.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional

from src.agents.context_types import MacroSnapshot, MarketContext, Prior
from src.agents.assemble import assemble_context
from src.agents.synthesize import synthesize_priors


# ---------------------------------------------------------------------------
# Output writer (INTERFACES §4 contract)
# ---------------------------------------------------------------------------

def write_priors(priors: List[Prior], path: Path) -> None:
    """Write *priors* as a JSON array to *path* (creates parent dirs).

    Each element is the exact INTERFACES §4 shape:
        {"ticker", "prior_prob", "rationale", "as_of", "sources"}
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [json.loads(p.model_dump_json()) for p in priors]
    path.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Real Anthropic call factory (only used in main(); injected in tests)
# ---------------------------------------------------------------------------

def make_anthropic_call(client) -> Callable[[str, MarketContext], Prior]:
    """Return a call_fn that sends *prompt* to Claude and returns a Prior."""
    def call(prompt: str, market: MarketContext) -> Prior:
        resp = client.messages.parse(
            model="claude-opus-4-8",
            max_tokens=2000,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            messages=[{"role": "user", "content": prompt}],
            output_format=Prior,
        )
        return resp.parsed_output
    return call


# ---------------------------------------------------------------------------
# Real source fetchers (stubbed — wire FMP / news / corpus below)
# ---------------------------------------------------------------------------

def _fetch_data_prints() -> List[str]:
    """Fetch recent macro data prints (CPI, PCE, payrolls, etc.) from FMP.

    # TODO: wire FMP /economic-calendar endpoint + parse last 30d prints.
    """
    print("[context_agent] data_prints: stub — returning empty (wire FMP)", file=sys.stderr)
    return []


def _fetch_futures() -> List[str]:
    """Fetch current Fed-funds futures pricing from FMP.

    # TODO: wire FMP /quote/ZQUSD or CME-fed-funds futures endpoint.
    """
    print("[context_agent] futures: stub — returning empty (wire FMP)", file=sys.stderr)
    return []


def _fetch_news() -> List[str]:
    """Fetch recent financial news headlines relevant to Fed policy.

    # TODO: wire FMP /stock_news or financialnewsapi endpoint; filter for
    # 'FOMC' / 'inflation' / 'Federal Reserve' keywords.
    """
    print("[context_agent] news: stub — returning empty (wire news API)", file=sys.stderr)
    return []


def _fetch_speaker_recent() -> List[str]:
    """Fetch Warsh's most-recent statements from the C1 corpus.

    # TODO: load corpus/warsh_corpus.jsonl, sort by date desc, return the
    # most-recent 3–5 doc snippets (first 500 chars each) as strings.
    """
    print("[context_agent] speaker_recent: stub — returning empty (wire corpus)", file=sys.stderr)
    return []


# ---------------------------------------------------------------------------
# Divergence table
# ---------------------------------------------------------------------------

def _print_divergence_table(priors: List[Prior], markets: List[MarketContext]) -> None:
    mkt_price = {m.ticker: m.yes_price for m in markets}
    header = f"{'Ticker':<30} {'Prior':>7} {'Market':>7} {'Edge':>8}"
    print("\n" + header)
    print("-" * len(header))
    for p in priors:
        mp = mkt_price.get(p.ticker, float("nan"))
        edge = p.prior_prob - mp
        flag = "  <-- BET" if abs(edge) > float(os.getenv("SIGNAL_THRESHOLD", "0.08")) else ""
        print(f"{p.ticker:<30} {p.prior_prob:>7.3f} {mp:>7.3f} {edge:>+8.3f}{flag}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(
    event_date: Optional[str] = None,
    markets: Optional[List[MarketContext]] = None,
    call_fn: Optional[Callable[[str, MarketContext], Prior]] = None,
    fetchers: Optional[dict] = None,
) -> List[Prior]:
    """Run the Mode-1 Context Agent.

    Parameters are injectable for testing; defaults use real fetchers +
    real Anthropic client.
    """
    import argparse

    if event_date is None:
        parser = argparse.ArgumentParser(description="C4 Mode-1 Context Agent")
        parser.add_argument("--event-date", required=True,
                            help="FOMC event date YYYY-MM-DD (used for output filename)")
        args = parser.parse_args()
        event_date = args.event_date

    # Default markets (placeholder — TODO: pull from Kalshi API)
    if markets is None:
        print("[context_agent] markets: stub — using placeholder FED-HOLD market", file=sys.stderr)
        markets = [
            MarketContext(
                ticker="FED-HOLD-PLACEHOLDER",
                title="Fed holds rates at next FOMC",
                question="Will the Federal Reserve hold the federal funds rate unchanged?",
                yes_price=0.50,
            )
        ]

    # Default fetchers: real stubs
    if fetchers is None:
        fetchers = {
            "data_prints": _fetch_data_prints,
            "futures": _fetch_futures,
            "news": _fetch_news,
            "speaker_recent": _fetch_speaker_recent,
        }

    # Default call_fn: real Anthropic client
    if call_fn is None:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("[context_agent] ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
            sys.exit(1)
        client = anthropic.Anthropic(api_key=api_key)
        call_fn = make_anthropic_call(client)

    # 1. Assemble context
    from datetime import datetime, timezone
    as_of = datetime.now(timezone.utc).isoformat()
    snapshot: MacroSnapshot = assemble_context(as_of=as_of, fetchers=fetchers)

    # 2. Synthesise priors
    priors = synthesize_priors(snapshot, markets, call_fn=call_fn)

    # 3. Write output
    out_path = Path("output") / "priors" / f"{event_date}.json"
    write_priors(priors, out_path)
    print(f"[context_agent] wrote {len(priors)} prior(s) to {out_path}")

    # 4. Print divergence table
    _print_divergence_table(priors, markets)

    return priors


if __name__ == "__main__":
    main()

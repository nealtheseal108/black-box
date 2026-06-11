"""Prompt construction + prior synthesis for C4 Mode-1 Context Agent.

``build_prompt`` renders a ``MacroSnapshot`` + ``MarketContext`` into the
instruction string sent to Claude.

``synthesize_priors`` iterates markets through an injected ``call_fn``
(``Callable[[str, MarketContext], Prior]``), so the loop is fully offline-
testable.  The real call_fn (wired in ``context_agent.py``) talks to
Claude claude-opus-4-8.
"""
from __future__ import annotations

import textwrap
from typing import Callable, List

from src.agents.context_types import MacroSnapshot, MarketContext, Prior

CallFn = Callable[[str, MarketContext], Prior]


def build_prompt(snapshot: MacroSnapshot, market: MarketContext) -> str:
    """Render *snapshot* + *market* into a synthesis instruction for Claude.

    The prompt asks Claude to:
    1. Assess how each data source should shift P(YES) away from the current
       market price.
    2. Pay special attention to the speaker's own linguistic patterns and
       stated positions — these are the primary alpha source (Brief §2).
    3. Return a single prior_prob, rationale, and source list.
    """
    data_prints_block = "\n".join(f"  - {d}" for d in snapshot.data_prints) or "  (none)"
    futures_block = "\n".join(f"  - {f}" for f in snapshot.futures) or "  (none)"
    news_block = "\n".join(f"  - {n}" for n in snapshot.news) or "  (none)"
    speaker_block = "\n".join(f"  - {s}" for s in snapshot.speaker_recent) or "  (none)"

    return textwrap.dedent(f"""\
        You are a prediction-market analyst preparing a pre-speech prior for a Kalshi market.
        Today's date / snapshot time: {snapshot.as_of}

        ## Macro data prints
{data_prints_block}

        ## Fed-funds futures pricing
{futures_block}

        ## Recent news headlines
{news_block}

        ## Speaker's recent statements (primary alpha source)
{speaker_block}

        ---
        ## Target market
        Ticker : {market.ticker}
        Title  : {market.title}
        Question: {market.question}
        Current YES price: {market.yes_price:.2f}

        ---
        ## Your task
        1. Assess every piece of context above and determine how each item shifts
           the probability of YES resolution, diverging from the market price where
           the speaker's linguistic patterns / known stance warrant it.
        2. Produce a single prior probability for P(YES) as a float in [0, 1].
        3. Write a concise rationale (≤ 3 sentences) explaining the key drivers.
        4. List the source identifiers (e.g. "fmp:cpi", "speaker:hearing-2026-04-21",
           "news:energy-shock") that most influenced your estimate.

        Respond with ONLY a JSON object matching this exact schema (no markdown fences):
        {{"ticker": "{market.ticker}", "prior_prob": <float>, "rationale": "<string>",
          "as_of": "{snapshot.as_of}", "sources": [<string>, ...]}}
    """)


def synthesize_priors(
    snapshot: MacroSnapshot,
    markets: List[MarketContext],
    call_fn: CallFn,
) -> List[Prior]:
    """Synthesise a prior for every market in *markets*.

    Parameters
    ----------
    snapshot:
        The assembled macro context.
    markets:
        Active Kalshi markets to price.
    call_fn:
        ``(prompt, market) -> Prior``.  Injected so tests need no network.
        In production, use :func:`context_agent.make_anthropic_call`.

    Returns
    -------
    list[Prior]
        One :class:`Prior` per market, in the same order as *markets*.
    """
    priors: List[Prior] = []
    for market in markets:
        prompt = build_prompt(snapshot, market)
        prior = call_fn(prompt, market)
        priors.append(prior)
    return priors

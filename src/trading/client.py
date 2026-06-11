"""Kalshi client protocol, paper client, and gated live client.

Safety invariant (E4):
  make_client("live") raises RuntimeError unless SPEECHEDGE_ALLOW_LIVE=1 AND
  KALSHI_API_KEY + KALSHI_API_SECRET are present in the environment.
  No real order can be placed without this explicit, loud opt-in.

Credentials are read from env only — never logged, never committed.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

from src.trading.types import Fill, KalshiMarket, Order

# Output path for paper trade log — pinned to the repo root so it doesn't depend on cwd
# (keeps tests hermetic regardless of where pytest is invoked from).
_PAPER_LOG = Path(__file__).resolve().parents[2] / "output" / "paper_trades.jsonl"


@runtime_checkable
class KalshiClient(Protocol):
    """Protocol that all Kalshi client implementations must satisfy."""

    def place(self, order: Order) -> Fill:
        """Submit an order and return the resulting fill."""
        ...

    def quotes(self) -> list[KalshiMarket]:
        """Return current market quotes."""
        ...


class PaperKalshiClient:
    """Simulated client — no network, no real money.

    Fills are simulated at the limit price and logged to output/paper_trades.jsonl.
    This is the default client and the only one that can run without credentials.

    The log destination is injectable via ``log_path`` (defaults to ``_PAPER_LOG``)
    so tests can point it at a temp dir and stay hermetic — no writes to the
    repo-root output file.
    """

    def __init__(self, log_path: Path = _PAPER_LOG) -> None:
        self._log_path = log_path

    def place(self, order: Order) -> Fill:
        timestamp = datetime.now(timezone.utc).isoformat()
        fill = Fill(
            ticker=order.ticker,
            side=order.side,
            count=order.count,
            fill_price=order.limit_price,
            simulated=True,
            timestamp=timestamp,
        )
        self._log(fill, order)
        return fill

    def quotes(self) -> list[KalshiMarket]:
        # Paper mode returns no live quotes — caller should inject market data
        return []

    def _log(self, fill: Fill, order: Order) -> None:
        """Append fill to the paper trade log (JSONL)."""
        record = {
            "ticker": fill.ticker,
            "side": fill.side,
            "count": fill.count,
            "fill_price": fill.fill_price,
            "simulated": fill.simulated,
            "timestamp": fill.timestamp,
            "mode": order.mode,
        }
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")


class LiveKalshiClient:
    """Real Kalshi REST + WebSocket client.

    This class is intentionally unreachable except through the gated make_client().
    Credentials are taken from env at construction time — never stored in plain text.

    # TODO: wire Kalshi REST/WS — implement place() and quotes() against Kalshi v2 API
    # using requests for REST and websockets for streaming quotes.
    # Do NOT import requests/websockets at module level — keep them lazy so the test
    # suite (which targets pure cores and PaperKalshiClient) never needs them.
    """

    def __init__(self) -> None:
        # Credentials from env only — never logged
        self._api_key = os.environ["KALSHI_API_KEY"]
        self._api_secret = os.environ["KALSHI_API_SECRET"]

    def place(self, order: Order) -> Fill:
        # TODO: wire Kalshi REST/WS
        raise NotImplementedError("LiveKalshiClient.place() not yet implemented")

    def quotes(self) -> list[KalshiMarket]:
        # TODO: wire Kalshi REST/WS
        raise NotImplementedError("LiveKalshiClient.quotes() not yet implemented")


def make_client(mode: str = "paper") -> KalshiClient:
    """Factory: returns the correct client for the given mode.

    For "paper" (the default): returns PaperKalshiClient with no guards.
    For "live": raises RuntimeError unless SPEECHEDGE_ALLOW_LIVE=1 AND keys are present.

    This is the ONLY entry point to LiveKalshiClient (E4 hard gate).
    """
    if mode == "paper":
        return PaperKalshiClient()

    if mode == "live":
        allow_live = os.environ.get("SPEECHEDGE_ALLOW_LIVE", "").strip()
        if allow_live != "1":
            raise RuntimeError(
                "live trading requires explicit opt-in: set SPEECHEDGE_ALLOW_LIVE=1 "
                "in the environment. Paper mode is the default and requires no opt-in."
            )
        # Also require credentials to be present
        key = os.environ.get("KALSHI_API_KEY", "").strip()
        secret = os.environ.get("KALSHI_API_SECRET", "").strip()
        if not key or not secret:
            raise RuntimeError(
                "live trading requires KALSHI_API_KEY and KALSHI_API_SECRET "
                "to be set in the environment."
            )
        return LiveKalshiClient()

    raise ValueError(f"Unknown mode {mode!r}. Use 'paper' or 'live'.")

#!/usr/bin/env python3
"""kalshi_trader.py — Kalshi paper/live trader CLI for SpeechEdge.

Usage:
    python kalshi_trader.py                 # paper mode (default)
    SPEECHEDGE_ALLOW_LIVE=1 \\
      KALSHI_API_KEY=... KALSHI_API_SECRET=... \\
      python kalshi_trader.py               # live mode (explicit opt-in required)

Config via environment (INTERFACES.md §5):
    SPEECHEDGE_MODE        paper (default) | live
    SIGNAL_THRESHOLD       0.08 (default)
    KALSHI_FEE             0.0  (default; set to realistic Kalshi fee before live use)
    BANKROLL               unset (Kelly sizer refuses live sizing without this)
    SPEECHEDGE_ALLOW_LIVE  must be "1" for live mode

Credentials (live only, never committed):
    KALSHI_API_KEY
    KALSHI_API_SECRET

Pipeline per signal:
    Signal → passes_gate? → size_order? → PositionManager.accept? → client.place

Every signal is logged with timestamped model-vs-market divergence (Brief §5 June-16).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone

# NOTE: keep all trading imports after sys.path setup (conftest handles this in tests)
from src.trading.client import make_client
from src.trading.gate import passes_gate
from src.trading.positions import PositionManager
from src.trading.sizing import size_order
from src.trading.types import Signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("kalshi_trader")


def _load_config() -> dict:
    """Load runtime config from environment. Credentials are NEVER logged."""
    mode = os.environ.get("SPEECHEDGE_MODE", "paper").strip()
    threshold = float(os.environ.get("SIGNAL_THRESHOLD", "0.08"))
    fee = float(os.environ.get("KALSHI_FEE", "0.0"))
    bankroll_raw = os.environ.get("BANKROLL", "").strip()
    bankroll = float(bankroll_raw) if bankroll_raw else None
    max_aggregate = float(os.environ.get("MAX_AGGREGATE", "10000"))
    if fee == 0.0:
        log.warning(
            "KALSHI_FEE is unset (0.0) — the G3 net-edge floor is NOT accounting for fees, "
            "so paper P&L will be overstated. Set KALSHI_FEE to the real Kalshi fee per "
            "contract before the June-16 run and before any live use."
        )
    return dict(mode=mode, threshold=threshold, fee=fee, bankroll=bankroll,
                max_aggregate=max_aggregate)


def _log_signal(signal: Signal, cfg: dict, gated: bool) -> None:
    """Log model-vs-market divergence for every signal (Brief §5 June-16 requirement)."""
    log.info(
        "SIGNAL ticker=%s model_prob=%.4f market_price=%.4f edge=%.4f side=%s "
        "threshold=%.4f fee=%.4f passed_gate=%s ts=%s",
        signal.ticker,
        signal.model_prob,
        signal.market_price,
        signal.edge,
        signal.side,
        cfg["threshold"],
        cfg["fee"],
        gated,
        signal.timestamp,
    )


def run_once(signals: list[Signal], cfg: dict | None = None) -> list:
    """Process a list of signals through the full pipeline.

    This is the testable core — separated from I/O so unit tests can call it directly.

    Returns a list of Fill objects for signals that were placed.
    """
    if cfg is None:
        cfg = _load_config()

    mode = cfg["mode"]
    threshold = cfg["threshold"]
    fee = cfg["fee"]
    bankroll = cfg["bankroll"]
    max_aggregate = cfg.get("max_aggregate", 10_000)

    client = make_client(mode=mode)
    pm = PositionManager(max_aggregate=max_aggregate)
    fills = []

    for signal in signals:
        gated = passes_gate(signal, threshold=threshold, fee=fee)
        _log_signal(signal, cfg, gated)

        if not gated:
            continue

        order = size_order(signal, bankroll=bankroll, mode=mode)
        if order is None:
            log.warning(
                "SKIPPED ticker=%s — size_order returned None "
                "(live mode without bankroll?)", signal.ticker
            )
            continue

        if not pm.accept(order):
            log.warning(
                "BLOCKED ticker=%s side=%s — position manager rejected "
                "(dedupe or aggregate cap)", signal.ticker, signal.side
            )
            continue

        fill = client.place(order)
        log.info(
            "PLACED ticker=%s side=%s count=%d fill_price=%.4f simulated=%s",
            fill.ticker, fill.side, fill.count, fill.fill_price, fill.simulated,
        )
        fills.append(fill)

    return fills


def main() -> None:
    """CLI entry point.

    In production this consumes signals from C6 (live presser pipeline).
    For the June-16 paper run, signals are loaded from output/signals.jsonl if present,
    or the trader exits cleanly with a reminder.
    """
    cfg = _load_config()
    log.info(
        "kalshi_trader starting | mode=%s threshold=%.4f fee=%.4f bankroll=%s",
        cfg["mode"],
        cfg["threshold"],
        cfg["fee"],
        cfg["bankroll"],
    )

    signals_path = "output/signals.jsonl"
    if not os.path.exists(signals_path):
        log.info("No signals file at %s — nothing to trade. "
                 "Pipe signals in from the C6 live presser pipeline.", signals_path)
        return

    signals: list[Signal] = []
    with open(signals_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # SAFETY: derive edge/side from model_prob vs market_price via the canonical
            # constructor. Never trust edge/side carried in the file — a stale or mismatched
            # side from upstream (C6) would otherwise open a position in the WRONG direction.
            signals.append(Signal.from_quote(
                ticker=d["ticker"],
                model_prob=d["model_prob"],
                market_price=d["market_price"],
                timestamp=d.get("timestamp", datetime.now(timezone.utc).isoformat()),
            ))

    fills = run_once(signals, cfg=cfg)
    log.info("Done. %d fills placed.", len(fills))


if __name__ == "__main__":
    main()

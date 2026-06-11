#!/usr/bin/env python3
"""live_predictor.py — C6 Live Predictor CLI for SpeechEdge.

Usage:
    python live_predictor.py [--priors output/priors/<date>.json] [--whisper]

Wires:
  - Transcriber (Deepgram default, Whisper alternate — both lazy-import SDKs)
  - C4 priors (output/priors/<date>.json)
  - C2 DictionModel (models/warsh_model.pkl or .json)
  - Market list with current prices (from Kalshi feed / config)
  - LivePredictor inference loop
  - Emitted Signals piped into C7 run_once

Config via environment (INTERFACES.md §5):
    SPEECHEDGE_MODE        paper (default) | live
    SIGNAL_THRESHOLD       0.08 (default)
    KALSHI_FEE             0.0  (default)
    BANKROLL               unset until E1 provided

Missing priors/model/audio → print guidance and exit 0.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("live_predictor")


# ---------------------------------------------------------------------------
# Injectable STT stubs — lazy-import heavy SDKs so tests never need them
# ---------------------------------------------------------------------------

class _DeepgramTranscriber:
    """Real-time STT via Deepgram (~300 ms latency). Default E3 audio source.

    # TODO: wire E3 audio source (Deepgram SDK connection + audio capture).
    """
    def words(self) -> Iterable[str]:
        try:
            import deepgram  # noqa: F401  # lazy import
        except ImportError:
            raise ImportError(
                "Deepgram SDK not installed. Run: pip install deepgram-sdk\n"
                "# TODO: wire E3 audio source"
            )
        # TODO: wire E3 audio source — yield words from Deepgram live stream
        raise NotImplementedError("# TODO: wire E3 audio source (Deepgram)")


class _WhisperTranscriber:
    """Local offline STT via OpenAI Whisper (higher latency, no API key needed).

    # TODO: wire E3 audio source (Whisper model load + mic capture).
    """
    def words(self) -> Iterable[str]:
        try:
            import whisper  # noqa: F401  # lazy import
        except ImportError:
            raise ImportError(
                "Whisper not installed. Run: pip install openai-whisper\n"
                "# TODO: wire E3 audio source"
            )
        # TODO: wire E3 audio source — yield words from Whisper mic stream
        raise NotImplementedError("# TODO: wire E3 audio source (Whisper)")


def make_deepgram_transcriber() -> _DeepgramTranscriber:
    """Return a Deepgram-backed Transcriber (lazy SDK import, no network at module load)."""
    return _DeepgramTranscriber()


def make_whisper_transcriber() -> _WhisperTranscriber:
    """Return a Whisper-backed Transcriber (lazy SDK import, no network at module load)."""
    return _WhisperTranscriber()


# ---------------------------------------------------------------------------
# Prior loading — join C4 priors with current prices + axis map
# ---------------------------------------------------------------------------

def load_market_states(priors_path: Path, prices: dict[str, float],
                       axes: dict[str, str]) -> list:
    """Join C4 priors (INTERFACES.md §4) with current prices and axis map.

    Args:
        priors_path: Path to a JSON file containing a list of Prior dicts.
        prices:      {ticker: yes_price} from Kalshi feed or config.
        axes:        {ticker: signal_axis} from docs/phrase_market_map.md or config.

    Returns:
        List of MarketState objects ready for LivePredictor.
    """
    from src.live.types import MarketState

    raw = json.loads(priors_path.read_text())
    states = []
    for item in raw:
        ticker = item["ticker"]
        if ticker not in prices:
            log.warning("No current price for %s — skipping", ticker)
            continue
        if ticker not in axes:
            log.warning("No signal axis for %s — skipping", ticker)
            continue
        states.append(MarketState(
            ticker=ticker,
            yes_price=prices[ticker],
            prior_prob=item["prior_prob"],
            signal_axis=axes[ticker],
        ))
    return states


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Load priors + C2 model + markets, stream Signals into C7 run_once."""
    import argparse
    from datetime import date

    parser = argparse.ArgumentParser(description="C6 Live Predictor — STT → inference → C7")
    parser.add_argument("--priors", type=Path,
                        default=Path(f"output/priors/{date.today().isoformat()}.json"),
                        help="C4 priors JSON (default: today's date)")
    parser.add_argument("--whisper", action="store_true",
                        help="Use Whisper (local) instead of Deepgram (default)")
    args = parser.parse_args()

    # --- Priors ---
    if not args.priors.exists():
        log.info("No priors file at %s — run C4 context_agent.py first.", args.priors)
        log.info("Example: python context_agent.py --date %s", date.today().isoformat())
        return

    # --- C2 Model ---
    model_path_pkl = Path("models/warsh_model.pkl")
    model_path_json = Path("models/warsh_model.json")
    if model_path_json.exists():
        from src.warsh.model import DictionModel
        model = DictionModel.load(model_path_json)
    elif model_path_pkl.exists():
        import pickle
        with open(model_path_pkl, "rb") as fh:
            model = pickle.load(fh)
    else:
        log.info("No trained model at models/warsh_model.json — run build_model.py first.")
        return

    # --- Market prices + axis map (placeholder — wire from C7 Kalshi feed) ---
    # TODO: fetch live prices from Kalshi REST/WS (C7). For now, use a stub map.
    prices: dict[str, float] = {}
    axes: dict[str, str] = {}

    markets = load_market_states(args.priors, prices, axes)
    if not markets:
        log.info("No markets after joining priors with prices/axes — check priors file.")
        return

    # --- Transcriber ---
    transcriber = make_whisper_transcriber() if args.whisper else make_deepgram_transcriber()

    # --- Inference loop ---
    from src.live.predictor import LivePredictor
    pred = LivePredictor(model, markets)

    log.info("Starting live predictor: %d markets, transcriber=%s",
             len(markets), "whisper" if args.whisper else "deepgram")

    signals = list(pred.run(transcriber))
    log.info("Inference complete: %d signals emitted", len(signals))

    # --- C7 hand-off ---
    from kalshi_trader import run_once
    fills = run_once(signals)
    log.info("C7 run_once complete: %d fills placed.", len(fills))


if __name__ == "__main__":
    main()

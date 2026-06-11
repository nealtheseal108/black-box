"""Dashboard state models and pure computation functions."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# P&L computation (pure)
# ---------------------------------------------------------------------------

def compute_pnl(fills: list[dict], resolutions: dict[str, int]) -> dict:
    """Kalshi binary contract: pays $1 if your side wins, $0 if it loses.
    realized P&L per fill = (1 - fill_price) on a win, -fill_price on a loss."""
    realized = 0.0
    open_positions = 0
    for f in fills:
        outcome = resolutions.get(f["ticker"])
        if outcome is None:
            open_positions += 1
            continue
        win = (f["side"] == "yes" and outcome == 1) or (f["side"] == "no" and outcome == 0)
        per = (1 - f["fill_price"]) if win else -f["fill_price"]
        realized += f["count"] * per
    return {"realized": round(realized, 4), "open_positions": open_positions}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ToneMeter(BaseModel):
    hawkish: float = 0.0
    dovish: float = 0.0
    independence: float = 0.0
    qt: float = 0.0


class MarketCard(BaseModel):
    ticker: str
    prior_prob: float = 0.0
    model_prob: float = 0.0
    yes_price: float = 0.0
    edge: float = 0.0
    side: str = ""


class BetLogEntry(BaseModel):
    ticker: str
    side: str
    count: int
    fill_price: float
    simulated: bool = True
    timestamp: str = ""


class PnLSummary(BaseModel):
    realized: float = 0.0
    open_positions: int = 0


class DashboardState(BaseModel):
    transcript: str = ""
    tone: ToneMeter = ToneMeter()
    markets: list[MarketCard] = []
    bets: list[BetLogEntry] = []
    pnl: PnLSummary = PnLSummary()


# ---------------------------------------------------------------------------
# State builder (pure)
# ---------------------------------------------------------------------------

def build_state(
    transcript: str,
    tone: dict[str, Any],
    markets: list[dict[str, Any]],
    fills: list[dict[str, Any]],
    resolutions: dict[str, int],
) -> DashboardState:
    """Assemble a DashboardState from raw dicts produced by the pipeline."""
    pnl_raw = compute_pnl(fills, resolutions)
    return DashboardState(
        transcript=transcript,
        tone=ToneMeter(**tone),
        markets=[MarketCard(**m) for m in markets],
        bets=[BetLogEntry(**f) for f in fills],
        pnl=PnLSummary(**pnl_raw),
    )

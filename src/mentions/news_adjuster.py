"""News adjuster: bounded per-term multipliers on the statistical base rate.

The LLM (real implementation lands in a later plan) can SCALE a base rate within
[0.25, 4.0] to reflect current news — e.g. "today's banking stress makes 'financial
stability' 4x more likely" — but can NEVER invent a probability from nothing. This
clamp is the guardrail that keeps a hallucinated multiplier from blowing up a position.

NullAdjuster is the offline default (all multipliers = 1.0) used in deterministic
tests and the calibration backtest.
"""
from __future__ import annotations

from typing import Protocol

MULT_MIN = 0.25
MULT_MAX = 4.0


class NewsAdjuster(Protocol):
    def multipliers(self, terms: list[str], news_context: str) -> dict[str, float]:
        """Return a raw (pre-clamp) multiplier per canonical term."""
        ...


def clamp_multiplier(m: float) -> float:
    return max(MULT_MIN, min(MULT_MAX, m))


def apply_multipliers(base_rates: dict[str, float], raw_mults: dict[str, float]) -> dict[str, float]:
    """Scale each base rate by its clamped multiplier (default 1.0), clip to [0, 1]."""
    out: dict[str, float] = {}
    for term, p in base_rates.items():
        m = clamp_multiplier(raw_mults.get(term, 1.0))
        out[term] = max(0.0, min(1.0, p * m))
    return out


class NullAdjuster:
    """No-op adjuster — every term gets a neutral 1.0 multiplier."""

    def multipliers(self, terms: list[str], news_context: str) -> dict[str, float]:
        return {t: 1.0 for t in terms}

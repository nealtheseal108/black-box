"""Pydantic models for C4 Mode-1 Context Agent.

Matches INTERFACES.md §4 — Prior distribution (Mode 1 C4 → C6).
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class Prior(BaseModel):
    """Per-market prior probability emitted by C4.

    Matches the JSON contract in INTERFACES.md §4:
        {"ticker": str, "prior_prob": float, "rationale": str,
         "as_of": "ISO8601", "sources": [str]}
    """
    ticker: str
    prior_prob: float = Field(ge=0.0, le=1.0)
    rationale: str
    as_of: str
    sources: List[str]


class MarketContext(BaseModel):
    """A Kalshi market passed into the synthesis step."""
    ticker: str
    title: str
    question: str
    yes_price: float = Field(ge=0.0, le=1.0)


class MacroSnapshot(BaseModel):
    """Assembled macro context for a pre-speech run."""
    as_of: str
    data_prints: List[str] = Field(default_factory=list)
    futures: List[str] = Field(default_factory=list)
    news: List[str] = Field(default_factory=list)
    speaker_recent: List[str] = Field(default_factory=list)

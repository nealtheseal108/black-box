"""Market mention terms and the surface patterns that count as 'saying' them.

A Kalshi/Polymarket mention sub-market resolves YES if the speaker utters a term
in any of its phrasings. MarketTerm bundles the canonical label with the regex
variants that all resolve it ("rate cut" / "cut rate" / "cut the policy rate").
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MarketTerm:
    canonical: str
    patterns: tuple[str, ...]

    def mentioned_in(self, text: str) -> bool:
        low = text.lower()
        return any(re.search(p, low) for p in self.patterns)


def load_terms(path: str | Path) -> list[MarketTerm]:
    raw = json.loads(Path(path).read_text())
    return [MarketTerm(canonical=t["canonical"], patterns=tuple(t["patterns"])) for t in raw]

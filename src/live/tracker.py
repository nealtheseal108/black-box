"""LiveMentionTracker — the Mode-2 streaming state machine.

Seeds each term's log-odds from its Mode-1 prior, then per caption delta:
  1. RESOLVE: a term whose pattern appears in the running text -> P=1, retired.
  2. PROPAGATE (fire-once): an evidence term seen for the FIRST time adds its
     frozen w(A->B) to every unsaid term's log-odds. Repeats do not re-fire.
Inference-only: weights are frozen; only the per-term log-odds (beliefs) move.
"""
from __future__ import annotations

from src.mentions.terms import MarketTerm
from src.live.inference import to_logodds, from_logodds


class LiveMentionTracker:
    def __init__(
        self,
        terms: list[MarketTerm],
        priors: dict[str, float],
        weights: dict[tuple[str, str], float],
        evidence_terms: list[MarketTerm],
    ) -> None:
        self._terms = terms
        self._weights = weights
        self._evidence_terms = evidence_terms
        self._logodds = {t.canonical: to_logodds(priors[t.canonical]) for t in terms}
        self._resolved: set[str] = set()
        self._fired: set[str] = set()
        self._running = ""

    def consume(self, delta: str) -> None:
        self._running = (self._running + " " + delta).strip()
        for t in self._terms:
            if t.canonical not in self._resolved and t.mentioned_in(self._running):
                self._resolved.add(t.canonical)
        for a in self._evidence_terms:
            if a.canonical in self._fired:
                continue
            if a.mentioned_in(self._running):
                self._fired.add(a.canonical)
                for b in self._terms:
                    if b.canonical in self._resolved or b.canonical == a.canonical:
                        continue
                    self._logodds[b.canonical] += self._weights.get((a.canonical, b.canonical), 0.0)

    def probabilities(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for t in self._terms:
            c = t.canonical
            out[c] = 1.0 if c in self._resolved else from_logodds(self._logodds[c])
        return out

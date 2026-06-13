"""LiveMentionTracker — the Mode-2 streaming state machine.

Seeds each term's log-odds from its Mode-1 prior, then per caption delta:
  1. RESOLVE: a term whose pattern appears in the running text -> P=1, retired.
  2. PROPAGATE (fire-once): an evidence term seen for the FIRST time adds its
     frozen w(A->B) to every unsaid term's log-odds. Repeats do not re-fire.
Inference-only: weights are frozen; only the per-term log-odds (beliefs) move.

The CUMULATIVE propagation shift per term is capped at +/- prop_cap log-odds.
Without this, a content-rich event (a 9k-word hearing touches ~20 of 40 lexicon
terms) sums many small positive co-occurrence shifts and saturates unsaid terms
toward 1.0 — confidently "predicting" words never spoken. The cap is the aggregate
analogue of the per-edge clamp.
"""
from __future__ import annotations

from src.mentions.terms import MarketTerm
from src.live.inference import to_logodds, from_logodds

DEFAULT_PROP_CAP = 2.0


class LiveMentionTracker:
    def __init__(
        self,
        terms: list[MarketTerm],
        priors: dict[str, float],
        weights: dict[tuple[str, str], float],
        evidence_terms: list[MarketTerm],
        prop_cap: float = DEFAULT_PROP_CAP,
    ) -> None:
        self._terms = terms
        self._weights = weights
        self._evidence_terms = evidence_terms
        self._prop_cap = prop_cap
        self._logodds = {t.canonical: to_logodds(priors[t.canonical]) for t in terms}
        self._prop_shift = {t.canonical: 0.0 for t in terms}  # cumulative propagation applied
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
                    w = self._weights.get((a.canonical, b.canonical), 0.0)
                    # Apply only the portion that keeps cumulative shift within +/- prop_cap.
                    prev = self._prop_shift[b.canonical]
                    capped = max(-self._prop_cap, min(self._prop_cap, prev + w))
                    self._logodds[b.canonical] += capped - prev
                    self._prop_shift[b.canonical] = capped

    def probabilities(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for t in self._terms:
            c = t.canonical
            out[c] = 1.0 if c in self._resolved else from_logodds(self._logodds[c])
        return out

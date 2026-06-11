"""DictionModel — C2 facade over NgramModel, fingerprints, and lexicon.

Inference-only at runtime (Appendix A.3 — no weight updates per phrase).
Public API matches INTERFACES.md §2:

    predict_next(context_tokens: list[str], k: int = 5) -> list[tuple[str, float]]
    phrase_signals(text: str) -> list[dict]   # [{phrase, axis, weight}, ...]
    score_diction(text: str) -> dict[str, float]  # {hawkish, dovish, independence, qt}
"""
from __future__ import annotations

import json
from pathlib import Path

from src.warsh.ngram import NgramModel
from src.warsh.tokenize import tokenize
from src.warsh.lexicon import phrase_signals as _phrase_signals, score_diction as _score_diction
from src.warsh.fingerprints import top_fingerprints, BASELINE_UNIGRAMS


class DictionModel:
    def __init__(self, max_n: int = 4):
        self.max_n = max_n
        self._ngram = NgramModel(max_n=max_n)
        self._fingerprints: list[tuple[str, float]] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, docs: list) -> "DictionModel":
        """Train on a list of dicts or Document objects that have a .text / ["text"] field."""
        token_lists = []
        for doc in docs:
            text = doc["text"] if isinstance(doc, dict) else doc.text
            toks = tokenize(text)
            if toks:
                token_lists.append(toks)

        self._ngram.train(token_lists)

        # Compute bigram fingerprints vs seed baseline
        if token_lists:
            self._fingerprints = top_fingerprints(
                token_lists,
                baseline_counts=BASELINE_UNIGRAMS,
                baseline_total=sum(BASELINE_UNIGRAMS.values()),
                n=2,
                k=50,
            )
        return self

    # ------------------------------------------------------------------
    # INTERFACES §2 public API
    # ------------------------------------------------------------------

    def predict_next(self, context_tokens: list[str], k: int = 5) -> list[tuple[str, float]]:
        """Return top-k (word, prob) continuations for context_tokens."""
        return self._ngram.predict_next(context_tokens, k=k)

    def phrase_signals(self, text: str) -> list[dict]:
        """Return list of {phrase, axis, weight} for all matching signal phrases in text."""
        return _phrase_signals(text)

    def score_diction(self, text: str) -> dict[str, float]:
        """Return {hawkish, dovish, independence, qt} axis scores for text."""
        return _score_diction(text)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "max_n": self.max_n,
            "ngram": self._ngram.to_dict(),
            "fingerprints": self._fingerprints,
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "DictionModel":
        path = Path(path)
        payload = json.loads(path.read_text())
        m = cls(max_n=payload["max_n"])
        m._ngram = NgramModel.from_dict(payload["ngram"])
        m._fingerprints = [tuple(x) for x in payload.get("fingerprints", [])]
        return m

    # ------------------------------------------------------------------
    # Helpers for build_model.py reporting
    # ------------------------------------------------------------------

    def top_fingerprints(self, k: int = 20) -> list[tuple[str, float]]:
        return self._fingerprints[:k]

#!/usr/bin/env python3
"""build_model.py — C2 Diction Model Training CLI

Usage:
    python build_model.py

Loads corpus/warsh_corpus.jsonl, trains a DictionModel, writes
models/warsh_model.json, and prints a fingerprint/lift report for review.

Handles empty/missing corpus gracefully.
"""
import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from src.warsh.model import DictionModel

CORPUS_PATH = Path("corpus/warsh_corpus.jsonl")
MODEL_PATH = Path("models/warsh_model.json")

# TODO (stretch goal): build a rhetorical-move transition matrix from the corpus.
# Identify sentence-level rhetorical moves (e.g. premise → claim → hedge) via
# keyword classifiers, count transitions, and store as a Markov matrix in the
# model artifact. C5 can use it to detect unusual move sequences as signals.
# Not blocking G1 — skip until fingerprint accuracy is measured.


def load_corpus(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[build_model] Corpus not found at {path}. "
              "Run the C1 scraper first to populate corpus/warsh_corpus.jsonl.")
        return []
    docs = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[build_model] Skipping malformed line {lineno}: {e}")
    return docs


def print_fingerprint_report(model: DictionModel, top_k: int = 30) -> None:
    fps = model.top_fingerprints(top_k)
    if not fps:
        print("[build_model] No fingerprints computed (corpus may be empty).")
        return
    print(f"\n{'='*60}")
    print(f"  TOP {top_k} WARSH PHRASE FINGERPRINTS (bigrams, by lift)")
    print(f"{'='*60}")
    print(f"{'Phrase':<35}  {'Lift':>10}")
    print(f"{'-'*35}  {'-'*10}")
    for phrase, lift in fps:
        print(f"{phrase:<35}  {lift:>10.1f}x")
    print()


def main() -> None:
    print("[build_model] Loading corpus ...")
    docs = load_corpus(CORPUS_PATH)

    if not docs:
        print("[build_model] Nothing to train on — exiting. "
              "Add documents to corpus/warsh_corpus.jsonl and re-run.")
        sys.exit(0)

    print(f"[build_model] Training on {len(docs)} documents ...")
    model = DictionModel(max_n=4)
    model.train(docs)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"[build_model] Model saved to {MODEL_PATH}")

    print_fingerprint_report(model, top_k=30)

    # Quick smoke-test: predict_next on a seed phrase
    demo_ctx = ["inflation", "is", "a"]
    preds = model.predict_next(demo_ctx, k=5)
    print(f"predict_next({demo_ctx!r}, k=5):")
    for word, prob in preds:
        print(f"  {word:<20}  {prob:.4f}")


if __name__ == "__main__":
    main()

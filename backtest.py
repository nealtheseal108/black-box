"""backtest.py — Two-level validation harness (Brief §6, Appendix A.4).

Level 1 (diction): held-out Warsh corpus → G1 top-1 phrase accuracy (>20%).
Level 2 (market mechanics): Powell pressers → G2 Brier calibration (<0.22)
                             and G3 market edge (>55% positive, avg >$0.06 after fees).

Usage:
    python backtest.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.backtest.splits import doc_level_split
from src.backtest.metrics import top1_accuracy, brier_score, market_edge_stats
from src.backtest.gates import evaluate_gates
from src.warsh.model import DictionModel


# ---------------------------------------------------------------------------
# Level 1 — diction model G1
# ---------------------------------------------------------------------------

def run_level1(docs: list[dict], test_frac: float = 0.2, seed: int = 0) -> float:
    """Split corpus, train DictionModel on train split, return top-1 accuracy on test split."""
    train, test = doc_level_split(docs, test_frac=test_frac, seed=seed)
    model = DictionModel()
    model.train(train)
    return top1_accuracy(model, test)


# ---------------------------------------------------------------------------
# Level 2 — market mechanics G2/G3
# ---------------------------------------------------------------------------

def run_level2(loader, fee: float = 0.02) -> tuple[dict, dict]:
    """Run Level-2 validation using the provided loader callable.

    loader() must return:
        {"forecasts": list[float], "outcomes": list[int],
         "signal_outcomes": list[{"edge": float}]}

    Returns (g2_metrics, g3_metrics).
    """
    data = loader()
    g2 = {"brier": brier_score(data["forecasts"], data["outcomes"])}
    g3 = market_edge_stats(data["signal_outcomes"], fee=fee)
    return g2, g3


# ---------------------------------------------------------------------------
# Stub Powell loader (owner provides real transcripts + Kalshi prices)
# ---------------------------------------------------------------------------

def _powell_stub_loader():
    # TODO: wire Powell transcripts + historical Kalshi prices
    return {"forecasts": [], "outcomes": [], "signal_outcomes": []}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    corpus_path = Path("corpus/warsh_corpus.jsonl")

    # --- Level 1 ---
    if not corpus_path.exists():
        print("INFO: corpus/warsh_corpus.jsonl not found — run scrape_warsh.py first.")
        print("Skipping Level-1 (G1) evaluation.")
        g1_accuracy = 0.0
    else:
        lines = corpus_path.read_text().strip().splitlines()
        docs = [json.loads(l) for l in lines if l.strip()]
        if not docs:
            print("INFO: corpus/warsh_corpus.jsonl is empty — skipping Level-1.")
            g1_accuracy = 0.0
        else:
            print(f"Level-1: training on {len(docs)} documents …")
            g1_accuracy = run_level1(docs)
            print(f"  G1 top-1 accuracy: {g1_accuracy:.4f}")

    # --- Level 2 ---
    g2, g3 = run_level2(_powell_stub_loader)
    print(f"Level-2 (stub loader): G2 brier={g2['brier']:.4f}  G3 n={g3['n']}")

    # --- Gate report ---
    metrics = {
        "g1_accuracy": g1_accuracy,
        "g2_brier": g2["brier"],
        "g3_positive_rate": g3["positive_rate"],
        "g3_avg_net_edge": g3["avg_net_edge"],
    }
    report = evaluate_gates(metrics)
    print("\n=== Gate Report ===")
    for key in ("g1", "g2", "g3"):
        print(f"  {key.upper()}: {'PASS' if report[key]['pass'] else 'FAIL'}")
    print(f"  ALL PASS: {report['all_pass']}")
    if not report["all_pass"]:
        print("  ⚠  No live capital until G1–G4 pass.")


if __name__ == "__main__":
    main()

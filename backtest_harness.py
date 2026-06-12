"""Level-1 calibration harness — Warsh leave-one-event-out over the curated lexicon.

Turns mention-model validation from n=1 into ~25 scored events. Reports Brier,
log-loss, AUC, and a reliability diagram over the pooled (prediction, outcome) pairs.

Usage: python backtest_harness.py
"""
from __future__ import annotations

from pathlib import Path

from src.backtest.events import JsonlCorpusLoader
from src.backtest.vocabulary import LexiconVocabulary
from src.backtest.loo import leave_one_out
from src.backtest.calibration import calibration_summary

CORPUS = Path("corpus/warsh_corpus.jsonl")
LEXICON = Path("corpus/lexicon/fed_macro_terms.json")


def run_harness(min_prior_docs: int = 3, half_life_years: float = 4.0) -> dict:
    docs = JsonlCorpusLoader(CORPUS).docs_for("warsh")
    vocab = LexiconVocabulary(LEXICON)
    loo = leave_one_out(docs, "warsh", vocab, min_prior_docs, half_life_years)
    return {
        "summary": calibration_summary(loo["pool"]),
        "n_scored": loo["n_scored"],
        "n_skipped": loo["n_skipped"],
    }


def main() -> None:
    r = run_harness()
    s = r["summary"]
    auc_str = f"{s['auc']:.4f}" if s["auc"] is not None else "n/a"
    print(f"events scored: {r['n_scored']}   skipped: {r['n_skipped']}")
    print(f"pool: {s['n']} (pred,outcome) pairs   base rate: {s['base_rate']:.3f}")
    print(f"Brier {s['brier']:.4f}   log-loss {s['log_loss']:.4f}   AUC {auc_str}")
    print("reliability (predicted vs observed):")
    print(f"  {'bin':>10} {'n':>5} {'mean_pred':>10} {'observed':>9}")
    for b in s["reliability"]:
        if b["count"]:
            print(f"  [{b['lo']:.1f},{b['hi']:.1f}) {b['count']:>5} "
                  f"{b['mean_pred']:>10.3f} {b['observed']:>9.3f}")


if __name__ == "__main__":
    main()

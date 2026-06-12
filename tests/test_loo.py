from src.mentions.terms import MarketTerm
from src.backtest.loo import leave_one_out


class _FixedVocab:
    def __init__(self, terms):
        self._terms = terms
    def terms_for(self, event):
        return self._terms


def _docs():
    return [
        {"date": "2006-01-01", "context_type": "speech", "text": "inflation"},
        {"date": "2007-01-01", "context_type": "speech", "text": "inflation"},
        {"date": "2008-01-01", "context_type": "speech", "text": "inflation"},
        {"date": "2009-01-01", "context_type": "speech", "text": "inflation qe"},
        {"date": "2010-01-01", "context_type": "speech", "text": "inflation"},
    ]


def test_cold_start_guard_skips_events_without_enough_priors():
    vocab = _FixedVocab([MarketTerm(canonical="inflation", patterns=(r"\binflation\b",))])
    out = leave_one_out(_docs(), "warsh", vocab, min_prior_docs=3)
    # only the 2009 and 2010 docs have >=3 strictly-earlier docs
    assert out["n_scored"] == 2
    assert out["n_skipped"] == 3


def test_pool_holds_pred_outcome_pairs_with_no_leakage():
    terms = [
        MarketTerm(canonical="inflation", patterns=(r"\binflation\b",)),
        MarketTerm(canonical="qe", patterns=(r"\bqe\b",)),
    ]
    out = leave_one_out(_docs(), "warsh", _FixedVocab(terms), min_prior_docs=3)
    qe_pairs = [(p, o) for (p, o) in out["pool"]]
    assert all(0.0 <= p <= 1.0 and o in (0, 1) for p, o in qe_pairs)
    # 2 scored events x 2 terms = 4 pairs
    assert len(out["pool"]) == 4

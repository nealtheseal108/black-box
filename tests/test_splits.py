from src.backtest.splits import doc_level_split

DOCS = [{"id": f"d{i}", "text": f"doc {i} text"} for i in range(10)]

def test_split_is_disjoint_and_deterministic():
    tr1, te1 = doc_level_split(DOCS, test_frac=0.3, seed=0)
    tr2, te2 = doc_level_split(DOCS, test_frac=0.3, seed=0)
    assert [d["id"] for d in te1] == [d["id"] for d in te2]   # deterministic
    train_ids = {d["id"] for d in tr1}
    test_ids = {d["id"] for d in te1}
    assert train_ids.isdisjoint(test_ids)                     # no doc in both — no leakage
    assert len(te1) == 3 and len(tr1) == 7

def test_split_covers_all_docs():
    tr, te = doc_level_split(DOCS, test_frac=0.2, seed=1)
    assert len(tr) + len(te) == len(DOCS)

import random


def doc_level_split(docs: list[dict], test_frac: float = 0.2, seed: int = 0):
    """Split at the DOCUMENT level (never sentence-level) so no text from one document
    appears in both train and test — the invariant that keeps G1 honest (Appendix A.4)."""
    idx = list(range(len(docs)))
    random.Random(seed).shuffle(idx)
    n_test = int(round(len(docs) * test_frac))
    test_idx = set(idx[:n_test])
    train = [docs[i] for i in range(len(docs)) if i not in test_idx]
    test = [docs[i] for i in range(len(docs)) if i in test_idx]
    return train, test

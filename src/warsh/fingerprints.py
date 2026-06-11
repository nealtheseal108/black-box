from collections import Counter


# Seed baseline unigram/bigram counts (approximate English frequencies).
# Users can swap in a fuller table — these are rough orders of magnitude.
BASELINE_UNIGRAMS: dict[str, int] = {
    "the": 69971597, "of": 36412950, "and": 32483827, "to": 30775246,
    "in": 22996460, "a": 22341763, "is": 12476684, "that": 11993832,
    "for": 8993892, "it": 8844145, "as": 8431231, "was": 8005255,
    "with": 7993700, "be": 7622895, "by": 7361823, "on": 7254460,
    "not": 6845786, "he": 6743701, "this": 6643452, "are": 6601615,
}


def _ngrams(tokens: list[str], n: int):
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def phrase_lift(phrase, corpus_counts, corpus_total, baseline_counts, baseline_total) -> float:
    p_corpus = corpus_counts.get(phrase, 0) / corpus_total if corpus_total else 0.0
    # add-one smoothing on baseline so unseen baseline phrases don't divide by zero
    p_base = (baseline_counts.get(phrase, 0) + 1) / (baseline_total + 1)
    return p_corpus / p_base if p_base else 0.0


def top_fingerprints(docs_tokens, baseline_counts, baseline_total, n=2, k=20):
    counts = Counter()
    for toks in docs_tokens:
        counts.update(_ngrams(toks, n))
    total = sum(counts.values()) or 1
    scored = [(ph, phrase_lift(ph, counts, total, baseline_counts, baseline_total))
              for ph in counts]
    return sorted(scored, key=lambda x: -x[1])[:k]

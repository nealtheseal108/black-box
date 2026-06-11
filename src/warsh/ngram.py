from collections import defaultdict


class NgramModel:
    def __init__(self, max_n: int = 4, backoff: float = 0.4):
        self.max_n = max_n
        self.backoff = backoff
        # counts[n][context_tuple][next_word] -> int
        self.counts = [defaultdict(lambda: defaultdict(int)) for _ in range(max_n)]

    def train(self, token_lists: list[list[str]]) -> "NgramModel":
        for toks in token_lists:
            for i in range(len(toks)):
                for n in range(1, self.max_n + 1):
                    if i - (n - 1) < 0:
                        continue
                    ctx = tuple(toks[i - (n - 1):i])
                    self.counts[n - 1][ctx][toks[i]] += 1
        return self

    def _scores(self, context: list[str]) -> dict[str, float]:
        # Stupid-backoff: highest available order, discounted by backoff^drop.
        scores: dict[str, float] = {}
        for drop in range(self.max_n):
            n = self.max_n - drop
            ctx = tuple(context[-(n - 1):]) if n > 1 else tuple()
            table = self.counts[n - 1].get(ctx)
            if not table:
                continue
            total = sum(table.values())
            weight = self.backoff ** drop
            for w, c in table.items():
                scores.setdefault(w, weight * c / total)
            if scores:
                break
        return scores

    def predict_next(self, context: list[str], k: int = 5) -> list[tuple[str, float]]:
        scores = self._scores(context)
        if not scores:
            return []
        total = sum(scores.values())
        ranked = sorted(((w, s / total) for w, s in scores.items()), key=lambda x: -x[1])
        return ranked[:k]

    def to_dict(self) -> dict:
        return {"max_n": self.max_n, "backoff": self.backoff,
                "counts": [{" ".join(ctx): dict(tbl) for ctx, tbl in order.items()} for order in self.counts]}

    @classmethod
    def from_dict(cls, d: dict) -> "NgramModel":
        m = cls(max_n=d["max_n"], backoff=d["backoff"])
        for n, order in enumerate(d["counts"]):
            for ctx_str, tbl in order.items():
                ctx = tuple(ctx_str.split()) if ctx_str else tuple()
                for w, c in tbl.items():
                    m.counts[n][ctx][w] = c
        return m

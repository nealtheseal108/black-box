from src.warsh.tokenize import tokenize


def top1_accuracy(model, test_docs: list[dict], min_context: int = 1) -> float:
    correct = total = 0
    for doc in test_docs:
        toks = tokenize(doc["text"])
        for i in range(min_context, len(toks)):
            preds = model.predict_next(toks[:i], k=1)
            if not preds:
                total += 1
                continue
            if preds[0][0] == toks[i]:
                correct += 1
            total += 1
    return correct / total if total else 0.0


def brier_score(forecasts: list[float], outcomes: list[int]) -> float:
    if not forecasts:
        return 0.0
    return sum((f - o) ** 2 for f, o in zip(forecasts, outcomes)) / len(forecasts)


def market_edge_stats(signal_outcomes: list[dict], fee: float) -> dict:
    nets = [abs(s["edge"]) - fee for s in signal_outcomes]  # net edge magnitude after fee
    n = len(nets)
    if n == 0:
        return {"n": 0, "avg_net_edge": 0.0, "positive_rate": 0.0, "above_floor_rate": 0.0}
    from src.trading.gate import G3_NET_EDGE_FLOOR
    return {
        "n": n,
        "avg_net_edge": sum(nets) / n,
        "positive_rate": sum(1 for x in nets if x > 0) / n,
        "above_floor_rate": sum(1 for x in nets if x > G3_NET_EDGE_FLOOR) / n,
    }

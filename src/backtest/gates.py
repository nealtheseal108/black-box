def evaluate_gates(m: dict) -> dict:
    g1 = m["g1_accuracy"] > 0.20
    g2 = m["g2_brier"] < 0.22
    g3 = m["g3_positive_rate"] > 0.55 and m["g3_avg_net_edge"] > 0.06
    report = {
        "g1": {"pass": g1, "metric": m["g1_accuracy"], "threshold": "> 0.20"},
        "g2": {"pass": g2, "metric": m["g2_brier"], "threshold": "< 0.22"},
        "g3": {"pass": g3, "positive_rate": m["g3_positive_rate"],
               "avg_net_edge": m["g3_avg_net_edge"], "threshold": "> 0.55 & > 0.06"},
    }
    report["all_pass"] = g1 and g2 and g3
    return report

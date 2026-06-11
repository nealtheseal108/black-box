import re

# axis in {hawkish, dovish, independence, qt}; weight is signal strength 0..1
SIGNAL_LEXICON: dict[str, dict] = {
    "inflation is a choice":          {"axis": "hawkish",      "weight": 0.9},
    "without excuse or equivocation": {"axis": "hawkish",      "weight": 1.0},
    "foursquare within its role":     {"axis": "independence",  "weight": 0.8},
    "forays far afield":              {"axis": "hawkish",      "weight": 0.6},
    "misallocation of capital":       {"axis": "qt",           "weight": 0.7},
    "bloated balance sheet":          {"axis": "qt",           "weight": 0.8},
    "regime change":                  {"axis": "independence",  "weight": 0.6},
    "fiscal dominance":               {"axis": "independence",  "weight": 0.9},
    "stronger not hotter":            {"axis": "dovish",       "weight": 0.7},
}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z ]+", " ", text.lower())


def phrase_signals(text: str) -> list[dict]:
    t = _norm(text)
    out = []
    for phrase, meta in SIGNAL_LEXICON.items():
        if phrase in t:
            out.append({"phrase": phrase, "axis": meta["axis"], "weight": meta["weight"]})
    return out


def score_diction(text: str) -> dict[str, float]:
    scores = {"hawkish": 0.0, "dovish": 0.0, "independence": 0.0, "qt": 0.0}
    for s in phrase_signals(text):
        scores[s["axis"]] += s["weight"]
    return scores

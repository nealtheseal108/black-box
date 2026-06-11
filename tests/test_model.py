from pathlib import Path
from src.warsh.model import DictionModel

DOCS = [
    {"text": "Inflation is a choice. The balance sheet is bloated."},
    {"text": "We remain foursquare within its role. Inflation is a choice."},
]

def test_train_then_predict_and_score():
    m = DictionModel().train(DOCS)
    assert m.predict_next(["inflation", "is", "a"], k=1)  # non-empty
    s = m.score_diction("inflation is a choice")
    assert s["hawkish"] > 0
    assert {"phrase", "axis", "weight"} <= set(m.phrase_signals("fiscal dominance")[0].keys()) \
        if m.phrase_signals("fiscal dominance") else True

def test_save_load_roundtrip(tmp_path: Path):
    m = DictionModel().train(DOCS)
    p = tmp_path / "model.json"
    m.save(p)
    m2 = DictionModel.load(p)
    assert m2.predict_next(["inflation", "is", "a"], k=1) == m.predict_next(["inflation", "is", "a"], k=1)

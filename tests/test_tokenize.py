from src.warsh.tokenize import tokenize, sentences

def test_tokenize_lowercases_and_keeps_words():
    assert tokenize("Inflation is a CHOICE.") == ["inflation", "is", "a", "choice"]

def test_tokenize_drops_pure_punctuation_but_keeps_hyphenated():
    assert tokenize("stronger, not hotter — really?") == ["stronger", "not", "hotter", "really"]
    assert tokenize("balance-sheet runoff") == ["balance-sheet", "runoff"]

def test_sentences_splits_on_terminal_punctuation():
    s = sentences("Inflation is a choice. Without excuse. Really?")
    assert s == ["Inflation is a choice.", "Without excuse.", "Really?"]

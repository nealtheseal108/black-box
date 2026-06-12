from src.mentions.terms import MarketTerm, load_terms


def test_term_matches_any_variant_pattern():
    term = MarketTerm(canonical="rate cut", patterns=("rate cut", "cut rate", r"cut\w*(?:\s+\w+)*\s+rates?"))
    assert term.mentioned_in("we may need a rate cut soon") is True
    assert term.mentioned_in("the committee could cut the policy rate") is True   # variant
    assert term.mentioned_in("inflation remains elevated") is False


def test_term_matching_is_case_insensitive():
    term = MarketTerm(canonical="trump", patterns=(r"\btrump\b",))
    assert term.mentioned_in("President TRUMP said") is True
    assert term.mentioned_in("they trumpeted the result") is False   # word boundary


def test_load_terms_from_json(tmp_path):
    p = tmp_path / "terms.json"
    p.write_text('[{"canonical": "trump", "patterns": ["\\\\btrump\\\\b"]}]')
    terms = load_terms(p)
    assert len(terms) == 1
    assert terms[0].canonical == "trump"
    assert terms[0].mentioned_in("Trump") is True

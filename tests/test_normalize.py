from src.warsh.normalize import clean_html, normalize_whitespace, word_count, tag_context_type


def test_clean_html_strips_tags_scripts_and_styles():
    html = """
    <html><head><style>.x{color:red}</style></head>
    <body><script>evil()</script>
    <p>Inflation is <b>a choice</b>.</p>
    <p>Without excuse or equivocation.</p></body></html>
    """
    out = clean_html(html)
    assert "evil" not in out
    assert "color:red" not in out
    assert "Inflation is a choice." in out
    assert "Without excuse or equivocation." in out


def test_normalize_whitespace_collapses_runs_and_trims():
    assert normalize_whitespace("  a\n\n\n  b\t\tc  ") == "a\n\nb c"


def test_word_count_counts_tokens():
    assert word_count("one two   three\nfour") == 4


def test_tag_context_type_classifies_by_source_and_title():
    assert tag_context_type("hearing", "Nomination Hearing") == "hearing"
    assert tag_context_type("fraser", "Remarks on Liquidity") == "speech"
    assert tag_context_type("wsj_archive", "The Fed's Mission Creep") == "op_ed"
    assert tag_context_type("hoover", "Commanding Heights Lecture") == "lecture"
    assert tag_context_type("hoover", "A Conversation with Kevin Warsh") == "interview"

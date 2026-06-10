from src.warsh.parsers import parse_fed_speech, parse_hoover_html


FED_HTML = """
<html><body>
<h1 class="title">Market Liquidity and Risk</h1>
<p class="article__time">March 5, 2007</p>
<div id="article">
  <p>Inflation is a choice, not an accident.</p>
  <p>The balance sheet has become bloated.</p>
</div>
</body></html>
"""

HOOVER_HTML = """
<html><body>
<h1>Commanding Heights</h1>
<time datetime="2025-04-15">April 15, 2025</time>
<article><p>Regime change in monetary policy is overdue.</p></article>
</body></html>
"""


def test_parse_fed_speech_extracts_title_date_text():
    doc = parse_fed_speech(FED_HTML, url="https://federalreserve.gov/x.htm")
    assert doc.source == "fed"
    assert doc.title == "Market Liquidity and Risk"
    assert doc.date == "2007-03-05"
    assert "Inflation is a choice" in doc.text
    assert "bloated" in doc.text
    assert doc.context_type == "speech"
    assert doc.id  # non-empty slug


def test_parse_hoover_html_extracts_iso_date_from_time_tag():
    doc = parse_hoover_html(HOOVER_HTML, url="https://hoover.org/y")
    assert doc.source == "hoover"
    assert doc.date == "2025-04-15"
    assert doc.context_type == "lecture"
    assert "Regime change" in doc.text

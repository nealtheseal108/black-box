import re
from bs4 import BeautifulSoup


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    # Fix spaces inserted before punctuation by the separator (e.g. "choice ." -> "choice.")
    text = re.sub(r" +([.,;:!?])", r"\1", text)
    return normalize_whitespace(text)


def normalize_whitespace(text: str) -> str:
    # Collapse intra-line whitespace to single spaces, preserve paragraph breaks.
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(text.split())


def tag_context_type(source: str, title: str) -> str:
    t = title.lower()
    if source == "hearing" or "hearing" in t:
        return "hearing"
    if source == "wsj_archive" or "op-ed" in t or "mission creep" in t:
        return "op_ed"
    if source in ("hoover", "hoover_pdf"):
        if "conversation" in t or "interview" in t:
            return "interview"
        if "lecture" in t or "commanding heights" in t:
            return "lecture"
        return "essay"
    return "speech"

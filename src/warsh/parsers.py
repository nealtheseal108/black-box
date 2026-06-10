import re
from datetime import datetime
from bs4 import BeautifulSoup
from src.warsh.models import Document
from src.warsh.normalize import clean_html, normalize_whitespace, tag_context_type

_MONTHS = {m: i for i, m in enumerate(
    ["january","february","march","april","may","june","july",
     "august","september","october","november","december"], start=1)}


def _slug(title: str, date: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60]
    return f"{date or 'undated'}-{base}".strip("-")


def _parse_date(text: str) -> str:
    text = text.strip()
    # ISO already?
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if m:
        return m.group(0)
    # "March 5, 2007"
    m = re.search(r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})", text)
    if m and m.group(1).lower() in _MONTHS:
        mo, d, y = _MONTHS[m.group(1).lower()], int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"
    return ""


def _doc_from_parts(source, title, date, url, body_html_or_text, is_html=True):
    text = clean_html(body_html_or_text) if is_html else normalize_whitespace(body_html_or_text)
    ctype = tag_context_type(source, title)
    return Document(
        id=_slug(title, date), source=source, title=title,
        date=date, url=url, context_type=ctype, text=text,
    )


def parse_fed_speech(html: str, url: str) -> Document:
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.find(class_="title") or soup.find("h1")).get_text(strip=True)
    date_node = soup.find(class_="article__time") or soup.find("time")
    date = _parse_date(date_node.get_text() if date_node else "")
    body = soup.find(id="article") or soup.find("article") or soup.body
    return _doc_from_parts("fed", title, date, url, str(body), is_html=True)


def parse_fraser(html: str, url: str) -> Document:
    doc = parse_fed_speech(html, url)
    doc.source = "fraser"
    doc.context_type = tag_context_type("fraser", doc.title)
    return doc


def parse_hoover_html(html: str, url: str) -> Document:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("h1").get_text(strip=True)
    time_node = soup.find("time")
    raw_date = time_node.get("datetime") if time_node and time_node.get("datetime") \
        else (time_node.get_text() if time_node else "")
    date = _parse_date(raw_date)
    body = soup.find("article") or soup.body
    return _doc_from_parts("hoover", title, date, url, str(body), is_html=True)


def parse_pdf(pdf_bytes: bytes, url: str, title: str, date: str, source: str = "hoover_pdf") -> Document:
    from io import BytesIO
    from pypdf import PdfReader
    reader = PdfReader(BytesIO(pdf_bytes))
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return _doc_from_parts(source, title, date, url, text, is_html=False)

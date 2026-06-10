from pathlib import Path
from src.warsh.models import Document
from src.warsh.normalize import normalize_whitespace, tag_context_type


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    if "\n---\n" not in raw:
        return {}, raw
    head, body = raw.split("\n---\n", 1)
    meta = {}
    for line in head.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    # Only treat as frontmatter if it looks like key:value metadata
    if not meta:
        return {}, raw
    return meta, body


def ingest_manual_dir(path) -> list[Document]:
    path = Path(path)
    docs: list[Document] = []
    for f in sorted(list(path.glob("*.txt")) + list(path.glob("*.md"))):
        raw = f.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        title = meta.get("title", f.stem)
        date = meta.get("date", "")
        source = "manual"
        ctype = meta.get("context_type") or tag_context_type(source, title)
        docs.append(Document(
            id=f"manual-{f.stem}", source=source, title=title, date=date,
            url=meta.get("url", "manual"), context_type=ctype,
            text=normalize_whitespace(body),
        ))
    return docs

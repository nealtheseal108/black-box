import json
from dataclasses import dataclass


@dataclass
class Document:
    id: str
    source: str       # fraser | fed | hoover | hoover_pdf | wsj_archive | hearing | manual
    title: str
    date: str         # ISO YYYY-MM-DD, "" if unknown
    url: str
    context_type: str # speech | essay | op_ed | interview | hearing | lecture
    text: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def to_jsonl_line(doc: Document) -> str:
    return json.dumps(
        {
            "id": doc.id, "source": doc.source, "title": doc.title,
            "date": doc.date, "url": doc.url, "context_type": doc.context_type,
            "text": doc.text,
        },
        ensure_ascii=False,
    )


def from_jsonl_line(line: str) -> Document:
    d = json.loads(line)
    return Document(
        id=d["id"], source=d["source"], title=d["title"], date=d["date"],
        url=d["url"], context_type=d["context_type"], text=d["text"],
    )


def build_manifest(docs: list[Document]) -> dict:
    by_source: dict[str, int] = {}
    for d in docs:
        by_source[d.source] = by_source.get(d.source, 0) + 1
    return {
        "document_count": len(docs),
        "total_words": sum(d.word_count for d in docs),
        "by_source": by_source,
    }

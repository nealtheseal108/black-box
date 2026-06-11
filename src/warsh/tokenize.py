import re

_WORD = re.compile(r"[a-z]+(?:-[a-z]+)*")
_SENT = re.compile(r"[^.!?]*[.!?]")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def sentences(text: str) -> list[str]:
    return [m.group().strip() for m in _SENT.finditer(text) if m.group().strip()]

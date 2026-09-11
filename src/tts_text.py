"""발화 텍스트를 재생 단위로 나누는 순수 로직."""
from __future__ import annotations

import re

MAX_CHUNK_CHARS = 40

_SENTENCE_END = re.compile(r"(?<=[.!?。？！])\s+|\n+")
_CLAUSE_END = re.compile(r"(?<=[,，、])\s*")


def split_sentences(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """발화 텍스트를 문장(필요하면 절) 단위로 끊어 돌려준다."""
    if not text or not text.strip():
        return []

    chunks: list[str] = []
    for sentence in _SENTENCE_END.split(text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            chunks.append(sentence)
            continue

        buffer = ""
        for clause in _CLAUSE_END.split(sentence):
            clause = clause.strip()
            if not clause:
                continue
            if buffer and len(buffer) + len(clause) > max_chars:
                chunks.append(buffer)
                buffer = clause
            else:
                buffer = f"{buffer} {clause}".strip()
        if buffer:
            chunks.append(buffer)

    return chunks

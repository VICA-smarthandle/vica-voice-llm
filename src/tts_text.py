"""발화 텍스트를 재생 단위로 나누는 순수 로직.

tts.py 와 분리한 이유는 두 가지다.
- supertonic(음성 모델) 없이 import 할 수 있어야 unit test 로 검증된다.
- 이 규칙은 '무엇을 말할지'가 아니라 '어떻게 끊을지'라서 합성기와 관심사가 다르다.
"""
from __future__ import annotations

import re

# 한 번에 재생할 최대 길이. 재생 중에는 긴급어 상시 감시가 쉬므로(ros_tts_node 가
# /vica/tts_state 를 true 로 둔다), 한 덩어리가 길수록 사용자의 진짜 "멈춰"를 놓치는
# 구간이 길어진다. 문장 사이마다 감시가 다시 열린다.
MAX_CHUNK_CHARS = 40

_SENTENCE_END = re.compile(r"(?<=[.!?。？！])\s+|\n+")
_CLAUSE_END = re.compile(r"(?<=[,，、])\s*")


def split_sentences(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """발화 텍스트를 문장(필요하면 절) 단위로 끊어 돌려준다.

    끊는 목적은 읽기 편해서가 아니라 안전이다. 위 MAX_CHUNK_CHARS 주석 참고.
    """
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

        # 너무 긴 문장은 쉼표 단위로 한 번 더 나눈다. 그래도 길면 그대로 둔다 —
        # 억지로 자르면 발음이 어색해져 안내 품질이 떨어진다.
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

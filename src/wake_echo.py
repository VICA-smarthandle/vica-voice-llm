"""호출 응답 "네?" 메아리 떼기 (3-나-1, 설계 2026-11-18 4.1).

호출이 확정되면 듣기 창이 바로 열리고 같은 순간 미션의 "네?"가 재생된다(11-18 13:15:19.50 · 19.52). 로봇
목소리가 녹음에 들어가 "네 잠깐만."으로 받아써지고, 통째 일치로 보는 글자 규칙을 빠져 LLM 으로 간다.
22일 자료에서 "네?" 뒤 378건 중 36건(9.5 %)이 "네"로 시작한다. "네?"는 네/아니요 질문이 아니라서
사용자가 실제로 "네, 화장실 가자"라고 했어도 떼어 뜻이 달라지지 않는다.
"""
from __future__ import annotations

import re
from typing import Optional

from .question_record import OpenQuestion
from .replies import WAKE_GREETING

# 맨 앞 "네" 한 낱말 + 문장부호나 공백. 뒤에 글자가 남을 때만 — "네." 한마디는 그대로 둔다.
_LEADING_NE = re.compile(r"^\s*네(?:[.,?!]+\s*|\s+)(?=\S)")


def strip_wake_echo(text: str, question: Optional[OpenQuestion]) -> str:
    """발화 때 열린 질문이 호출 응답 "네?"면 맨 앞 "네"를 뗀 글자를, 아니면 받은 글자를 그대로 돌려준다."""
    if question is None or question.kind != "open" or question.text.strip() != WAKE_GREETING:
        return text
    return _LEADING_NE.sub("", text, count=1)

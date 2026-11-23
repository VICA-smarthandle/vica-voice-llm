"""호출 응답 "네?" 메아리 떼기 (3-나-1, 설계 2026-11-18 4.1)."""
from __future__ import annotations

import re
from typing import Optional

from .question_record import OpenQuestion
from .replies import WAKE_GREETING

_LEADING_NE = re.compile(r"^\s*네(?:[.,?!…~]+\s*|\s+)(?=[^\s.,?!…~])")

HELD_QUESTION_STATES = frozenset({
    "confirming", "asking_next", "asking_wait_time", "waiting_release", "returning", "awaiting_user"})


def strip_wake_echo(text: str, question: Optional[OpenQuestion], dialog_state: str = "") -> str:
    """발화 때 열린 질문이 호출 응답 "네?"면 맨 앞 "네"를 뗀 글자를, 아니면 받은 글자를 그대로 돌려준다."""
    if question is None or question.kind != "open" or question.text.strip() != WAKE_GREETING:
        return text
    if dialog_state in HELD_QUESTION_STATES:
        return text
    return _LEADING_NE.sub("", text, count=1)

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

# 맨 앞 "네" 한 낱말 + 문장부호나 공백. 뒤에 말(문장부호가 아닌 글자)이 남을 때만 — "네." "네..." 는 그대로 둔다.
_LEADING_NE = re.compile(r"^\s*네(?:[.,?!…~]+\s*|\s+)(?=[^\s.,?!…~])")

# 미션이 물어 둔 질문을 쥔 채 "네?"를 내는 단계(RobotState.dialog_state): 확인 질문·도착 질문·대기 안내 중의
# "잠깐", 도착 질문·접근 질문 중의 "비카야", 복귀 중의 "잠깐". 이때 맨 앞 "네"는 그 질문의 승낙일 수 있어
# 떼지 않는다 — 떼면 "네 가 주세요"가 승낙을 잃고 LLM 으로 간다(11-18 최종 검토 I2).
HELD_QUESTION_STATES = frozenset({
    "confirming", "asking_next", "asking_wait_time", "waiting_release", "returning", "awaiting_user"})


def strip_wake_echo(text: str, question: Optional[OpenQuestion], dialog_state: str = "") -> str:
    """발화 때 열린 질문이 호출 응답 "네?"면 맨 앞 "네"를 뗀 글자를, 아니면 받은 글자를 그대로 돌려준다."""
    if question is None or question.kind != "open" or question.text.strip() != WAKE_GREETING:
        return text
    if dialog_state in HELD_QUESTION_STATES:
        return text
    return _LEADING_NE.sub("", text, count=1)

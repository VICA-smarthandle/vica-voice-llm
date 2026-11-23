"""시설 예약 문장 9개(설계 2026-11-18 6절) — 2026-11-18 사용자 확정본. 글자를 바꾸면 미리 합성이 어긋난다."""
from __future__ import annotations

from datetime import date

from .reservation_time import speak_date, speak_range

ASK_NAME = "성함을 말씀해 주세요."
DONE = "네, 예약이 완료되었습니다. 이제 어디로 갈까요?"
TAKEN = "그 시간에는 이미 예약이 있습니다. 다른 시간을 말씀해 주세요."
ASK_WHEN = "언제로 예약할까요?"
NOT_RESERVED = "예약하지 않았습니다."
ASK_WHERE = "어디로 예약할까요?"
BAD_TIME = "그 시간에는 예약할 수 없습니다. 다른 시간을 말씀해 주세요."
UNAVAILABLE = "지금은 예약을 받을 수 없습니다."
CANCELLED = "예약이 취소되었습니다."
FIXED = (ASK_NAME, DONE, TAKEN, ASK_WHEN, NOT_RESERVED, ASK_WHERE, BAD_TIME, UNAVAILABLE, CANCELLED)


def confirm_sentence(name: str, day: date, start: str, end: str, facility: str, today: date) -> str:
    """② 확인 질문. 저장할 값으로 만든다 — 되읽은 내용과 저장 내용이 같아야 한다."""
    return f"{name} 님 이름으로 {speak_date(day, today)} {speak_range(start, end)} {facility} 예약할까요?"

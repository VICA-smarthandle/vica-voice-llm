"""예약 날짜 계산 · 낱말이 있는지 보기 · 말로 읽기(설계 2026-11-18 5.2·6절) — 순수 모듈."""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Optional

_WEEKDAYS = "월화수목금토일"
_RELATIVE = (("내일모레", 2), ("오늘", 0), ("내일", 1), ("모레", 2), ("글피", 3))
_HOUR_WORDS = {1: "한", 2: "두", 3: "세", 4: "네", 5: "다섯", 6: "여섯", 7: "일곱", 8: "여덟", 9: "아홉",
               10: "열", 11: "열한", 12: "열두"}
_SINO = ("", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구")

_DATE_WORD = re.compile(r"오늘|내일|모레|글피|다음\s*주|이번\s*주|[월화수목금토일]요일|\d+\s*월|\d+\s*일")
_TIME_WORD = re.compile(r"오전|오후|아침|저녁|새벽|(?:\d+|열한|열두|다섯|여섯|일곱|여덟|아홉|한|두|세|네|열)\s*시(?!간)")
_SPAN_WORD = re.compile(r"까지|시간|동안")


def resolve_date(expr: str, today: date) -> Optional[date]:
    """LLM 이 준 날짜 표현 → 날짜. 못 바꾸면 None. 지난 날짜도 올해 날짜로 돌려준다(거절은 부르는 쪽)."""
    s = re.sub(r"\s+", "", str(expr or ""))
    if not s:
        return None
    iso = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if iso:
        return _date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
    for word, days in _RELATIVE:
        if word in s:
            return today + timedelta(days=days)
    month_day = re.search(r"(\d{1,2})월(\d{1,2})일", s)
    if month_day:
        return _date(today.year, int(month_day.group(1)), int(month_day.group(2)))
    weekday = re.search(r"(다음주|이번주)?([월화수목금토일])요일", s)
    if weekday:
        target = _WEEKDAYS.index(weekday.group(2))
        monday = today - timedelta(days=today.weekday())
        if weekday.group(1) == "다음주":
            return monday + timedelta(days=7 + target)
        if weekday.group(1) == "이번주":
            return monday + timedelta(days=target)
        return today + timedelta(days=(target - today.weekday()) % 7)
    day_only = re.fullmatch(r"(\d{1,2})일", s)
    if day_only:
        day = int(day_only.group(1))
        this_month = _date(today.year, today.month, day)
        if this_month is not None and this_month >= today:
            return this_month
        year, month = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
        return _date(year, month, day)
    return None


def _date(year: int, month: int, day: int) -> Optional[date]:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def valid_hhmm(value: str) -> Optional[str]:
    """"9:30" → "09:30". 시각 꼴이 아니면 None."""
    m = re.fullmatch(r"\s*(\d{1,2})\s*:\s*(\d{2})\s*", str(value or ""))
    if not m or not (0 <= int(m.group(1)) <= 23 and 0 <= int(m.group(2)) <= 59):
        return None
    return f"{int(m.group(1)):02d}:{m.group(2)}"


def has_date_word(text: str) -> bool:
    return bool(_DATE_WORD.search(text or ""))


def has_time_word(text: str) -> bool:
    """시각을 가리키는 낱말이 있나. "세 시간"의 "시"는 쓰는 시간이라 세지 않는다."""
    return bool(_TIME_WORD.search(text or ""))


def has_span_word(text: str) -> bool:
    """끝 시각이나 쓰는 시간을 말했나("까지"·"시간"·"동안")."""
    return bool(_SPAN_WORD.search(text or ""))


def speak_date(day: date, today: date) -> str:
    return {0: "오늘", 1: "내일", 2: "모레"}.get((day - today).days, f"{day.month}월 {day.day}일")


def _period(hhmm: str) -> str:
    return "오전" if int(hhmm[:2]) < 12 else "오후"


def _clock(hhmm: str) -> str:
    hour, minute = int(hhmm[:2]), int(hhmm[3:5])
    text = f"{_HOUR_WORDS[hour % 12 or 12]} 시"
    if minute == 30:
        return text + " 반"
    if minute:
        tens, ones = divmod(minute, 10)
        tens_word = ("" if tens == 1 else _SINO[tens]) + "십" if tens else ""
        return f"{text} {tens_word}{_SINO[ones]} 분"
    return text


def speak_time(hhmm: str) -> str:
    """"14:00" → "오후 두 시", "13:30" → "오후 한 시 반"."""
    return f"{_period(hhmm)} {_clock(hhmm)}"


def speak_range(start: str, end: str) -> str:
    """"오후 두 시부터 네 시까지" — 끝이 시작과 같은 오전·오후면 끝의 표시는 뺀다."""
    end_text = _clock(end) if _period(start) == _period(end) else speak_time(end)
    return f"{speak_time(start)}부터 {end_text}까지"

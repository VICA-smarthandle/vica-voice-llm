"""예약 날짜 계산·낱말 보기·말로 읽기(설계 2026-11-18 5.2·6절)."""
from datetime import date

import pytest

from src.reservation_time import (has_date_word, has_span_word, has_time_word, resolve_date, speak_date,
                                  speak_range, speak_time, valid_hhmm)

TODAY = date(2026, 11, 18)


@pytest.mark.parametrize("expr, expected", [
    ("오늘", date(2026, 11, 18)), ("내일", date(2026, 11, 19)), ("모레", date(2026, 11, 20)),
    ("글피", date(2026, 11, 21)), ("다음 주 월요일", date(2026, 11, 23)), ("다음주 월요일", date(2026, 11, 23)),
    ("이번 주 금요일", date(2026, 11, 20)), ("금요일", date(2026, 11, 20)), ("11월 20일", date(2026, 11, 20)),
    ("20일", date(2026, 11, 20)), ("11월 1일", date(2026, 11, 1)), ("2026-11-20", date(2026, 11, 20)),
    ("언젠가", None), ("", None),
])
def test_resolve_date(expr, expected):
    assert resolve_date(expr, TODAY) == expected


def test_valid_hhmm():
    assert valid_hhmm("14:00") == "14:00" and valid_hhmm("9:30") == "09:30"
    assert valid_hhmm("25:00") is None and valid_hhmm("오후 두 시") is None and valid_hhmm("") is None


def test_word_presence():
    assert has_date_word("내일 세 시") and has_date_word("11월 20일에") and has_date_word("금요일")
    assert not has_date_word("세미나실 예약해 줘.")
    assert has_time_word("내일 세 시") and has_time_word("14시에") and has_time_word("오후에 쓸게요")
    assert not has_time_word("세미나실 예약해 줘.") and not has_time_word("세미나실이 어디 있어?")
    assert not has_time_word("세 시간만 쓸게요")
    assert has_span_word("두 시부터 세 시까지") and has_span_word("세 시간만 쓸게요") and has_span_word("두 시간 동안")
    assert not has_span_word("내일 오후 두 시에 세미나실 예약해 줘.")


def test_speaking():
    assert [speak_date(date(2026, 11, d), TODAY) for d in (18, 19, 20, 23)] == ["오늘", "내일", "모레", "11월 23일"]
    assert speak_time("14:00") == "오후 두 시" and speak_time("13:30") == "오후 한 시 반"
    assert speak_time("10:10") == "오전 열 시 십 분" and speak_time("12:00") == "오후 열두 시"
    assert speak_range("14:00", "16:00") == "오후 두 시부터 네 시까지"
    assert speak_range("11:00", "13:00") == "오전 열한 시부터 오후 한 시까지"

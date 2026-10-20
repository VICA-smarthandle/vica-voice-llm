"""한글 숫자 읽기('사백칠호')를 숫자('407호')로 — 목적지 찾기·에코 비교용(2026-10-15 Qwen 교체)."""
from src.ko_numbers import korean_numbers_to_digits as k


def test_room_numbers_read_as_hundreds():
    assert k("사백칠호로가자") == "407호로가자"
    assert k("사백구호") == "409호"
    assert k("사백칠로안내") == "407로안내"
    assert k("천이백") == "1200"


def test_tens():
    assert k("이십분대기") == "20분대기"
    assert k("십오분") == "15분"
    assert k("구십") == "90"


def test_digit_by_digit_only_before_a_unit():
    assert k("사공칠호") == "407호"
    assert k("삼층") == "3층"
    assert k("일번") == "1번"


def test_ordinary_words_stay():
    for word in ("사람", "사무실", "구내식당", "이비인후과", "오른쪽", "천천히", "백화점", "이거", "삼성", "사이", "구로"):
        assert k(word) == word, word


def test_digits_and_mixed_text_untouched():
    assert k("407호로가자") == "407호로가자"
    assert k("") == ""
    assert k(None) == ""


def test_invalid_runs_stay():
    assert k("오사백칠") == "오사백칠"

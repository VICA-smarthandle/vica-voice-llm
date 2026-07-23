"""발화 분할(split_sentences) 검증. 음성 모델 불필요.

분할은 안전 장치다 — 재생 중에는 긴급어 감시가 쉬므로, 한 덩어리가 길수록
사용자의 진짜 긴급 발화를 놓치는 구간이 길어진다.
"""
from __future__ import annotations

from src.tts_text import MAX_CHUNK_CHARS, split_sentences


def test_splits_on_sentence_end():
    assert split_sentences("지금 이동 중입니다. 먼저 현재 안내를 취소해 주세요.") == [
        "지금 이동 중입니다.",
        "먼저 현재 안내를 취소해 주세요.",
    ]


def test_short_text_stays_one_chunk():
    assert split_sentences("안전을 위해 멈추겠습니다.") == ["안전을 위해 멈추겠습니다."]


def test_empty_input():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_no_chunk_exceeds_limit_when_splittable():
    text = (
        "왼쪽에 계단이 있습니다, 오른쪽으로 돌아가겠습니다, "
        "잠시 뒤에 다시 안내를 이어가겠습니다, 그대로 따라와 주세요."
    )
    chunks = split_sentences(text)
    assert len(chunks) > 1
    assert all(len(chunk) <= MAX_CHUNK_CHARS for chunk in chunks)


def test_keeps_all_content():
    """분할 과정에서 글자가 사라지지 않아야 한다 (안내 누락 방지)."""
    text = "첫 번째 문장입니다. 두 번째 문장입니다. 세 번째 문장입니다."
    joined = "".join(split_sentences(text)).replace(" ", "")
    assert joined == text.replace(" ", "")


def test_splits_on_newline():
    assert split_sentences("첫째 줄\n둘째 줄") == ["첫째 줄", "둘째 줄"]


def test_unsplittable_long_sentence_is_kept_whole():
    """쉼표가 없어 나눌 수 없으면 억지로 자르지 않는다 (발음 품질 우선)."""
    text = "가" * (MAX_CHUNK_CHARS + 10)
    assert split_sentences(text) == [text]

"""스마트핸들 모드 질문·응답 판정 검증 (ROS·LLM 없이)."""
from __future__ import annotations

import pytest

from src.handle_mode import (
    DEFAULT_ANSWER_WINDOW_SEC,
    NO,
    YES,
    ModeQuestion,
    classify_short_reply,
    normalize_short_reply,
)


# ---- 짧은 응답 판정 -----------------------------------------------------------


@pytest.mark.parametrize("text", ["네", "네.", " 예! ", "응", "그래요", "맞아요"])
def test_affirmative_forms(text):
    assert classify_short_reply(text) == YES


@pytest.mark.parametrize("text", ["아니", "아니요.", "아뇨", "싫어요", "취소"])
def test_negative_forms(text):
    assert classify_short_reply(text) == NO


def test_destination_utterance_is_not_an_answer():
    assert classify_short_reply("화장실 가줘") is None


def test_answer_glued_to_destination_is_not_short_reply():
    """"네, 화장실 가줘"는 답으로 보지 않는다.

    모드를 정하지 못하더라도 일반 처리로 넘겨 목적지를 뽑는 편이 낫다.
    되물으면 안내가 한 턴 늦어진다.
    """
    assert classify_short_reply("네, 화장실 가줘") is None


def test_normalize_strips_punctuation_and_space():
    assert normalize_short_reply(" 네, 맞아요! ") == "네맞아요"


# ---- 질문 상태 ---------------------------------------------------------------


def test_no_answer_is_taken_before_asking():
    q = ModeQuestion()
    assert q.waiting is False
    assert q.take_answer("네", 0.0) is None


def test_answer_right_after_asking():
    q = ModeQuestion()
    q.on_asked(100.0)
    assert q.waiting is True
    assert q.take_answer("네", 101.0) == YES


def test_answer_clears_the_wait():
    q = ModeQuestion()
    q.on_asked(100.0)
    q.take_answer("아니요", 101.0)

    assert q.waiting is False
    # 다음 발화는 더 이상 모드 답이 아니다.
    assert q.take_answer("네", 102.0) is None


def test_destination_first_keeps_waiting():
    """질문을 못 듣고 목적지부터 말하면 대기를 유지한다.

    되묻지 않고 안내를 진행시키되, 사용자가 뒤늦게 "네"라고 하면 아직 받는다.
    """
    q = ModeQuestion()
    q.on_asked(100.0)

    assert q.take_answer("화장실 가줘", 101.0) is None
    assert q.waiting is True
    assert q.take_answer("네", 102.0) == YES


def test_answer_after_window_is_ignored():
    q = ModeQuestion()
    q.on_asked(100.0)
    late = 100.0 + DEFAULT_ANSWER_WINDOW_SEC

    assert q.take_answer("네", late) is None
    assert q.waiting is False


def test_reset_clears_the_wait():
    q = ModeQuestion()
    q.on_asked(100.0)
    q.reset()

    assert q.waiting is False
    assert q.take_answer("네", 101.0) is None


def test_asking_again_restarts_the_window():
    """다음 사용자에게 다시 물으면 창이 새로 열린다."""
    q = ModeQuestion()
    q.on_asked(100.0)
    q.on_asked(500.0)

    assert q.take_answer("네", 501.0) == YES

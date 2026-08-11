"""일시정지·재개 발화 판정 검증 (ROS·LLM 없이)."""
from __future__ import annotations

import pytest

from src.emergency_filter import detect_emergency
from src.mission_command import (
    PAUSE,
    PAUSE_PHRASES,
    RESUME,
    RESUME_PHRASES,
    classify_mission_command,
)


# ---- 일시정지 -----------------------------------------------------------------


@pytest.mark.parametrize(
    "text", ["잠깐", "잠깐만", "잠깐만요.", " 잠시만 ", "세워줘", "기다려줘"]
)
def test_pause_forms(text):
    assert classify_mission_command(text) == PAUSE


# ---- 재개 ---------------------------------------------------------------------


@pytest.mark.parametrize("text", ["다시 출발", "출발해줘", "계속 가자", "이제 가자"])
def test_resume_forms(text):
    assert classify_mission_command(text) == RESUME


# ---- 그 밖 --------------------------------------------------------------------


@pytest.mark.parametrize("text", ["화장실 가줘", "지금 몇 층이야", "취소해줘", ""])
def test_other_utterances_are_not_commands(text):
    assert classify_mission_command(text) is None


def test_cancel_is_left_to_the_llm():
    """취소는 여기서 판정하지 않는다.

    Mission Manager 가 "취소할까요?"로 되묻기 때문에 오인식이 회복된다.
    되묻지 않는 일시정지·재개와 위험이 다르다.
    """
    assert classify_mission_command("취소해줘") is None
    assert classify_mission_command("그만 갈래") is None


# ---- "가자" — 문맥으로 가른다 --------------------------------------------------


def test_ambiguous_word_is_resume_only_when_paused():
    assert classify_mission_command("가자", is_paused=True) == RESUME
    assert classify_mission_command("가자", is_paused=False) is None


def test_ambiguous_word_stays_a_confirmation_while_confirm_is_pending():
    """확인 질문 직후의 "가자"는 그 대답이다.

    로봇이 일시정지 중이더라도 방금 물었으면 대답이 먼저다.
    """
    assert (
        classify_mission_command("가자", is_paused=True, confirm_pending=True) is None
    )


def test_explicit_resume_wins_even_while_confirm_is_pending():
    """모호하지 않은 말은 확인 대기 중이어도 재개다."""
    assert (
        classify_mission_command("다시 출발", confirm_pending=True) == RESUME
    )


# ---- 긴급어와 겹치지 않는다 ----------------------------------------------------


def test_no_phrase_collides_with_an_emergency_keyword():
    """일시정지·재개 목록에 긴급어가 섞이면 도달하지 못하는 죽은 항목이 된다.

    emergency_filter 가 LLM 이전 단계에서 먼저 잡아 E-stop 으로 보내기 때문이다.
    "멈춰"·"정지해줘"·"스톱"이 여기 없는 이유다.
    """
    offenders = [
        f'{p!r} <- {detect_emergency(p)!r}'
        for p in sorted(PAUSE_PHRASES | RESUME_PHRASES)
        if detect_emergency(p)
    ]
    assert offenders == [], "긴급어와 겹쳐 도달할 수 없는 항목:\n  " + "\n  ".join(
        offenders
    )

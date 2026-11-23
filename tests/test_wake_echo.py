"""호출 응답 "네?" 메아리 떼기(3-나-1, 설계 2026-11-18 4.1)."""
import pytest

from src.question_record import OpenQuestion
from src.replies import WAKE_GREETING
from src.wake_echo import strip_wake_echo


def q(kind="open", text=WAKE_GREETING):
    return OpenQuestion(id="q1", kind=kind, text=text, option_ids=(), opened_at=0.0)


@pytest.mark.parametrize("said, want", [
    ("네 잠깐만.", "잠깐만."),
    ("네. 다시 가자.", "다시 가자."),
    ("네, 화장실로 가자", "화장실로 가자"),
    ("네? 화장실로 가자", "화장실로 가자"),
    ("네.다시 가자", "다시 가자"),
    ("  네 나 배 아파", "나 배 아파"),
    ("네 네 네", "네 네"),
    ("네… 잠깐만", "잠깐만"),
    ("네~ 화장실 가자", "화장실 가자"),
])
def test_leading_ne_is_stripped_after_the_wake_greeting(said, want):
    assert strip_wake_echo(said, q()) == want


@pytest.mark.parametrize("said", ["네", "네.", "네?", "네!", "네...", "네..", "네 .", "네?!", "네 ?", "네. .", "네…"])
def test_bare_ne_is_kept(said):
    assert strip_wake_echo(said, q()) == said


@pytest.mark.parametrize("said", ["네가 누구야", "네네 알겠어", "네모난 방으로 가자", "그래 네 말이 맞아", "화장실로 가자"])
def test_other_words_starting_with_ne_are_kept(said):
    assert strip_wake_echo(said, q()) == said


@pytest.mark.parametrize("question", [
    None,
    q(kind="confirm", text="화장실로 안내해드릴까요?"),
    q(kind="yesno", text="다시 출발할까요?"),
    q(kind="open", text="어디로 안내해드릴까요?"),
    q(kind="yesno", text=WAKE_GREETING),
])
def test_only_after_the_wake_greeting(question):
    assert strip_wake_echo("네 화장실로 가자", question) == "네 화장실로 가자"


@pytest.mark.parametrize("dialog_state", ["confirming", "asking_next", "asking_wait_time", "waiting_release",
                                          "returning", "awaiting_user"])
def test_kept_while_the_mission_still_holds_another_question(dialog_state):
    assert strip_wake_echo("네 가 주세요", q(), dialog_state) == "네 가 주세요"


@pytest.mark.parametrize("dialog_state", ["", "idle", "navigating", "paused", "waiting"])
def test_stripped_when_no_other_question_is_held(dialog_state):
    assert strip_wake_echo("네 잠깐만.", q(), dialog_state) == "잠깐만."

"""확인 질문 중 같은 곳을 또 말하면 LLM 은 다시 묻지 않는다(2026-10-11 사용자 결정)."""
from __future__ import annotations

from src.mission_question import confirming_same_place, quiet_for_mission
from src.schema import VicaIntent, should_forward_intent
from src.tts_queue import request_for_intent

NAMES = ["사회 복지창구", "남자 화장실", "여자 화장실", "407호", "화장실"]


def _nav(name, reply, need_confirm=True):
    return VicaIntent(intent="navigate", destination_candidate=name, reply=reply,
                      need_confirm=need_confirm, safety_flag="normal")


def test_same_place_again_is_left_to_the_mission():
    intent = _nav("사회 복지창구", "사회 복지창구로 안내해드릴까요?")
    out = quiet_for_mission(intent, "confirming", "사회 복지창구로 안내해드릴까요?", 3.0, NAMES)
    assert out.reply == "" and request_for_intent(out) is None
    assert out.intent == "navigate" and out.need_confirm is True and should_forward_intent(out)


def test_mission_switch_question_counts_too():
    for last in ("네, 407호로 안내해드릴까요?", "407호는 로봇관 4층에 있는데, 안내해드릴까요?"):
        assert confirming_same_place(_nav("407호", "407호로 안내해드릴까요?"), "confirming", last, NAMES)


def test_another_place_keeps_the_question():
    intent = _nav("남자 화장실", "남자 화장실로 안내해드릴까요?")
    out = quiet_for_mission(intent, "confirming", "사회 복지창구로 안내해드릴까요?", 3.0, NAMES)
    assert out.reply == "남자 화장실로 안내해드릴까요?"


def test_shorter_name_inside_a_longer_one_is_another_place():
    intent = _nav("화장실", "화장실로 안내해드릴까요?")
    assert not confirming_same_place(intent, "confirming", "남자 화장실로 안내해드릴까요?", NAMES)


def test_only_while_confirming():
    intent = _nav("사회 복지창구", "사회 복지창구로 안내해드릴까요?")
    for state in ("idle", "navigating", "waiting", "asking_next"):
        out = quiet_for_mission(intent, state, "사회 복지창구로 안내해드릴까요?", 3.0, NAMES)
        assert out.reply == "사회 복지창구로 안내해드릴까요?", state


def test_confirmed_navigate_is_untouched():
    intent = _nav("사회 복지창구", "", need_confirm=False)
    assert quiet_for_mission(intent, "confirming", "사회 복지창구로 안내해드릴까요?", 3.0, NAMES) is intent


def test_answer_to_a_choice_question_is_not_a_repeat():
    intent = _nav("여자 화장실", "여자 화장실로 안내해드릴까요?")
    last = "남자 화장실과 여자 화장실 중 어디로 갈까요?"
    assert not confirming_same_place(intent, "confirming", last, NAMES)
    assert quiet_for_mission(intent, "confirming", last, 3.0, NAMES).reply == "여자 화장실로 안내해드릴까요?"

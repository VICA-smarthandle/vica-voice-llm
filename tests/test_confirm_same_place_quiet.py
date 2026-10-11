"""확인 질문 중 같은 곳을 또 말하면 LLM 은 다시 묻지 않는다(2026-10-11 사용자 결정).

10-10 18:34 "407호로 안내해드릴까요?" 중 "그럼 407호" → LLM 이 같은 확인 질문을 또 하고, 미션은 확인 중인
곳의 재제안을 승낙으로 보고 출발해 "안내를 시작합니다"와 겹쳤다. 다른 곳을 말하면 미션은 조용히 확인
대상을 바꾸므로 LLM 의 질문이 그대로 나가야 한다.
"""
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
    # 미션에는 그대로 간다 — 미션이 승낙으로 받고 출발한다
    assert out.intent == "navigate" and out.need_confirm is True and should_forward_intent(out)


def test_mission_switch_question_counts_too():
    # 미션이 물은 "네, OO로 안내해드릴까요?"(확인 중 다른 곳 확정)·위치 질문 꼴도 같다
    for last in ("네, 407호로 안내해드릴까요?", "407호는 로봇관 4층에 있는데, 안내해드릴까요?"):
        assert confirming_same_place(_nav("407호", "407호로 안내해드릴까요?"), "confirming", last, NAMES)


def test_another_place_keeps_the_question():
    intent = _nav("남자 화장실", "남자 화장실로 안내해드릴까요?")
    out = quiet_for_mission(intent, "confirming", "사회 복지창구로 안내해드릴까요?", 3.0, NAMES)
    assert out.reply == "남자 화장실로 안내해드릴까요?"


def test_shorter_name_inside_a_longer_one_is_another_place():
    # "남자 화장실로 안내해드릴까요?" 중 "화장실"(다른 목적지)을 제안하면 같은 곳이 아니다
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
    # "남자 화장실과 여자 화장실 중 어디로 갈까요?"의 답 "여자" — 확인 질문이 아니라 고르기 질문이다
    intent = _nav("여자 화장실", "여자 화장실로 안내해드릴까요?")
    last = "남자 화장실과 여자 화장실 중 어디로 갈까요?"
    assert not confirming_same_place(intent, "confirming", last, NAMES)
    assert quiet_for_mission(intent, "confirming", last, 3.0, NAMES).reply == "여자 화장실로 안내해드릴까요?"

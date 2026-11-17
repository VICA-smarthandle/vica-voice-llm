"""한 목소리(설계 2026-11-14 4.4, 3-가) — 질문 문장은 미션에 넘기고 음성은 말하지 않는다."""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.langchain_intent_parser import _shortcut_intent
from src.one_voice import hand_over_question, held_resume_carrier, voice_may_speak
from src.replies import (APPROACH_ONBOARDING, COMMAND_DECLINED, RESUME_CONFIRM, RETRY_PROMPT, WAKE_GREETING,
                         choose_between)
from src.question_record import OpenQuestion
from src.schema import DestinationData, VicaIntent

DESTS = [
    DestinationData(id="wc_m", name="남자 화장실"),
    DestinationData(id="wc_w", name="여자 화장실"),
    DestinationData(id="wc", name="화장실"),
    DestinationData(id="hs1", name="서울로봇고등학교"),
    DestinationData(id="hs2", name="서울로봇고등학교"),
    DestinationData(id="elev", name="엘리베이터"),
]


def test_confirm_question_goes_to_mission():
    """확인 질문은 미션이 목적지 confirm_prompt 로 묻는다(결정 A) — 음성 문장은 넘기지 않는다."""
    intent = VicaIntent(intent="navigate", matched_destination_id="elev", need_confirm=True,
                        reply="엘리베이터로 안내해드릴까요?")
    out = hand_over_question(intent, DESTS)
    assert (out.reply, out.ask_text, out.need_confirm) == ("", "", True)


def test_choice_options_use_outer_names():
    intent = VicaIntent(intent="clarify", need_confirm=False, reply=choose_between(["남자 화장실", "여자 화장실"]))
    out = hand_over_question(intent, DESTS)
    assert (out.ask_kind, out.ask_option_ids, out.reply) == ("choice", ["wc_m", "wc_w"], "")
    assert out.ask_text == "남자 화장실과 여자 화장실 중 어디로 갈까요?"


def test_choice_options_keep_same_name_twins():
    intent = VicaIntent(intent="clarify", reply="서울로봇고등학교와 서울로봇고등학교 중 어디로 갈까요?")
    assert hand_over_question(intent, DESTS).ask_option_ids == ["hs1", "hs2"]


def test_llm_choice_variant_is_choice():
    intent = VicaIntent(intent="clarify", reply="남자 화장실과 여자 화장실 중에 어디로 갈까요?")
    assert hand_over_question(intent, DESTS).ask_kind == "choice"


def test_llm_clarify_goes_to_mission_as_open():
    intent = VicaIntent(intent="clarify", need_confirm=False, reply="어떤 서류를 내러 가시나요?")
    out = hand_over_question(intent, DESTS)
    assert (out.ask_kind, out.ask_text, out.reply) == ("open", "어떤 서류를 내러 가시나요?", "")


@pytest.mark.parametrize("reply", [WAKE_GREETING, RETRY_PROMPT, APPROACH_ONBOARDING])
def test_fixed_questions_go_to_mission(reply):
    out = hand_over_question(VicaIntent(intent="unknown", need_confirm=False, reply=reply), DESTS)
    assert (out.ask_kind, out.ask_text, out.reply) == ("open", reply, "")


def test_held_resume_carries_yesno():
    out = hand_over_question(VicaIntent(intent="resume", need_confirm=True, reply=RESUME_CONFIRM), DESTS)
    assert (out.ask_kind, out.ask_text, out.reply, out.intent) == ("yesno", RESUME_CONFIRM, "", "resume")
    carrier = held_resume_carrier(out)
    assert (carrier.intent, carrier.ask_text, carrier.ask_kind, carrier.need_confirm, carrier.reply) == (
        "clarify", RESUME_CONFIRM, "yesno", False, "")


def test_question_answer_with_question_tail_goes_to_mission():
    intent = VicaIntent(intent="question", need_confirm=False, reply="화장실은 2층에 있어요. 안내해드릴까요?")
    assert hand_over_question(intent, DESTS).ask_kind == "open"


def test_statement_stays_with_voice():
    intent = VicaIntent(intent="question", need_confirm=False, reply="화장실은 2층에 있어요.")
    assert hand_over_question(intent, DESTS) is intent


def test_emergency_untouched():
    intent = VicaIntent(intent="unknown", need_confirm=False, safety_flag="emergency", reply="괜찮으세요?")
    assert hand_over_question(intent, DESTS) is intent


def _open(text: str):
    return OpenQuestion(id="q1", kind="confirm", text=text, option_ids=(), opened_at=0.0)


def test_voice_may_speak_only_answers_while_question_open():
    answer = VicaIntent(intent="question", reply="화장실은 2층에 있어요.")
    failed = VicaIntent(intent="unknown", reply=COMMAND_DECLINED)
    emergency = VicaIntent(intent="unknown", safety_flag="emergency", reply="")
    mission_q = _open("화장실로 안내해드릴까요?")
    assert [voice_may_speak(i, mission_q) for i in (answer, failed, emergency)] == [True, False, True]
    assert [voice_may_speak(i, None) for i in (answer, failed, emergency)] == [True, True, True]


def test_voice_speaks_its_own_resume_answer():
    """"다시 출발할까요?"는 판단을 음성이 쥔 질문(결정 B) — 그 대답 "아니"의 확인 멘트는 음성이 말한다(검토 I1)."""
    declined = _shortcut_intent("아니", [HumanMessage("다시 가자"), AIMessage(RESUME_CONFIRM)], DESTS)
    assert (declined.intent, declined.reply) == ("unknown", COMMAND_DECLINED)
    assert hand_over_question(declined, DESTS) is declined
    assert voice_may_speak(declined, _open(RESUME_CONFIRM)) is True

"""한 목소리(설계 2026-11-14 4.4, 3-가) — 질문 문장은 미션에 넘기고 음성은 말하지 않는다."""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.langchain_intent_parser import _shortcut_intent
from src.one_voice import demote_unasked_confirm, hand_over_question, held_resume_carrier, voice_may_speak
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


def test_confirmed_navigate_sentence_is_not_handed_over():
    """확정 navigate 에 붙은 문장은 예전에도 음성이 말하지 않았다 — 미션에 질문으로 넘기지 않는다.

    11-18 실기: 접근 불가 목적지(407호)의 확정 navigate 에 붙은 "407호로 안내해드릴까요?"가 열린 질문으로 4번 실려 갔다.
    """
    intent = VicaIntent(intent="navigate", matched_destination_id="elev", need_confirm=False,
                        reply="엘리베이터로 안내해드릴까요?")
    assert hand_over_question(intent, DESTS) is intent
    said = VicaIntent(intent="navigate", matched_destination_id="elev", need_confirm=False, reply="엘리베이터 안내를 시작합니다.")
    assert hand_over_question(said, DESTS) is said


# --- 확인 없이 나온 '확정'은 제안으로 낮춘다 (11-18 사용자: "LLM 이 확정으로 내는 것도 막아야") ---------------------
def _confirm_q(*ids):
    return OpenQuestion(id="q9", kind="confirm", text="엘리베이터로 안내해드릴까요?", option_ids=tuple(ids), opened_at=0.0)


def _go(dest_id="elev", **kw):
    base = dict(intent="navigate", matched_destination_id=dest_id, need_confirm=False, reply="엘리베이터 안내를 시작합니다.")
    base.update(kw)
    return VicaIntent(**base)


def test_unasked_confirmed_navigate_becomes_proposal():
    """11-18 13:05:39 실기 — 두 번 거절한 뒤 "화장실로 가자"에 LLM 이 확정을 내 확인 없이 출발했다."""
    out = demote_unasked_confirm(_go(), None, DESTS)
    assert (out.intent, out.matched_destination_id, out.need_confirm, out.reply) == ("navigate", "elev", True, "")


def test_confirmed_navigate_kept_when_its_confirm_question_is_open():
    intent = _go()
    assert demote_unasked_confirm(intent, _confirm_q("elev"), DESTS) is intent


def test_confirmed_navigate_demoted_when_another_destination_was_asked():
    assert demote_unasked_confirm(_go(), _confirm_q("wc"), DESTS).need_confirm is True


def test_confirmed_navigate_demoted_when_open_question_is_not_a_confirm():
    wake = OpenQuestion(id="q1", kind="open", text="네?", option_ids=(), opened_at=0.0)
    assert demote_unasked_confirm(_go(), wake, DESTS).need_confirm is True


def test_unapproachable_destination_stays_confirmed():
    """갈 수 없는 곳은 그대로 보낸다 — 미션이 이유를 말하고 거절한다(확인 질문을 하면 "네" 뒤에야 거절한다)."""
    dests = DESTS + [DestinationData(id="p407", name="407호", is_approachable=False)]
    intent = _go("p407", reply="407호로 안내해드릴까요?")
    assert demote_unasked_confirm(intent, None, dests) is intent


@pytest.mark.parametrize("intent", [
    VicaIntent(intent="navigate", matched_destination_id="elev", need_confirm=True, reply="엘리베이터로 안내해드릴까요?"),
    VicaIntent(intent="resume", need_confirm=False),
    VicaIntent(intent="affirm", need_confirm=False),
    VicaIntent(intent="navigate", matched_destination_id="모르는id", need_confirm=False),
    VicaIntent(intent="navigate", matched_destination_id="elev", need_confirm=False, safety_flag="emergency"),
])
def test_demote_leaves_everything_else_alone(intent):
    assert demote_unasked_confirm(intent, None, DESTS) is intent


def test_resume_question_is_always_the_fixed_sentence():
    """"다시 출발할까요?"는 정해진 문장만 넘긴다 — 11-18 온라인 실기에서 LLM 이 "…로봇은 그대로 서 있어요."를 덧붙였다(사용자 결정)."""
    padded = VicaIntent(intent="resume", need_confirm=True, reply="다시 출발할까요? 로봇은 그대로 서 있어요.")
    out = hand_over_question(padded, DESTS)
    assert (out.ask_text, out.ask_kind, out.reply) == (RESUME_CONFIRM, "yesno", "")


def test_p3_pick_from_choice_is_demoted_to_a_confirm_question():
    """원칙 3 — 보기에서 고른 뒤에도 한 번 더 묻는다(설계 2026-11-20 6.7). 10월 기록의 '고른 뒤 바로 출발' 33건은 고르기 질문이
    열린 채 확정 navigate 가 온 것이다 — 그 목적지의 확인 질문이 아니므로 제안으로 낮춰진다(11-21 기록 33건 모두 확인)."""
    choice = OpenQuestion(id="q2", kind="choice", text="남자 화장실과 여자 화장실 중 어디로 갈까요?",
                          option_ids=("wc_m", "wc_w"), opened_at=0.0)
    out = demote_unasked_confirm(_go("wc_m", reply="남자 화장실로 안내를 시작합니다."), choice, DESTS)
    assert (out.matched_destination_id, out.need_confirm, out.reply) == ("wc_m", True, "")

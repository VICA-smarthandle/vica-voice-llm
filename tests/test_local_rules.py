"""로컬 규칙 스위치(VICA_LOCAL_RULES) 시험 — 2026-10-05 인수인계 "로컬 LLM 정비".

스위치를 끈 상태는 기존 시험 전체가 그대로 통과하는 것으로 증명한다(OpenAI 쪽
무변경). 여기서는 켠 상태의 새 동작만 본다. 모두 LLM 없이 도는 경로다.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src import local_rules
from src.langchain_intent_parser import (
    _build_system_prompt,
    _finalize,
    _IntentDraft,
    _shortcut_intent,
    is_scripted_intent,
    parse_intent,
    parse_wait_minutes,
)
from src.replies import RETRY_PROMPT
from src.schema import DestinationData, RobotState

WC = DestinationData(id="wc", name="화장실", confirm_prompt="화장실으로 안내해드릴까요?")
REST = DestinationData(id="rest", name="휴게실", confirm_prompt="휴게실으로 안내해드릴까요?")
DESTS = [WC, REST]
BUSY = "지금은 다른 응대 중입니다. 잠시 후 다시 말씀해 주세요."


@pytest.fixture
def local_on(monkeypatch):
    monkeypatch.setenv("VICA_LOCAL_RULES", "1")
    monkeypatch.setenv("VICA_INTENT_INPUT", "text")
    yield


def state(dialog: str) -> RobotState:
    return RobotState(dialog_state=dialog)


# ---------------------------------------------------------------- 스위치
def test_switch_needs_both_flag_and_text_mode(monkeypatch):
    monkeypatch.delenv("VICA_LOCAL_RULES", raising=False)
    monkeypatch.setenv("VICA_INTENT_INPUT", "text")
    assert local_rules.enabled() is False
    monkeypatch.setenv("VICA_LOCAL_RULES", "1")
    assert local_rules.enabled() is True
    monkeypatch.setenv("VICA_INTENT_INPUT", "audio")   # Realtime 실행이면 켜도 꺼짐
    assert local_rules.enabled() is False


# ------------------------------------------- '음' 되묻기 의도 유지 (검토에서 짚은 위험)
def test_hmm_while_confirming_reasks_even_if_mission_line_interleaved(local_on):
    """확인 질문 뒤 미션의 다른 말이 기록에 끼어도, 미션이 confirming 이면 '음'은
    affirm 이 되지 않고 같은 확인 질문으로 되묻는다. affirm 이 나가면 미션은
    CONFIRMING + affirm 을 확정으로 받아 출발한다."""
    history = [HumanMessage("화장실 가고 싶어요"), AIMessage(WC.confirm_prompt), AIMessage(BUSY)]
    r = parse_intent("음.", DESTS, history=history, robot_state=state("confirming"))
    assert r.intent == "clarify"
    assert r.reply == WC.confirm_prompt
    assert not r.matched_destination_id


def test_hmm_while_confirming_without_known_prompt_uses_retry_and_is_not_swallowed(local_on):
    r = parse_intent("어어", DESTS, history=[AIMessage(BUSY)], robot_state=state("confirming"))
    assert r.intent == "clarify" and r.reply == RETRY_PROMPT
    assert is_scripted_intent(r, DESTS) is True          # 재청취 기각에 삼켜지지 않음


def test_yes_while_confirming_finds_recent_prompt_through_interleaved_line(local_on):
    history = [AIMessage(REST.confirm_prompt), AIMessage(BUSY)]
    r = parse_intent("그래.", DESTS, history=history, robot_state=state("confirming"))
    assert r.intent == "navigate" and r.matched_destination_id == "rest"
    assert r.need_confirm is False


def test_yes_while_confirming_without_prompt_sends_affirm(local_on):
    r = parse_intent("네", DESTS, history=[], robot_state=state("confirming"))
    assert r.intent == "affirm"


def test_no_while_confirming_is_deny(local_on):
    r = parse_intent("아니. 그건 아니고", DESTS, history=[AIMessage(WC.confirm_prompt)],
                     robot_state=state("confirming"))
    assert r.intent == "deny"


# ------------------------------------------- 출발 확정은 confirming 일 때만 (수리안 가)
def test_yes_when_idle_does_not_confirm_stale_prompt(local_on):
    """미션이 확인 중이 아닌데 기록에 옛 확인 질문이 남아 있어도 "그래"로 출발이
    확정되지 않는다(10-02 시연 ①). idle 은 대답 대기 단계가 아니라 지름길도 안 쓴다."""
    history = [AIMessage(WC.confirm_prompt)]
    assert _shortcut_intent("그래", history, DESTS, dialog_state="idle", local=True) is None


def test_yes_when_awaiting_user_is_affirm(local_on):
    r = _shortcut_intent("그래", [], DESTS, dialog_state="awaiting_user", local=True)
    assert r is not None and r.intent == "affirm"


def test_unknown_dialog_state_keeps_old_behavior(local_on):
    """미션 상태를 모르면(옛 메시지·수신 전) 예전처럼 마지막 로봇 말로 판단한다."""
    r = _shortcut_intent("그래", [AIMessage(WC.confirm_prompt)], DESTS, dialog_state="", local=True)
    assert r.intent == "navigate" and r.matched_destination_id == "wc"


def test_switch_off_ignores_dialog_state(monkeypatch):
    monkeypatch.delenv("VICA_LOCAL_RULES", raising=False)
    history = [AIMessage(WC.confirm_prompt)]
    r = parse_intent("그래", DESTS, history=history, robot_state=state("idle"))
    assert r.intent == "navigate"            # 예전 그대로(스위치 꺼짐 = OpenAI 쪽 무변경)


# ---------------------------------------------------------------- 대기 시간
def test_wait_range_takes_max_without_bonus_when_local():
    draft = _IntentDraft(intent="wait")
    text = "한 10분에서 15분 기다려줘"
    assert _finalize(draft, DESTS, user_text=text, local=True).wait_minutes == 15
    assert _finalize(draft, DESTS, user_text=text, local=False).wait_minutes == 23


def test_wait_without_number_is_empty_when_local_even_if_model_guessed():
    draft = _IntentDraft(intent="wait", wait_minutes=5)
    assert _finalize(draft, DESTS, user_text="좀 있다 올게", local=True).wait_minutes == -1
    assert _finalize(draft, DESTS, user_text="좀 있다 올게", local=False).wait_minutes == 5


def test_parse_wait_minutes_default_unchanged():
    assert parse_wait_minutes("5분에서 10분") == 15
    assert parse_wait_minutes("5분에서 10분", range_factor=1.0) == 10
    assert parse_wait_minutes("십 분") == 10


# ---------------------------------------------------------------- 지시문
def test_local_prompt_adds_five_rules_only_when_local():
    on = _build_system_prompt(DESTS, local=True)
    off = _build_system_prompt(DESTS, local=False)
    assert "로컬 추가 규칙" in on and "로컬 추가 규칙" not in off
    assert on.startswith(off)                # 기존 지시문 뒤에 덧붙일 뿐


# ---------------------------------------------------------------- 기록·못 알아들음
def test_history_cleared_only_when_approach_question_starts():
    assert local_rules.should_clear_history("idle", "awaiting_user") is True
    assert local_rules.should_clear_history("seeking", "awaiting_user") is True
    assert local_rules.should_clear_history("awaiting_user", "awaiting_user") is False
    assert local_rules.should_clear_history("confirming", "navigating") is False


def test_short_answer_allowed_states():
    for s in ("awaiting_user", "confirming", "asking_next", "asking_wait_time", ""):
        assert local_rules.short_answer_allowed(s) is True
    for s in ("idle", "navigating", "seeking", "waiting"):
        assert local_rules.short_answer_allowed(s) is False


def test_misheard_gate_once_then_silent_and_resets():
    g = local_rules.MisheardGate()
    assert g.decide(RETRY_PROMPT) == RETRY_PROMPT
    assert g.decide(RETRY_PROMPT) is None
    assert g.decide(RETRY_PROMPT) is None
    g.reset()
    assert g.decide(RETRY_PROMPT) == RETRY_PROMPT

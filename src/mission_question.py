"""미션이 질문 중일 때 LLM 대답을 줄이는 판단 (미션 요청 반응표 2026-10-08, 규칙 2·다시 묻기)."""
from __future__ import annotations

from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage

from .local_rules import ANSWER_WAIT_STATES
from .mission_phrases import HELP_CANCEL_CONFIRM, HELP_CONFIRM, WAIT_FINISH_ASK, WAIT_NEED_ASK
from .replies import CANCEL_CONFIRM, RESUME_CONFIRM, WAKE_GREETING, expects_answer
from .schema import VicaIntent

MISSION_QUESTIONS = frozenset({WAIT_FINISH_ASK, WAIT_NEED_ASK, CANCEL_CONFIRM,
                               HELP_CONFIRM, HELP_CANCEL_CONFIRM})
ASK_FRESH_SEC = 40.0
APPROACH_ANSWER_STATES = frozenset({"awaiting_user", "turning"})


def mission_is_asking(dialog_state: str, last_robot_text: str, last_robot_age_sec: float) -> bool:
    """미션이 사용자 대답을 기다리는 중인가."""
    if dialog_state in ANSWER_WAIT_STATES:
        return True
    return ((last_robot_text or "").strip() in MISSION_QUESTIONS
            and last_robot_age_sec <= ASK_FRESH_SEC)


def confirming_same_place(intent, dialog_state: str, last_robot_text: str,
                          place_names: Sequence[str] = ()) -> bool:
    """확인 질문("OO로 안내해드릴까요?")에 답을 기다리는 중 같은 곳을 또 제안했나."""
    if dialog_state != "confirming" or intent.intent != "navigate" or not intent.need_confirm:
        return False
    name = (intent.destination_candidate or "").strip()
    last = (last_robot_text or "").strip()
    if not name or not last.endswith("?") or name not in last:
        return False
    if not place_names:
        return True
    named = [n for n in place_names if n and n in last]
    outer = [n for n in named if not any(m != n and n in m for m in named)]
    return outer == [name]


def quiet_for_mission(intent, dialog_state: str, last_robot_text: str,
                      last_robot_age_sec: float, place_names: Sequence[str] = ()):
    """미션이 이어서 말할 자리면 LLM reply 를 비운 intent 를, 아니면 그대로 돌려준다."""
    if getattr(intent, "safety_flag", "") == "emergency" or not intent.reply:
        return intent
    if dialog_state == "awaiting_user" and intent.intent != "question":
        return intent.model_copy(update={"reply": "", "need_confirm": False}
                                 if intent.intent == "resume" else {"reply": ""})
    if dialog_state == "awaiting_user" and expects_answer(intent.reply):
        return intent.model_copy(update={"reply": ""})
    if intent.intent == "pause":
        return intent.model_copy(update={"reply": ""})
    if intent.intent == "help":
        return intent.model_copy(update={"reply": ""})
    if intent.intent == "cancel":
        return intent.model_copy(update={"reply": ""})
    if confirming_same_place(intent, dialog_state, last_robot_text, place_names):
        return intent.model_copy(update={"reply": ""})
    if intent.intent == "resume" and intent.need_confirm and dialog_state == "navigating":
        return intent.model_copy(update={"reply": "", "need_confirm": False})
    if (intent.intent == "unknown" and intent.reply != WAKE_GREETING
            and mission_is_asking(dialog_state, last_robot_text, last_robot_age_sec)):
        return intent.model_copy(update={"reply": ""})
    if (intent.intent == "navigate" and intent.need_confirm
            and dialog_state in APPROACH_ANSWER_STATES):
        return intent.model_copy(update={"reply": ""})
    return intent


def _robot_lines(history: list[BaseMessage]) -> list[str]:
    return [m.content for m in history if isinstance(m, AIMessage)]


def resume_reasked(history: list[BaseMessage]) -> bool:
    """이미 한 번 다시 물었나 — 로봇 말 마지막 두 줄이 모두 "다시 출발할까요?"."""
    lines = _robot_lines(history or [])
    return len(lines) >= 2 and lines[-1] == RESUME_CONFIRM and lines[-2] == RESUME_CONFIRM


def reask_held_resume(intent, history: list[BaseMessage]):
    """"다시 출발할까요?"에 못 알아들은 답(unknown)이면 같은 질문을 한 번 다시 묻는 intent 로"""
    if intent.intent != "unknown" or getattr(intent, "safety_flag", "") == "emergency":
        return intent
    lines = _robot_lines(history or [])
    if not lines or lines[-1] != RESUME_CONFIRM or resume_reasked(history):
        return intent
    return VicaIntent(intent="resume", confidence=1.0, reply=RESUME_CONFIRM, need_confirm=True)


def should_reask_resume_on_silence(history: list[BaseMessage], last_robot_text: str,
                                   last_robot_age_sec: float) -> bool:
    """듣기 창이 빈손으로 닫혔을 때 "다시 출발할까요?"를 한 번 다시 물을까. 질문 뒤 사용자"""
    if not history or not isinstance(history[-1], AIMessage):
        return False
    if (last_robot_text or "").strip() != RESUME_CONFIRM or last_robot_age_sec > ASK_FRESH_SEC:
        return False
    return history[-1].content == RESUME_CONFIRM and not resume_reasked(history)

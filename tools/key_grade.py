"""새 답지 채점 — 로봇이 한 일(행동 · 대상 · 분 · 할 말)을 답지와 견준다 (설계 2026-11-20 G2 · 8절, 계획 Task 1-3)."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Optional, Sequence

from src.one_voice import _CHOICE_TAIL
from src.question_record import OpenQuestion
from src.replies import WAKE_GREETING
from tools.answer_grade import question_destination_id

NOT_AVAILABLE_HEAD = "제가 아는 목적지 중에서"
WAKE_ANSWERS = ("네", "네말씀하세요", "네말씀해주세요")
END_QUESTION = "마칠까요"
UNASKED = ("go", "confirm", "pick", "wait", "finish", "bye")
PRINCIPLE = {"silent": 7, "wake": 13, "yes": 6, "no": 6, "pick": 6, "wait": 6, "confirm": 5, "which": 4,
             "notavail": 11, "reask": 8, "finish": 3, "answer": 25, "bye": 14, "repeat": 28}


@dataclass(frozen=True)
class Reaction:
    """로봇이 한 일 — 행동, 대상(목적지 id), 분, 한 말."""

    action: str
    target: str = ""
    minutes: str = ""
    say: str = ""


def is_wake_answer(said: str) -> bool:
    """로봇이 낸 문장이 호출에 답한 말인가("네?" · "네, 말씀하세요." 등)."""
    said = (said or "").strip()
    return said == WAKE_GREETING or "".join(ch for ch in said if ch.isalnum()) in WAKE_ANSWERS


def _navigate(intent, question: Optional[OpenQuestion], destinations: Sequence) -> Reaction:  # noqa: ANN001
    dest_id = intent.matched_destination_id or ""
    if not dest_id:
        return Reaction("etc")
    if question is not None:
        if question.kind == "confirm" and (dest_id in question.option_ids
                                           or dest_id == question_destination_id(question.text, destinations)):
            return Reaction("yes")
        if question.kind == "open" and dest_id == question_destination_id(question.text, destinations):
            return Reaction("yes")
        if question.kind == "choice" and dest_id in question.option_ids:
            return Reaction("pick", dest_id)
    dest = next((d for d in destinations if d.id == dest_id), None)
    if dest is not None and not dest.is_approachable:
        return Reaction("notavail", dest_id)
    return Reaction("confirm" if intent.need_confirm else "go", dest_id)


def reaction_of(intent, question: Optional[OpenQuestion], destinations: Sequence) -> Reaction:  # noqa: ANN001
    """미션이 받는 의도 → 로봇이 하는 일. question 은 이 말을 들을 때 열려 있던 질문(없으면 None)."""
    kind = question.kind if question is not None else ""
    asked = question.text if question is not None else ""
    said = (getattr(intent, "ask_text", "") or getattr(intent, "reply", "") or "").strip()
    i = intent.intent
    if i == "dismiss":
        return Reaction("no" if question is not None and kind != "open" else "bye")
    if i in ("not_for_robot", "repeat"):
        return Reaction({"not_for_robot": "silent", "repeat": "repeat"}[i])
    if i in ("affirm", "deny"):
        return Reaction("yes" if i == "affirm" else "no")
    if i == "navigate":
        return _navigate(intent, question, destinations)
    if i == "wait":
        minutes = getattr(intent, "wait_minutes", -1) or -1
        if kind == "minutes" and minutes > 0:
            return Reaction("wait", minutes=str(minutes))
        if kind == "yesno":
            return Reaction("no" if END_QUESTION in asked else "yes")
        return Reaction("wait", minutes=str(minutes) if minutes > 0 else "")
    if i in ("cancel", "finish"):
        return Reaction("yes" if kind == "yesno_inverted" or END_QUESTION in asked else "finish")
    if i == "clarify":
        if _CHOICE_TAIL.search(said):
            return Reaction("which", say=said)
        return Reaction("notavail" if said.startswith(NOT_AVAILABLE_HEAD) else "reask", say=said)
    if i == "unknown":
        if is_wake_answer(said):
            return Reaction("wake", say=said)
        if said:
            return Reaction("reask", say=said)
        return Reaction("reask" if question is not None and asked.strip() != WAKE_GREETING else "silent")
    if i == "question":
        return Reaction("answer", say=getattr(intent, "reply", "") or "")
    return Reaction("etc")


ANSWER_READ_KINDS = ("confirm", "choice", "yesno", "yesno_inverted", "minutes")


def mission_reaction(intent, question: Optional[OpenQuestion], destinations: Sequence,  # noqa: ANN001
                     answer_field: bool = False) -> Reaction:
    """미션 파라미터 answer_field 가 켜졌을 때 로봇이 하는 일(계획 Task 3.3 — mission_logic._intent_from_answer 를 옮긴 것)."""
    answer = getattr(intent, "answer", "") or ""
    if not answer_field or question is None or question.kind not in ANSWER_READ_KINDS:
        return reaction_of(intent, question, destinations)
    if not answer:
        return _as_intent(intent, question, destinations)
    value = (getattr(intent, "answer_value", "") or "").strip()
    if answer in ("yes", "no"):
        return Reaction(answer)
    if answer in ("destination", "choice"):
        if not value:
            return Reaction("reask", say=question.text)
        dest = next((d for d in destinations if d.id == value), None)
        if dest is not None and not dest.is_approachable:
            return Reaction("notavail", value)
        if question.kind == "choice" and value in question.option_ids:
            return Reaction("pick", value)
        if question.kind == "confirm" and (value in question.option_ids
                                           or value == question_destination_id(question.text, destinations)):
            return Reaction("reask", say=question.text)
        return Reaction("confirm", value)
    if answer == "minutes":
        return Reaction("wait", minutes=value) if value.isdigit() and int(value) > 0 else Reaction("reask", say=question.text)
    if answer == "unclear":
        said = (getattr(intent, "ask_text", "") or getattr(intent, "reply", "") or "").strip()
        return Reaction("wake", say=said) if is_wake_answer(said) else Reaction("reask", say=question.text)
    return _as_intent(intent, question, destinations)


def _as_intent(intent, question: OpenQuestion, destinations: Sequence) -> Reaction:  # noqa: ANN001
    """answer_field 가 켜진 미션이 의도대로 받을 때. 확인 질문 중 같은 곳의 **제안**(need_confirm)은 승낙이 아니라 같은 질문을"""
    reaction = reaction_of(intent, question, destinations)
    if (question.kind == "confirm" and intent.intent == "navigate" and getattr(intent, "need_confirm", False)
            and reaction.action == "yes"):
        return Reaction("reask", say=question.text)
    return reaction


def _same(key: dict, reaction: Reaction) -> bool:
    want = key.get("action", "")
    if {want, reaction.action} <= {"confirm", "pick"}:
        return str(key.get("target", "")) == reaction.target
    if want != reaction.action:
        return False
    if want == "wait" and str(key.get("minutes", "") or ""):
        return str(key["minutes"]) == reaction.minutes
    return True


def grade_key(key: dict, reaction: Reaction, kind: str = "") -> tuple[str, int]:
    """답지 행(key)과 로봇이 한 일(reaction) → (판정, 깨진 원칙 번호 — 맞거나 채점하지 않으면 0)."""
    if key.get("drop"):
        return "drop", 0
    if key.get("source") == "pending":
        return "pending", 0
    if _same(key, reaction):
        return "ok", 0
    want = key.get("action", "")
    if want == "no" and reaction.action == "bye" and kind == "open":
        return "ok", 0
    principle = 3 if reaction.action == "go" else PRINCIPLE.get(want, 0)
    if reaction.action == "reask":
        return "reask", principle
    if {want, reaction.action} == {"yes", "no"}:
        return "wrong", principle
    if reaction.action in UNASKED or (reaction.action == "yes" and kind != "open"):
        return "wrong", principle
    if reaction.action == "silent" and want not in ("silent", "reask", "wake"):
        return "wrong", principle
    return "mismatch", principle


def load_key(path) -> dict[str, dict]:  # noqa: ANN001
    """answer_key.csv → {ts: 행}."""
    with open(path, encoding="utf-8") as f:
        return {row["ts"]: row for row in csv.DictReader(f)}

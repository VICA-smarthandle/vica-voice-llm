"""새 답지 채점 — 로봇이 한 일(행동 · 대상 · 분 · 할 말)을 답지와 견준다 (설계 2026-11-20 G2 · 8절, 계획 Task 1-3).

답지 한 행 = 로봇이 해야 할 일(~/vica_data/voice_answer_audit_20261113/review2/answer_key.csv). 옛 채점(answer_grade)은
"의도를 맞혔나"를 봤고, 여기는 "원칙을 지켰나"를 본다 — 맞지 않은 행에는 그 행이 지켜야 했던 원칙 번호를 함께 낸다.
reaction_of 는 **미션이 받는 꼴**의 의도를 읽는다(음성 노드가 demote_unasked_confirm · hand_over_question 을 적용한 뒤).
채점: ok = 맞음 · reask = 되묻기 · wrong = 잘못된 행동(정반대 · 묻지 않은 동작 · 조용히 버림) · mismatch = 그 밖 ·
drop = 버린 행 · pending = 사용자가 아직 확인하지 않은 행.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Optional, Sequence

from src.one_voice import _CHOICE_TAIL
from src.question_record import OpenQuestion
from src.replies import WAKE_GREETING
from tools.answer_grade import question_destination_id

NOT_AVAILABLE_HEAD = "제가 아는 목적지 중에서"
# 호출에 답하는 말 — 띄어쓰기·문장부호를 뗀 꼴. 소리 경로의 지금 지시문은 "네?"나 "네, 말씀하세요." 같은 짧은 응답을 허용한다.
# 로봇이 하는 일(부름에 답하고 듣는다)은 같으므로 같은 반응(wake)으로 읽는다. 로봇이 낸 문장만 본다(사람 말의 뜻이 아니다).
WAKE_ANSWERS = ("네", "네말씀하세요", "네말씀해주세요")
END_QUESTION = "마칠까요"
# 묻지 않았는데 하면 잘못된 행동으로 세는 반응(로봇이 움직이거나 무언가를 정한다).
UNASKED = ("go", "confirm", "pick", "wait", "finish", "bye")
# 답지의 행동이 지키려는 원칙(설계 2절의 번호). 반응이 go(확인 없이 출발)면 답지와 상관없이 3번이다.
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
            return Reaction("yes")                     # LLM 이 지은 확인 질문이 묻는 곳
        if question.kind == "choice" and dest_id in question.option_ids:
            return Reaction("pick", dest_id)
    dest = next((d for d in destinations if d.id == dest_id), None)
    if dest is not None and not dest.is_approachable:
        return Reaction("notavail", dest_id)           # 미션이 갈 수 없다고 알린다
    return Reaction("confirm" if intent.need_confirm else "go", dest_id)


def reaction_of(intent, question: Optional[OpenQuestion], destinations: Sequence) -> Reaction:  # noqa: ANN001
    """미션이 받는 의도 → 로봇이 하는 일. question 은 이 말을 들을 때 열려 있던 질문(없으면 None)."""
    kind = question.kind if question is not None else ""
    asked = question.text if question is not None else ""
    said = (getattr(intent, "ask_text", "") or getattr(intent, "reply", "") or "").strip()
    i = intent.intent
    if i in ("not_for_robot", "dismiss", "repeat"):
        return Reaction({"not_for_robot": "silent", "dismiss": "bye", "repeat": "repeat"}[i])
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
        # 말이 없는 unknown — 미션 질문이 열려 있으면 미션이 같은 질문을 한 번 다시 한다. "네?"뿐이면 아무 일도 없다.
        return Reaction("reask" if question is not None and asked.strip() != WAKE_GREETING else "silent")
    if i == "question":
        return Reaction("answer", say=getattr(intent, "reply", "") or "")
    return Reaction("etc")


def _same(key: dict, reaction: Reaction) -> bool:
    want = key.get("action", "")
    # 보기에서 고른 것(pick)과 확인 질문(confirm)은 같은 곳이면 로봇이 하는 일이 같다 — 고른 곳으로 확인 질문을 한다.
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
    principle = 3 if reaction.action == "go" else PRINCIPLE.get(want, 0)
    if reaction.action == "reask":
        return "reask", principle
    if {want, reaction.action} == {"yes", "no"}:
        return "wrong", principle                      # 정반대
    if reaction.action in UNASKED or (reaction.action == "yes" and kind != "open"):
        return "wrong", principle                      # 묻지 않은 동작
    if reaction.action == "silent" and want not in ("silent", "reask", "wake"):
        return "wrong", principle                      # 조용히 버림
    return "mismatch", principle


def load_key(path) -> dict[str, dict]:  # noqa: ANN001
    """answer_key.csv → {ts: 행}."""
    with open(path, encoding="utf-8") as f:
        return {row["ts"]: row for row in csv.DictReader(f)}

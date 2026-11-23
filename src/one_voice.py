"""한 목소리 — 질문은 미션이 말한다(설계 2026-11-14 C2·4.4, 3-가)."""
from __future__ import annotations

import re
from typing import Optional, Sequence

from . import answer_field
from .question_record import OpenQuestion
from .replies import RESUME_CONFIRM, expects_answer
from .schema import DestinationData, VicaIntent

_CHOICE_TAIL = re.compile(r"\s*중에?\s*어디로 갈까요\?$")


def mission_asks_resume(intent: VicaIntent) -> bool:
    """재개 제안을 미션이 묻는가(원칙 3 · 9, 11-18 설계 4.4·4.5, Task 3.4) — 대답 칸이 켜졌을 때(.env VICA_ANSWER=on)만 참."""
    return answer_field.on() and intent.intent == "resume" and bool(intent.need_confirm)


def _choice_option_ids(head: str, destinations: Sequence[DestinationData]) -> list[str]:
    """고르기 질문 앞부분에 이름이 든 목적지 id. 다른 맞은 이름 속에 든 이름("화장실"·"남자 화장실")은 뺀다."""
    named = {d.name for d in destinations if d.name and d.name in head}
    outer = {n for n in named if not any(m != n and n in m for m in named)}
    return [d.id for d in destinations if d.name in outer]


def hand_over_question(intent: VicaIntent, destinations: Sequence[DestinationData]) -> VicaIntent:
    """질문이면 reply 를 비우고 미션에 넘길 칸을 채운 사본, 아니면 그대로 돌려준다."""
    reply = (intent.reply or "").strip()
    if intent.safety_flag == "emergency" or not reply:
        return intent
    if intent.intent == "navigate" and not intent.need_confirm:
        return intent
    tail = _CHOICE_TAIL.search(reply)
    if tail:
        return intent.model_copy(update={
            "reply": "", "ask_text": reply, "ask_kind": "choice",
            "ask_option_ids": _choice_option_ids(reply[:tail.start()], destinations)})
    if intent.intent == "navigate" and intent.need_confirm and intent.matched_destination_id:
        return intent.model_copy(update={"reply": ""})
    if mission_asks_resume(intent):
        return intent.model_copy(update={"reply": ""})
    if intent.intent == "resume" and intent.need_confirm:
        return intent.model_copy(update={"reply": "", "ask_text": RESUME_CONFIRM, "ask_kind": "yesno"})
    if intent.intent == "clarify" or expects_answer(reply):
        return intent.model_copy(update={"reply": "", "ask_text": reply, "ask_kind": "open"})
    return intent


def demote_unasked_confirm(intent: VicaIntent, question: Optional[OpenQuestion],
                           destinations: Sequence[DestinationData]) -> VicaIntent:
    """확인 질문 없이 나온 navigate '확정'을 제안으로 낮춘다 — 미션이 "…로 안내해드릴까요?"를 묻게 된다."""
    if intent.intent != "navigate" or intent.need_confirm or intent.safety_flag == "emergency":
        return intent
    dest = next((d for d in destinations if d.id == intent.matched_destination_id), None)
    if dest is None or not dest.is_approachable:
        return intent
    if question is not None and question.kind == "confirm" and dest.id in question.option_ids:
        return intent
    return intent.model_copy(update={"need_confirm": True, "reply": ""})


def voice_may_speak(intent: VicaIntent, question: Optional[OpenQuestion]) -> bool:
    """음성 노드가 이 의도의 reply 를 말해도 되나. 미션 질문이 열려 있으면 사용자 질문의 답만(설계 4.4)."""
    if intent.safety_flag == "emergency" or question is None:
        return True
    if question.text == RESUME_CONFIRM and not answer_field.on():
        return True
    return intent.intent == "question"


def held_resume_carrier(intent: VicaIntent) -> VicaIntent:
    """확인 전 resume 은 미션에 보내지 않는다(schema.should_forward_intent) — 그 질문 문장만 실어 보낼 의도."""
    return VicaIntent(intent="clarify", need_confirm=False, confidence=1.0, reply="",
                      ask_text=intent.ask_text, ask_kind=intent.ask_kind)

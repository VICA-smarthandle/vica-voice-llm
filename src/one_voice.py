"""한 목소리 — 질문은 미션이 말한다(설계 2026-11-14 C2·4.4, 3-가).

11-17 사용자 "왜 아직도 음성과 미션에서 둘 다 말을 하나요". 음성 노드는 대답을 기다리는 문장을 TTS 로 보내지 않고
VicaIntent 의 ask_text·ask_kind·ask_option_ids 에 실어 미션에 넘긴다. 미션이 말하고 질문 기록·듣기 요청을 낸다.
목적지 확인 질문은 문장을 넘기지 않는다 — 미션이 목적지 confirm_prompt 로 묻는다(결정 A).
ros_node 는 rclpy 없이 시험할 수 없어 판단은 여기 순수 함수로 둔다.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

from .question_record import OpenQuestion
from .replies import RESUME_CONFIRM, expects_answer
from .schema import DestinationData, VicaIntent

# 고르기 질문 끝 — replies.choose_between 의 "… 중 어디로 갈까요?"와 LLM 이 지은 "… 중에 어디로 갈까요?".
_CHOICE_TAIL = re.compile(r"\s*중에?\s*어디로 갈까요\?$")


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
        # 확정 navigate 의 문장은 예전에도 음성이 말하지 않았다(tts_queue.request_for_intent) — 넘기지 않는다.
        # 접근 불가 목적지는 해석기가 확정으로 바꾸며 확인 문구를 붙여 둔다(11-18 실기: 407호 4번).
        return intent
    tail = _CHOICE_TAIL.search(reply)
    if tail:
        return intent.model_copy(update={
            "reply": "", "ask_text": reply, "ask_kind": "choice",
            "ask_option_ids": _choice_option_ids(reply[:tail.start()], destinations)})
    if intent.intent == "navigate" and intent.need_confirm and intent.matched_destination_id:
        return intent.model_copy(update={"reply": ""})
    if intent.intent == "resume" and intent.need_confirm:
        return intent.model_copy(update={"reply": "", "ask_text": reply, "ask_kind": "yesno"})
    if intent.intent == "clarify" or expects_answer(reply):
        return intent.model_copy(update={"reply": "", "ask_text": reply, "ask_kind": "open"})
    return intent


def voice_may_speak(intent: VicaIntent, question: Optional[OpenQuestion]) -> bool:
    """음성 노드가 이 의도의 reply 를 말해도 되나. 미션 질문이 열려 있으면 사용자 질문의 답만(설계 4.4).
    "다시 출발할까요?"는 판단을 음성이 쥔 질문이라(결정 B) 그 대답("알겠습니다. 계속 진행할게요.")도 음성이 말한다."""
    if intent.safety_flag == "emergency" or question is None or question.text == RESUME_CONFIRM:
        return True
    return intent.intent == "question"


def held_resume_carrier(intent: VicaIntent) -> VicaIntent:
    """확인 전 resume 은 미션에 보내지 않는다(schema.should_forward_intent) — 그 질문 문장만 실어 보낼 의도."""
    return VicaIntent(intent="clarify", need_confirm=False, confidence=1.0, reply="",
                      ask_text=intent.ask_text, ask_kind=intent.ask_kind)

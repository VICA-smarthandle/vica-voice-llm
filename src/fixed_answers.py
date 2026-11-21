"""정해진 대답 둘(2026-11-21 voice-principles-intent-recut 4단계, 설계 6.4) — 순수 함수만 둔다.

"다시 말해줘"(repeat, 원칙 28)·"어디 갈 수 있어?"(places, 원칙 29)는 LLM 이 값을 고르기만 하고 문장은 코드가 만든다.
`.env` VICA_TALK=on 이고 재단(VICA_RECUT=on)이 함께 켜졌을 때만 움직인다(기본 꺼짐 = 지금과 같다). 로컬(Ollama)이 답하는
호출에는 싣지 않는다(원칙 16) — 지시문 두 줄은 재단을 싣는 지시문에만 붙고, 로컬이 답한 결과는 이 뒤처리를 지나지 않는다.
ROS·LLM 을 모른다 — 장치 없이 시험된다.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

from . import intent_recut
from .question_record import OpenQuestion
from .replies import RETRY_PROMPT, WAKE_GREETING, places_sentence
from .schema import DestinationData, VicaIntent

REPEAT = "repeat"
PLACES = "places"
# 미션이 물어 쥐고 있는 질문 종류 — 여기에 "다시 말해줘"면 미션이 그 질문을 다시 말한다(미션 Task 4.3, _open_question_name).
# open("네?"·음성이 넘긴 되묻기)과 choice(음성이 넘긴 고르기 질문 — 미션은 choice 를 스스로 묻지 않는다)는 미션이 쥐고 있지 않아
# repeat 에 말이 없다 — 음성이 마지막 말을 다시 하고, 질문 꼴이면 hand_over_question 이 같은 종류로 다시 넘긴다.
MISSION_KINDS = ("confirm", "yesno", "yesno_inverted", "minutes")

# 지시문 두 벌(글자·소리)이 재단의 의도 정의(intent_recut.PROMPT_INTENTS) 바로 뒤에 싣는 두 줄. 켰을 때만 들어간다.
PROMPT_LINES = """- repeat: 로봇이 방금 한 말을 다시 해 달라고 한다("다시 말해줘", "뭐라고?", "못 들었어"). reply 는 빈 문자열.
- places: 어디로 갈 수 있는지, 무엇을 할 수 있는지 묻는다("어디 갈 수 있어?", "뭐 할 수 있어?"). reply 는 빈 문자열 — 문장은 시스템이 만든다."""


def on() -> bool:
    """스위치. `.env` VICA_TALK=on 이고 재단(intent_recut.on())이 켜졌을 때만 참 — 4단계는 2단계 위에서만 켜진다. 매번 읽는다."""
    return os.environ.get("VICA_TALK", "").strip().lower() == "on" and intent_recut.on()


def places_names(destinations: Sequence[DestinationData]) -> list[str]:
    """갈 수 있는 곳(is_approachable)의 이름, 목록 순서(원칙 18 — 갈 수 없는 곳은 말하지 않는다)."""
    return [d.name for d in destinations if d.name and d.is_approachable]


def last_said(robot_recent: Sequence[tuple[float, str]]) -> str:
    """로봇의 마지막 말(재생이 끝난 문장, /vica/tts_done). 호출 응답 "네?"는 건너뛴다. 없으면 빈 글자."""
    for _, text in reversed(list(robot_recent)):
        line = (text or "").strip()
        if line and line != WAKE_GREETING:
            return line
    return ""


def settle_repeat(question: Optional[OpenQuestion], robot_recent: Sequence[tuple[float, str]]) -> VicaIntent:
    """repeat 를 정한다(원칙 28 · 9). 미션이 쥔 질문이 열려 있으면 repeat 를 미션에 보내고(미션이 다시 말한다 — 음성은 말하지
    않는다), 아니면 음성이 마지막 말을 다시 한다. 다시 할 말이 없으면 unknown — 되묻는다. 대답 칸은 싣지 않는다."""
    if question is not None and question.kind in MISSION_KINDS:
        return VicaIntent(intent=REPEAT, reply="", need_confirm=False)
    said = last_said(robot_recent)
    if said:
        return VicaIntent(intent="question", reply=said, need_confirm=False)
    return VicaIntent(intent="unknown", reply=RETRY_PROMPT, need_confirm=False)


def early(draft, destinations: Sequence[DestinationData]) -> Optional[VicaIntent]:
    """뒤처리 맨 앞(원칙 28·29): repeat 는 문장·목적지 없이 그대로(노드가 settle_repeat 로 정한다), places 는 갈 수 있는 곳을
    알리는 question 으로 돌려보낸다. 꺼졌거나 다른 값이거나 갈 수 있는 곳이 없으면 None(as_known 이 지금 길로 바꾼다)."""
    if not on():
        return None
    kind = getattr(draft, "intent", "")
    confidence = getattr(draft, "confidence", None) or 0.0
    if kind == REPEAT:
        return VicaIntent(intent=REPEAT, reply="", need_confirm=False, confidence=confidence)
    if kind == PLACES:
        sentence = places_sentence(places_names(destinations))
        if sentence:
            return VicaIntent(intent="question", reply=sentence, need_confirm=False, confidence=confidence)
    return None


def as_known(draft):
    """갈 수 있는 곳이 하나도 없어 early 가 문장을 만들지 못한 places 를 지금 길의 값으로 읽는다(VicaIntent 에 places 는 없다).

    LLM 이 reply 를 썼으면 question(그 말 그대로), 지시대로 비웠으면 unknown — 재단 뒤처리가 되묻는 말(RETRY_PROMPT)로 둔다.
    question 의 빈 reply 는 글자 경로에선 RETRY_PROMPT, 소리 경로에선 침묵이 돼 두 경로가 갈린다. 다른 값은 그대로 돌려준다.
    """
    if getattr(draft, "intent", "") != PLACES:
        return draft
    return draft.model_copy(update={"intent": "question" if (draft.reply or "").strip() else "unknown"})

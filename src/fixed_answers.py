"""정해진 대답(2026-11-21 voice-principles-intent-recut 4단계, 설계 6.4) — 순수 함수만 둔다."""
from __future__ import annotations

import os
from typing import Iterator, Optional, Sequence

from . import answer_field, intent_recut, vica_facts
from .mission_phrases import WAIT_BEACON
from .question_record import OpenQuestion
from .replies import RETRY_PROMPT, WAKE_GREETING, expects_answer, places_sentence
from .schema import DestinationData, VicaIntent

REPEAT = "repeat"
PLACES = "places"
_VOICE_HANDED_KINDS = ("choice",)

PROMPT_LINES = """- repeat: 로봇이 방금 한 말을 다시 해 달라고 한다("다시 말해줘", "뭐라고?", "못 들었어", "방금 뭐라고 했어?"). reply 는 빈 문자열 — 다시 할 말은 시스템이 고른다(직접 옮겨 말하지 않는다).
- places: 어디로 갈 수 있는지 묻는다("어디 갈 수 있어?", "갈 수 있는 곳 알려줘"). 이 물음은 question 으로 답하지 않고 반드시 places 다 — reply 는 빈 문자열, 문장은 시스템이 만든다(목적지 이름을 직접 늘어놓거나 "알려드릴게요"라고 하지 않는다). 무엇을 할 수 있는지 물으면 places 가 아니라 question 이고 about 은 abilities 다."""


def on() -> bool:
    """스위치. `.env` VICA_TALK=on 이고 재단(intent_recut.on())·대답 칸(answer_field.on())이 켜졌을 때만 참 — 4단계는 2·3단계"""
    return (os.environ.get("VICA_TALK", "").strip().lower() == "on"
            and intent_recut.on() and answer_field.on())


_BUNDLE_NOTE = "다섯 개(VICA_RECUT·VICA_ANSWER + 미션 answer_field·reask_ladder·wake_revive)는 한 묶음으로 켠다"


def switch_warning() -> str:
    """음성 스위치 조합이 어긋났으면 경고 한 줄, 맞으면 빈 글자(최종 검토 I4). 동작은 막지 않는다 — 의도 노드가 기동할 때 로그로만 낸다."""
    recut, answer = intent_recut.on(), answer_field.on()
    talk = os.environ.get("VICA_TALK", "").strip().lower() == "on"
    wrong = []
    if answer and not recut:
        wrong.append("VICA_ANSWER=on 인데 VICA_RECUT 꺼짐")
    if talk and not (recut and answer):
        off = "·".join(key for key, value in (("VICA_RECUT", recut), ("VICA_ANSWER", answer)) if not value)
        wrong.append(f"VICA_TALK=on 인데 {off} 꺼짐(VICA_TALK 은 움직이지 않는다)")
    if not wrong:
        return ""
    return f"[스위치 조합] {' · '.join(wrong)} — {_BUNDLE_NOTE}"


def places_names(destinations: Sequence[DestinationData]) -> list[str]:
    """갈 수 있는 곳(is_approachable)의 이름, 목록 순서(원칙 18 — 갈 수 없는 곳은 말하지 않는다)."""
    return [d.name for d in destinations if d.name and d.is_approachable]


def _said_lines(robot_recent: Sequence[tuple[float, str]]) -> Iterator[str]:
    """최근 로봇 말(재생이 끝난 문장, /vica/tts_done), 새것부터. 호출 응답 "네?"와 대기 알림(M3)은 뺀다 — 노드도 마지막 로봇 말·"""
    for _, text in reversed(list(robot_recent)):
        line = (text or "").strip()
        if line and line not in (WAKE_GREETING, WAIT_BEACON):
            yield line


def last_said(robot_recent: Sequence[tuple[float, str]]) -> str:
    """로봇의 마지막 말. "네?"·대기 알림은 건너뛴다. 없으면 빈 글자."""
    return next(_said_lines(robot_recent), "")


def _to_mission(ask_text: str = "", ask_kind: str = "", option_ids: Sequence[str] = ()) -> VicaIntent:
    """미션에 보내는 repeat — 말은 미션이 한다: 쥐거나 접어 둔 질문이 있으면 그것을, 없으면 실려 간 글(ask_text)을 그 종류로."""
    return VicaIntent(intent=REPEAT, reply="", need_confirm=False, ask_text=ask_text, ask_kind=ask_kind,
                      ask_option_ids=list(option_ids))


def _voice_says(text: str) -> VicaIntent:
    """음성이 다시 말한다 — 지금 question 길(질문 꼴이면 hand_over_question 이 미션에 넘긴다)."""
    return VicaIntent(intent="question", reply=text, need_confirm=False)


def settle_repeat(question: Optional[OpenQuestion], robot_recent: Sequence[tuple[float, str]]) -> VicaIntent:
    """repeat 를 정한다(원칙 28 · 9, 수정 1·2차). 누가 쥔 질문인지는 미션이 가른다 — 음성은 종류로 짐작하지 않는다."""
    if question is None or (question.text or "").strip() == WAKE_GREETING:
        said = last_said(robot_recent)
        if said:
            return _to_mission(said, "open") if expects_answer(said) else _voice_says(said)
        if question is None:
            return VicaIntent(intent="unknown", reply=RETRY_PROMPT, need_confirm=False)
        return _to_mission()
    if question.kind in _VOICE_HANDED_KINDS:
        return _voice_says(question.text)
    return _to_mission(question.text, question.kind, question.option_ids)


def early(draft, destinations: Sequence[DestinationData]) -> Optional[VicaIntent]:
    """뒤처리 맨 앞(원칙 28·29·25): repeat 는 문장·목적지 없이 그대로(노드가 settle_repeat 로 정한다), places 는 갈 수 있는 곳을"""
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
    if kind == "question":
        about = getattr(draft, "about", "")
        sentence = vica_facts.ABOUT_SENTENCES.get(about) if isinstance(about, str) else None
        if sentence:
            return VicaIntent(intent="question", reply=sentence, need_confirm=False, confidence=confidence)
    return None


def as_known(draft):
    """갈 수 있는 곳이 하나도 없어 early 가 문장을 만들지 못한 places 를 지금 길의 값으로 읽는다(VicaIntent 에 places 는 없다)."""
    if getattr(draft, "intent", "") != PLACES:
        return draft
    return draft.model_copy(update={"intent": "question" if (draft.reply or "").strip() else "unknown"})

"""대답 칸(answer) — 열린 질문에 대한 이 말의 대답 값 (설계 2026-11-18 answer-interpretation 4.1~4.3, 3-나-2 1단계).

LLM 이 열린 질문(질문 기록 /vica/question)의 종류를 알고 answer 를 함께 낸다. 1단계에서는 로그와 재생 채점에만 쓴다 —
미션으로는 나가지 않는다. 전부 `.env` VICA_ANSWER=on 일 때만 움직인다(기본 꺼짐 = 예전 동작 그대로).
여기는 ROS·LLM 을 모르는 순수 판단만 둔다 — 장치 없이 시험된다.
"""
from __future__ import annotations

import os
from typing import Literal, Optional

from pydantic import BaseModel, Field, create_model

from .handle_mode import AFFIRMATIVES, NEGATIVES, SOFT_AFFIRMATIVES, normalize_short_reply
from .question_record import OpenQuestion
from .schema import VicaIntent

ANSWERS = ("yes", "no", "destination", "choice", "minutes", "other", "unclear")

# 스키마 칸 설명 — 세 경로(로컬 구조화 출력 · set_intent 인자 · Realtime 인자)가 같은 글을 쓴다. 지시문 본문은 고치지 않는다.
ANSWER_DESCRIPTION = (
    "[지금 상황]에 '로봇이 물은 질문' 줄이 있을 때만 채운다 — 그 질문에 대한 이 말의 대답. "
    "yes(승낙) · no(거절) · destination(다른 곳으로 가자 — destination_candidate 에 그곳) · "
    "choice(고르기 질문의 보기 중 하나 — destination_candidate 에 그곳) · minutes(시간 — wait_minutes 에 분) · "
    "other(질문과 상관없는 명령이나 질문) · unclear(못 알아들음). 그 줄에 적힌 값만 쓴다. 줄이 없으면 빈 문자열.")

# 질문 종류마다 받는 대답(바탕 설계 2026-11-14 4.1 표). 공통 넷 앞에 그 종류의 값 하나.
_COMMON = ("no", "destination", "other", "unclear")
_OWN = {"open": "yes", "confirm": "yes", "yesno": "yes", "yesno_inverted": "yes",
        "choice": "choice", "minutes": "minutes"}
_KIND_KO = {"open": "열린 질문", "confirm": "확인 질문", "choice": "고르기 질문", "yesno": "네·아니요 질문",
            "yesno_inverted": "부정 질문", "minutes": "시간 질문"}
_YES_NO_KINDS = ("confirm", "yesno", "yesno_inverted")
_INVERTED_NOTE = ". 필요 없다는 뜻이면 yes, 필요하다는 뜻이면 no"   # 지금 지시문의 affirm·deny 기준과 같은 방향
_HEADER = "\n[지금 상황]\n"

_narrowed_cache: dict = {}


def on() -> bool:
    """스위치. `.env` VICA_ANSWER=on 일 때만 참이다. 매번 읽는다 — 시험이 monkeypatch 로 켜고 끌 수 있게."""
    return os.environ.get("VICA_ANSWER", "").strip().lower() == "on"


def allowed(kind: str) -> tuple[str, ...]:
    """이 종류의 질문이 받는 answer 값. 모르는 종류(옛 미션·빈 값)는 전부."""
    own = _OWN.get(kind)
    return (own, *_COMMON) if own else ANSWERS


def question_line(question: Optional[OpenQuestion]) -> str:
    """[지금 상황]에 넣을 한 줄 — 로봇이 물은 질문과 그 종류가 받는 값. 열린 질문이 없으면 빈 글자."""
    if question is None:
        return ""
    note = _INVERTED_NOTE if question.kind == "yesno_inverted" else ""
    return (f'- 로봇이 물은 질문: "{question.text}" '
            f"({_KIND_KO.get(question.kind, '질문')} — answer 는 {'·'.join(allowed(question.kind))} 중 하나{note})")


def with_question(block: str, question: Optional[OpenQuestion]) -> str:
    """[지금 상황] 글 끝에 질문 줄을 붙인다. 꺼졌거나 열린 질문이 없으면 그대로, 글이 비면 머리글을 단다."""
    if not on() or question is None:
        return block
    line = question_line(question)
    if not block:
        return _HEADER + line + "\n"
    return block.rstrip("\n") + "\n" + line + "\n"


def short_answer(text: str, question: Optional[OpenQuestion], local: bool = False) -> Optional[tuple[str, str]]:
    """이미 있는 짧은 대답 규칙만(설계 4.3-3): 통째로 네/아니요, N분. 질문 종류에 맞을 때만 (answer, 값), 아니면 None.

    새 낱말 목록·정규식을 두지 않는다 — handle_mode 의 낱말과 parse_wait_minutes 만 쓴다. "안내가 필요 없으신가요?"
    (yesno_inverted)도 들은 대로 yes/no 다 — 뜻은 미션이 질문 종류로 읽는다.
    """
    if not on() or question is None:
        return None
    if question.kind in _YES_NO_KINDS:
        word = normalize_short_reply(text or "")
        if word in NEGATIVES:
            return ("no", "")
        if word in AFFIRMATIVES:
            return ("yes", "")
        if word in SOFT_AFFIRMATIVES:
            # 혼잣소리('음')는 확인 질문의 승낙이 아니다(2026-09-28) — 움직이지 않는 자리에서만 승낙.
            return ("unclear", "") if question.kind == "confirm" else ("yes", "")
        return None
    if question.kind == "minutes":
        from .langchain_intent_parser import parse_wait_minutes   # 지연 import — 해석기가 이 모듈을 import 한다

        minutes = parse_wait_minutes(text or "", range_factor=1.0 if local else 1.5)
        return ("minutes", str(minutes)) if minutes else None
    return None


def fill(intent: VicaIntent, question: Optional[OpenQuestion], ruled: Optional[tuple[str, str]] = None,
         llm_answer: str = "") -> VicaIntent:
    """의도에 question_id·answer·answer_value 를 채운 사본. 꺼짐·질문 없음·긴급이면 받은 의도 그대로.

    규칙 대답(ruled)이 있으면 그것, 없으면 LLM 이 낸 값이다. 허용 밖의 값, 목적지 id 없는 destination·choice,
    분 없는 minutes 는 unclear 로 바꾼다(설계 4.1 — 외부 API 는 형식을 강제하지 않는다).
    """
    if not on() or question is None or intent.safety_flag == "emergency":
        return intent
    if ruled is not None:
        answer, value = ruled
    else:
        answer, value = (llm_answer or "").strip(), ""
        if answer and answer not in allowed(question.kind):
            answer = "unclear"
        elif answer in ("destination", "choice"):
            value = intent.matched_destination_id or ""
            answer = answer if value else "unclear"
        elif answer == "minutes":
            value = str(intent.wait_minutes) if intent.wait_minutes > 0 else ""
            answer = answer if value else "unclear"
    return intent.model_copy(update={"question_id": question.id, "answer": answer, "answer_value": value})


def narrowed(base: type[BaseModel], kind: str) -> type[BaseModel]:
    """로컬 구조화 출력용 — base 에 answer 칸(그 질문 종류의 값만, 꼭 채움)을 더한 스키마. kind 가 비면 base 그대로."""
    if not kind:
        return base
    key = (base, kind)
    if key not in _narrowed_cache:
        _narrowed_cache[key] = create_model(
            f"{base.__name__}_{kind}", __base__=base,
            answer=(Literal[allowed(kind)], Field(description=ANSWER_DESCRIPTION)))
    return _narrowed_cache[key]


def log_line(intent: VicaIntent, question: Optional[OpenQuestion]) -> str:
    """발화마다 남기는 한 줄 — 이 말에 채워진 대답 값."""
    if not intent.answer:
        return "[대답] 없음"
    kind = question.kind if question is not None else "-"
    return f"[대답] answer={intent.answer} value={intent.answer_value or '-'} 질문={intent.question_id} kind={kind}"

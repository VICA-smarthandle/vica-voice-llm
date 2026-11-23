"""대답 칸(answer) — 열린 질문에 대한 이 말의 대답 값 (설계 2026-11-18 answer-interpretation 4.1~4.3, 3-나-2 1단계)."""
from __future__ import annotations

import os
from typing import Literal, Optional

from pydantic import BaseModel, Field, create_model

from .handle_mode import AFFIRMATIVES, NEGATIVES, normalize_short_reply
from .question_record import OpenQuestion
from .schema import VicaIntent

ANSWERS = ("yes", "no", "destination", "choice", "minutes", "other", "unclear")

ANSWER_DESCRIPTION = (
    "[지금 상황]에 '로봇이 물은 질문' 줄이 있을 때만 채운다 — 그 질문에 대한 이 말의 대답. "
    "yes(승낙) · no(거절) · destination(다른 곳으로 가자 — destination_candidate 에 그곳) · "
    "choice(고르기 질문의 보기 중 하나 — destination_candidate 에 그곳) · minutes(시간 — wait_minutes 에 분) · "
    "other(질문과 상관없는 명령이나 질문) · unclear(못 알아들음). 그 줄에 적힌 값만 쓴다. 줄이 없으면 빈 문자열.")

_COMMON = ("no", "destination", "other", "unclear")
_OWN = {"open": "yes", "confirm": "yes", "yesno": "yes", "yesno_inverted": "yes",
        "choice": "choice", "minutes": "minutes"}
_KIND_KO = {"open": "열린 질문", "confirm": "확인 질문", "choice": "고르기 질문", "yesno": "네·아니요 질문",
            "yesno_inverted": "부정 질문", "minutes": "시간 질문"}
_MEANING = {"yes": "승낙", "no": "거절", "destination": "다른 곳으로 가자(destination_candidate 에 그곳)",
            "choice": "보기 중 하나를 고름(destination_candidate 에 그곳)", "minutes": "시간을 말함(wait_minutes 에 분)",
            "other": "질문과 상관없는 명령이나 질문", "unclear": "못 알아들음"}
_YES_NO_KINDS = ("confirm", "yesno", "yesno_inverted")
_GO_WORDS = ("가자", "가줘")
_MINUTES_CORRECTS = ("", "minutes", "unclear")
_MINUTES_INTENTS = ("wait",)
_NOT_FOR_ROBOT = "not_for_robot"
_YES_NO_INTENT = {"yes": "affirm", "no": "deny"}
_CROSSED_INTENT = {"yes": "deny", "no": "affirm"}
_OPPOSITE = {"yes": "no", "no": "yes"}
_YES_NO_INTENTS = ("affirm", "deny")
_INVERTED_NOTE = ". 필요 없다는 뜻이면 yes, 필요하다는 뜻이면 no"
_NOT_FOR_ROBOT_NOTE = " 이 줄이 [판단 순서]의 '로봇이 방금 한 말'이다. not_for_robot 으로 판단한 말이면 answer 는 빈 문자열로 둔다."
_HEADER = "\n[지금 상황]\n"

_narrowed_cache: dict = {}


def on() -> bool:
    """스위치. `.env` VICA_ANSWER=on 일 때만 참이다. 매번 읽는다 — 시험이 monkeypatch 로 켜고 끌 수 있게."""
    return os.environ.get("VICA_ANSWER", "").strip().lower() == "on"


def local_on() -> bool:
    """로컬(Ollama)에도 질문 줄·대답 칸을 줄지. 대답 칸이 켜져 있고 `.env` VICA_ANSWER_LOCAL=on 일 때만 참이다."""
    return on() and os.environ.get("VICA_ANSWER_LOCAL", "").strip().lower() == "on"


def _recut_on() -> bool:
    """재단 스위치(VICA_RECUT). intent_recut.on() 과 같은 판정 — 그 모듈이 이 모듈을 import 해서 여기서 직접 읽는다(순환 import)."""
    return os.environ.get("VICA_RECUT", "").strip().lower() == "on"


def allowed(kind: str) -> tuple[str, ...]:
    """이 종류의 질문이 받는 answer 값. 모르는 종류(옛 미션·빈 값)는 전부."""
    own = _OWN.get(kind)
    return (own, *_COMMON) if own else ANSWERS


def question_line(question: Optional[OpenQuestion], recut: Optional[bool] = None) -> str:
    """[지금 상황]에 넣을 한 줄 — 로봇이 물은 질문과 그 종류가 받는 값. 열린 질문이 없으면 빈 글자."""
    if question is None:
        return ""
    recut = _recut_on() if recut is None else recut
    note = _INVERTED_NOTE if question.kind == "yesno_inverted" else ""
    values = allowed(question.kind)
    meanings = " · ".join(f"{v} {_MEANING[v]}" for v in values)
    return (f'- 로봇이 물은 질문: "{question.text}" '
            f"({_KIND_KO.get(question.kind, '질문')} — answer 는 {'·'.join(values)} 중 하나: {meanings}{note})"
            f"{_NOT_FOR_ROBOT_NOTE if recut else ''}")


def with_question(block: str, question: Optional[OpenQuestion], recut: Optional[bool] = None) -> str:
    """[지금 상황] 글 끝에 질문 줄을 붙인다. 꺼졌거나 열린 질문이 없으면 그대로, 글이 비면 머리글을 단다. recut 은 question_line 과 같다."""
    if not on() or question is None:
        return block
    recut = _recut_on() if recut is None else recut
    if recut and question.kind == "open":
        return block
    line = question_line(question, recut)
    if not block:
        return _HEADER + line + "\n"
    return block.rstrip("\n") + "\n" + line + "\n"


def short_answer(text: str, question: Optional[OpenQuestion], local: bool = False) -> Optional[tuple[str, str]]:
    """이미 있는 짧은 대답 규칙만(설계 4.3-3): 통째로 네/아니요, N분. 질문 종류에 맞을 때만 (answer, 값), 아니면 None."""
    if not on() or question is None:
        return None
    if question.kind in _YES_NO_KINDS:
        from .langchain_intent_parser import _CANCEL_WORDS

        word = normalize_short_reply(text or "")
        if word in _CANCEL_WORDS:
            return None
        if word in NEGATIVES:
            return ("no", "")
        if word in AFFIRMATIVES:
            return None if word in _GO_WORDS and question.kind != "confirm" else ("yes", "")
        return None
    if question.kind == "minutes":
        from .langchain_intent_parser import parse_wait_minutes

        minutes = parse_wait_minutes(text or "", range_factor=1.0 if local else 1.5)
        return ("minutes", str(minutes)) if minutes else None
    return None


def standing(ruled: Optional[tuple[str, str]], intent_name: str, llm_answer: str) -> bool:
    """규칙 대답(ruled)이 서는가(Task 3.5b 규칙 1, 원칙 6 — 바탕 설계 D11). 글자·소리 경로가 같이 쓴다."""
    if ruled is None:
        return False
    llm = (llm_answer or "").strip()
    if ruled[0] in _YES_NO_INTENT:
        return llm == ruled[0] or (llm in ("", "unclear") and intent_name == _YES_NO_INTENT[ruled[0]])
    return llm in _MINUTES_CORRECTS and intent_name in _MINUTES_INTENTS


def _ruled(intent: VicaIntent, question: OpenQuestion, ruled: tuple[str, str]) -> VicaIntent:
    """선 규칙 대답을 실은 사본(Task 3.5b 규칙 2). 대답 칸을 채우고, 분이면 wait_minutes 도 규칙의 분으로 맞춘다(수정 1차 —"""
    answer, value = ruled
    update = {"question_id": question.id, "answer": answer, "answer_value": value}
    if answer == "minutes":
        update.update(wait_minutes=int(value))
    return intent.model_copy(update=update)


def _not_dropped(intent: VicaIntent, question: OpenQuestion) -> VicaIntent:
    """원칙 6·7(수정 2·3차 규칙 3): 대답으로 보이는 말(네/아니요·분)이 not_for_robot 으로 왔다 — 버리지도(노드가 말없이 버린다)"""
    return intent.model_copy(update={
        "intent": "unknown", "reply": "", "ask_text": "", "ask_kind": "", "need_confirm": False,
        "destination_candidate": None, "matched_destination_id": "",
        "question_id": question.id, "answer": "unclear", "answer_value": ""})


def _crossed(llm_answer: str, intent_name: str, ruled: Optional[tuple[str, str]]) -> bool:
    """원칙 3·6(최종 검토 I1 ①②): LLM 대답(yes·no)이 정면으로 어긋나는가 — ① 의도가 그 반대(yes + deny · no + affirm), 또는"""
    if llm_answer not in _OPPOSITE:
        return False
    if intent_name == _CROSSED_INTENT[llm_answer]:
        return True
    return ruled is not None and ruled[0] == _OPPOSITE[llm_answer]


def _elsewhere(intent: VicaIntent, question: OpenQuestion) -> bool:
    """원칙 3(최종 검토 I1 ③): 확인 질문(kind confirm)이 열렸는데 그 질문의 후보(option_ids)가 아닌 목적지로 가는 navigate 인가."""
    return (question.kind == "confirm" and bool(question.option_ids) and intent.intent == "navigate"
            and bool(intent.matched_destination_id) and intent.matched_destination_id not in question.option_ids)


def _settled(intent: VicaIntent, question: OpenQuestion) -> bool:
    """원칙 3·6(최종 검토 I2): 승낙·거절, 또는 확정(need_confirm 거짓) navigate·resume — 미션이 의도대로 받으면 출발·재출발한다."""
    if intent.intent in _YES_NO_INTENTS:
        return True
    if intent.intent == "navigate":
        return not intent.need_confirm and question.kind == "confirm"
    return intent.intent == "resume" and not intent.need_confirm


def fill(intent: VicaIntent, question: Optional[OpenQuestion], ruled: Optional[tuple[str, str]] = None,
         llm_answer: str = "") -> VicaIntent:
    """의도에 question_id·answer·answer_value 를 채운 사본. 꺼짐·질문 없음·긴급이면 받은 의도 그대로."""
    if not on() or question is None or intent.safety_flag == "emergency":
        return intent
    if ruled is not None and intent.intent == _NOT_FOR_ROBOT:
        return _not_dropped(intent, question)
    if standing(ruled, intent.intent, llm_answer):
        return _ruled(intent, question, ruled)
    answer, value = (llm_answer or "").strip(), ""
    if _crossed(answer, intent.intent, ruled):
        answer = "unclear"
    elif answer == "yes" and _elsewhere(intent, question):
        answer = "destination"
    elif (answer == "unclear" and intent.intent not in ("unknown", "clarify", _NOT_FOR_ROBOT)
          and not _settled(intent, question)):
        answer = "other"
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

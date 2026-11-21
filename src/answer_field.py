"""대답 칸(answer) — 열린 질문에 대한 이 말의 대답 값 (설계 2026-11-18 answer-interpretation 4.1~4.3, 3-나-2 1단계).

LLM 이 열린 질문(질문 기록 /vica/question)의 종류를 알고 answer 를 함께 낸다. 메시지에 칸이 있으면 미션으로 나간다
(ros_convert, Task 3.2). 이미 있는 짧은 대답 규칙(short_answer)은 LLM 의 읽기를 뒤집지 않는다 — 같은 읽기일 때 대답 칸을
채우고(분은 LLM 도 wait 로 읽었을 때 값만 바로잡고), 대답으로 보이는 not_for_robot 은 되묻게 한다(standing·fill, Task 3.5b).
전부 `.env` VICA_ANSWER=on 일 때만 움직인다(기본 꺼짐 = 예전 동작 그대로).
여기는 ROS·LLM 을 모르는 순수 판단만 둔다 — 장치 없이 시험된다.
"""
from __future__ import annotations

import os
from typing import Literal, Optional

from pydantic import BaseModel, Field, create_model

from .handle_mode import AFFIRMATIVES, NEGATIVES, normalize_short_reply
from .question_record import OpenQuestion
from .schema import VicaIntent

ANSWERS = ("yes", "no", "destination", "choice", "minutes", "other", "unclear")

# 스키마 칸 설명 — gpt(구조화 출력·set_intent 인자)와 Realtime(도구 인자)이 본다. 로컬(Ollama)은 스키마를 출력 문법으로만
# 쓰고 이 글을 모델에 보여 주지 않는다 — 그래서 값의 뜻은 질문 줄(question_line)에도 적는다. 지시문 본문은 고치지 않는다.
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
# 질문 줄에 적는 값의 뜻. 2026-11-19 로컬 재생: 뜻 없이 값 이름만 주자 midm2-mini 가 "네?"·고르기 질문의 대답
# 대부분에 no 를 냈다(380행 중 171) — answer 를 '대답인가 아닌가'로 읽은 것으로 보인다(검토 I2).
_MEANING = {"yes": "승낙", "no": "거절", "destination": "다른 곳으로 가자(destination_candidate 에 그곳)",
            "choice": "보기 중 하나를 고름(destination_candidate 에 그곳)", "minutes": "시간을 말함(wait_minutes 에 분)",
            "other": "질문과 상관없는 명령이나 질문", "unclear": "못 알아들음"}
_YES_NO_KINDS = ("confirm", "yesno", "yesno_inverted")
# 분 규칙이 서는 LLM 의 읽기(원칙 6, standing) — 대답이 이 셋이고 의도가 wait 일 때만(수정 3차: 값 바로잡기뿐).
_MINUTES_CORRECTS = ("", "minutes", "unclear")
_MINUTES_INTENTS = ("wait",)
# intent_recut.NOT_FOR_ROBOT 과 같은 글자 — intent_recut 이 이 모듈을 import 해서 글자 그대로 둔다(순환 import).
_NOT_FOR_ROBOT = "not_for_robot"
# 네/아니요 규칙 대답이 서는 의도(LLM 대답이 비었을 때, 수정 2차) — 같은 읽기일 때만 대답 칸을 채운다.
_YES_NO_INTENT = {"yes": "affirm", "no": "deny"}
_INVERTED_NOTE = ". 필요 없다는 뜻이면 yes, 필요하다는 뜻이면 no"   # 지금 지시문의 affirm·deny 기준과 같은 방향
_HEADER = "\n[지금 상황]\n"

_narrowed_cache: dict = {}


def on() -> bool:
    """스위치. `.env` VICA_ANSWER=on 일 때만 참이다. 매번 읽는다 — 시험이 monkeypatch 로 켜고 끌 수 있게."""
    return os.environ.get("VICA_ANSWER", "").strip().lower() == "on"


def local_on() -> bool:
    """로컬(Ollama)에도 질문 줄·대답 칸을 줄지. 대답 칸이 켜져 있고 `.env` VICA_ANSWER_LOCAL=on 일 때만 참이다.

    꺼져 있으면(기본) 로컬 호출은 지시문에 질문 줄이 없고 스키마에 answer 칸이 없다 — 규칙이 정한 대답(short_answer)만 채운다
    (설계 2026-11-18 5.1: midm2-mini 는 대답 칸을 주면 의도까지 흔들렸다). 매번 읽는다 — on() 과 같다.
    """
    return on() and os.environ.get("VICA_ANSWER_LOCAL", "").strip().lower() == "on"


def allowed(kind: str) -> tuple[str, ...]:
    """이 종류의 질문이 받는 answer 값. 모르는 종류(옛 미션·빈 값)는 전부."""
    own = _OWN.get(kind)
    return (own, *_COMMON) if own else ANSWERS


def question_line(question: Optional[OpenQuestion]) -> str:
    """[지금 상황]에 넣을 한 줄 — 로봇이 물은 질문과 그 종류가 받는 값. 열린 질문이 없으면 빈 글자."""
    if question is None:
        return ""
    note = _INVERTED_NOTE if question.kind == "yesno_inverted" else ""
    values = allowed(question.kind)
    meanings = " · ".join(f"{v} {_MEANING[v]}" for v in values)
    return (f'- 로봇이 물은 질문: "{question.text}" '
            f"({_KIND_KO.get(question.kind, '질문')} — answer 는 {'·'.join(values)} 중 하나: {meanings}{note})")


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

    새 낱말 목록·정규식을 두지 않는다 — handle_mode 의 낱말과 parse_wait_minutes 만 쓴다(취소 낱말은 해석기의
    _CANCEL_WORDS 로 빼기만 한다 — 뜻이 질문에 따라 갈린다). "안내가 필요 없으신가요?"
    (yesno_inverted)도 들은 대로 yes/no 다 — 뜻은 미션이 질문 종류로 읽는다. 혼잣소리('음'·'어어', SOFT_AFFIRMATIVES)는
    여기서 정하지 않는다 — yesno 에는 "다시 출발할까요?"처럼 로봇이 움직이는 질문도 있다(2026-09-28, 검토 I4).
    """
    if not on() or question is None:
        return None
    if question.kind in _YES_NO_KINDS:
        from .langchain_intent_parser import _CANCEL_WORDS   # 지연 import — 해석기가 이 모듈을 import 한다

        word = normalize_short_reply(text or "")
        if word in _CANCEL_WORDS:
            # 원칙 6(수정 1차): "취소"는 NEGATIVES 에도 있지만 뜻이 질문에 따라 갈린다 — "화장실로 안내해드릴까요?"에는 거절,
            # "안내를 취소할까요?"에는 승낙. 규칙이 정하지 않는다 — 의도는 지금 길(취소 지름길·LLM)이 정한다.
            return None
        if word in NEGATIVES:
            return ("no", "")
        if word in AFFIRMATIVES:
            return ("yes", "")
        return None
    if question.kind == "minutes":
        from .langchain_intent_parser import parse_wait_minutes   # 지연 import — 해석기가 이 모듈을 import 한다

        minutes = parse_wait_minutes(text or "", range_factor=1.0 if local else 1.5)
        return ("minutes", str(minutes)) if minutes else None
    return None


def standing(ruled: Optional[tuple[str, str]], intent_name: str, llm_answer: str) -> bool:
    """규칙 대답(ruled)이 서는가(Task 3.5b 규칙 1, 원칙 6 — 바탕 설계 D11). 글자·소리 경로가 같이 쓴다.

    규칙은 LLM 의 읽기를 뒤집지 않는다(수정 2차, 검토 Critical — "가자"는 확인 질문에는 승낙이지만 "안내를 취소할까요?"·
    "기다릴까요?"에는 아니다). 네/아니요는 LLM 의 읽기와 같을 때만 선다 — LLM 대답(llm_answer)이 규칙 대답과 같거나, 빈 값이고
    의도(intent_name)가 그 대답의 의도(yes ↔ affirm, no ↔ deny)일 때. 분은 LLM 도 wait 로 읽었을 때만(수정 3차 — 값 바로잡기뿐,
    못 알아들은 말을 규칙이 동작으로 바꾸지 않는다) — LLM 대답이 빈 값·minutes·unclear 이고 의도가 wait 일 때.
    parse_wait_minutes 는 문장 속의 분도 읽는다("십 분 말고 그냥 끝내 줘"·"아까 한 시간 기다렸잖아", 검토 I3·재현).
    """
    if ruled is None:
        return False
    llm = (llm_answer or "").strip()
    if ruled[0] in _YES_NO_INTENT:
        return llm == ruled[0] or (llm == "" and intent_name == _YES_NO_INTENT[ruled[0]])
    return llm in _MINUTES_CORRECTS and intent_name in _MINUTES_INTENTS


def _ruled(intent: VicaIntent, question: OpenQuestion, ruled: tuple[str, str]) -> VicaIntent:
    """선 규칙 대답을 실은 사본(Task 3.5b 규칙 2). 대답 칸을 채우고, 분이면 wait_minutes 도 규칙의 분으로 맞춘다(수정 1차 —
    answer_value 와 갈리면 미션이 읽는 칸에 따라 대기 시간이 달라진다). 의도와 다른 칸은 LLM(·지름길)이 정한 그대로다(수정 2·3차)."""
    answer, value = ruled
    update = {"question_id": question.id, "answer": answer, "answer_value": value}
    if answer == "minutes":        # 의도는 wait 다(standing) — 값만 바로잡는다
        update.update(wait_minutes=int(value))
    return intent.model_copy(update=update)


def _not_dropped(intent: VicaIntent, question: OpenQuestion) -> VicaIntent:
    """원칙 6·7(수정 2·3차 규칙 3): 대답으로 보이는 말(네/아니요·분)이 not_for_robot 으로 왔다 — 버리지도(노드가 말없이 버린다)
    동작으로 살리지도(옆 사람의 "네"가 출발이, "오 분만 있다가 가자"가 대기가 된다) 않는다. unknown·대답 칸 unclear 로 — 로봇은
    움직이지 않고 미션이 열린 질문을 다시 묻는다."""
    return intent.model_copy(update={
        "intent": "unknown", "reply": "", "ask_text": "", "ask_kind": "", "need_confirm": False,
        "destination_candidate": None, "matched_destination_id": "",
        "question_id": question.id, "answer": "unclear", "answer_value": ""})


def fill(intent: VicaIntent, question: Optional[OpenQuestion], ruled: Optional[tuple[str, str]] = None,
         llm_answer: str = "") -> VicaIntent:
    """의도에 question_id·answer·answer_value 를 채운 사본. 꺼짐·질문 없음·긴급이면 받은 의도 그대로.

    규칙은 LLM 의 읽기를 뒤집지 않고, 못 알아들은 말을 동작으로 바꾸지 않는다(수정 2·3차). 하는 일은 셋뿐이다 — 같은 읽기일 때
    대답 칸 채우기(네/아니요), LLM 도 wait 로 읽은 분의 값 바로잡기(_ruled), 대답으로 보이는 말(네/아니요·분)을 not_for_robot
    으로 버리지 않고 되묻게 하기(_not_dropped). 규칙이 서지 않으면 LLM 이 낸 값이다. 허용 밖의 값, 목적지 id 없는
    destination·choice, 분 없는 minutes 는 unclear 로 바꾼다(설계 4.1 — 외부 API 는 형식을 강제하지 않는다).
    """
    if not on() or question is None or intent.safety_flag == "emergency":
        return intent
    if ruled is not None and intent.intent == _NOT_FOR_ROBOT:
        return _not_dropped(intent, question)
    if standing(ruled, intent.intent, llm_answer):
        return _ruled(intent, question, ruled)
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

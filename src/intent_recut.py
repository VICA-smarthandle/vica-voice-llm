"""의도 재단(2026-11-20 voice-principles-intent-recut) — 순수 함수만 둔다.

되묻는 문장을 LLM 이 짓지 않고 코드의 틀에서 만든다(원칙 4·11), [지금 상황]에 실을 단서 줄을 만든다(원칙 7·19).
전부 `.env` VICA_RECUT=on 일 때만 움직인다(기본 꺼짐 = 지금과 같다). ROS·LLM 을 모른다 — 장치 없이 시험된다.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

from . import answer_field
from .one_voice import _CHOICE_TAIL, _choice_option_ids
from .question_record import OpenQuestion
from .replies import (ASK_DESTINATION, LLM_UNAVAILABLE, RETRY_PROMPT, WAKE_GREETING, choose_between,
                      not_available)
from .schema import DestinationData, VicaIntent

NOT_FOR_ROBOT = "not_for_robot"
DISMISS = "dismiss"
CLARIFY_KINDS = ("several", "not_available", "no_destination")

# 지시문 두 벌(글자·소리)이 함께 싣는 글(원칙 1·4·7·12·31). 켰을 때만 지시문에 들어간다. 줄바꿈으로 시작·끝나지 않는다.
PROMPT_INTENTS = """- clarify: 로봇에게 한 말이고 가고 싶은 곳에 대한 말인데 한 곳으로 정할 수 없다. reply 는 빈 문자열로 두고(되묻는 문장은 시스템이 만든다) clarify_kind 에 까닭을 적는다.
  several = 목록의 여러 곳에 똑같이 걸린다. clarify_options 에 그 곳들의 name 을 모두 적는다(하나만 적지 않는다).
  not_available = 목록에 없는 곳을 찾는다. clarify_wanted 에 찾는 것을 사용자가 말한 낱말 그대로 적는다. 비슷한 다른 곳으로 바꾸지 않는다.
  no_destination = 어디로 갈지 말하지 않았다.
- unknown: 로봇에게 한 말인데 무슨 뜻인지 알아듣지 못했다. reply 는 빈 문자열 — 시스템이 다시 말해 달라고 한다.
- not_for_robot: 로봇에게 한 말이 아니다(옆 사람과 나눈 말, 로봇을 두고 설명하는 말, 혼잣말, 로봇 목소리의 메아리, 소음). reply 는 빈 문자열 — 로봇은 끼어들지 않는다.
- dismiss: 안내를 받고 있지 않은데 로봇과의 대화를 그만하려 한다("됐어", "아니야 괜찮아", "나중에 할게"). reply 는 빈 문자열.
  안내 중에 그만두려는 말은 cancel, 로봇이 방금 한 질문을 거절하는 말은 deny 다."""

PROMPT_ORDER = """[판단 순서] 들은 말을 이 순서로 판단하고, 앞에서 정해지면 뒤로 가지 않는다.
1. 로봇에게 한 말인가. 아니면 not_for_robot 이다. [지금 상황]의 '로봇이 방금 한 말' 줄을 함께 본다 — 로봇이 방금 부름에 대답했거나 질문했다면 이어진 말은 대개 로봇에게 한 말이다. 그래도 사람끼리 주고받는 말이거나 로봇을 가리켜 설명하는 말이면 not_for_robot 이다.
2. 로봇이 방금 한 질문의 대답인가. 대답이면 그 질문의 뜻으로 읽는다.
3. 새 요청인가. 무엇을 원하는지 읽는다. 가려는 곳이 목록에 없으면 clarify(not_available)다.
4. 로봇에게 한 말인데 알아듣지 못했으면 unknown 이다.
로봇에게 한 말인지 확실하지 않으면 not_for_robot 이 아니라 unknown 을 고른다 — 진짜 요청을 버리는 것이 되묻는 것보다 나쁘다."""

# 두 지시문이 같은 말로 쓰는 조각(켰을 때).
SEVERAL_RULE = "clarify(clarify_kind=several)로 내고 clarify_options 에 그 곳들의 name 을 모두 적는다. 하나만 적지 않는다(2026-10-11 실기)."
NOT_AVAILABLE_TAG = "(clarify_kind=not_available)"
NO_DESTINATION_TAG = "(clarify_kind=no_destination)"
WAKE_LINE = '호출어 "비카야"만 들리면 로봇을 부른 것이다 — intent=unknown, reply="네?".'

_HEADER = "\n[지금 상황]\n"


def on() -> bool:
    """스위치. `.env` VICA_RECUT=on 일 때만 참이다. 매번 읽는다 — 시험이 monkeypatch 로 켜고 끌 수 있게."""
    return os.environ.get("VICA_RECUT", "").strip().lower() == "on"


def clarify_reply(kind: str, wanted: str, names: Sequence[str]) -> str:
    """되묻기 종류별 문장(원칙 4). 이름이 하나뿐인 several·모르는 종류는 짐작하지 않고 목적지를 묻는다."""
    if kind == "several":
        names = [n for n in names if n]
        return choose_between(names) if len(names) >= 2 else ASK_DESTINATION
    if kind == "not_available":
        return not_available(wanted)
    return ASK_DESTINATION


def clue_line(question: Optional[OpenQuestion]) -> str:
    """[지금 상황]에 넣을 '로봇이 방금 한 말' 한 줄(원칙 7). 질문이 없으면 빈 글자. 스위치는 보지 않는다."""
    if question is None:
        return ""
    if question.kind == "open":
        note = '사용자가 "비카야"라고 불러서 대답했다'
    else:
        note = f"{answer_field._KIND_KO.get(question.kind, '질문')} — 대답을 기다리는 중"
    return f'- 로봇이 방금 한 말: "{question.text}" ({note})'


def with_clue(block: str, question: Optional[OpenQuestion]) -> str:
    """[지금 상황] 글 끝에 단서 줄을 붙인다. 꺼졌거나·대답 칸이 켜졌거나(그쪽 질문 줄이 같은 내용을 싣는다)·질문이 없으면 그대로."""
    if not on() or answer_field.on() or question is None:
        return block
    line = clue_line(question)
    if not block:
        return _HEADER + line + "\n"
    return block.rstrip("\n") + "\n" + line + "\n"


def speaker_line(verdict: str) -> str:
    """화자 확인 결과 한 줄(원칙 19). pass·fail 만 말하고, 확인하지 못한 값은 빈 글자. 스위치는 보지 않는다."""
    if verdict == "pass":
        return "- 이 말의 목소리: 안내를 요청한 사용자와 같다"
    if verdict == "fail":
        return "- 이 말의 목소리: 안내를 요청한 사용자와 다르다"
    return ""


def early(draft) -> Optional[VicaIntent]:
    """뒤처리 맨 앞(원칙 7·31): not_for_robot·dismiss 면 문장·목적지 없는 의도로 바로 돌려보낸다. 꺼졌거나 다른 값이면 None.

    긴급어는 이 함수보다 앞 단계(emergency_filter)가 처리한다 — 여기서 다루지 않는다.
    """
    if not on() or getattr(draft, "intent", "") not in (NOT_FOR_ROBOT, DISMISS):
        return None
    return VicaIntent(intent=draft.intent, reply="", need_confirm=False,
                      confidence=getattr(draft, "confidence", None) or 0.0)


def _is_wake(heard: str) -> bool:
    # 해석기가 이미 쓰는 정규화·호출어 집합 — 해석기가 이 모듈을 import 하므로 함수 안에서 늦게 가져온다(순환 방지).
    from . import langchain_intent_parser as parser
    return parser._normalize_short_reply(heard) in parser._WAKE_WORDS


def _all_approachable(head: str, destinations: Sequence[DestinationData]) -> bool:
    """고르기 문장 앞부분에 이름이 든 목적지가 둘 이상이고 전부 갈 수 있는 곳이다(원칙 18)."""
    ids = _choice_option_ids(head, destinations)
    by_id = {d.id: d for d in destinations}
    return len(ids) >= 2 and all(by_id[i].is_approachable for i in ids)


def settle(result: VicaIntent, draft, destinations: Sequence[DestinationData], heard: str) -> VicaIntent:
    """뒤처리 끝(원칙 4·8·11·18): clarify 의 문장을 코드 틀에서 만들고 unknown 의 문장을 정해진 되묻는 말로 둔다.

    꺼졌거나 긴급이면 받은 결과 그대로. 호출어만 들렸거나 reply 가 "네?"인 unknown 은 "네?"로 둔다.
    """
    if not on() or result.safety_flag == "emergency":
        return result
    if result.intent == "clarify":
        reply = (result.reply or "").strip()
        tail = _CHOICE_TAIL.search(reply)
        if tail and reply != (getattr(draft, "reply", "") or "").strip() and _all_approachable(reply[:tail.start()], destinations):
            return result                      # 코드가 이미 만든 고르기 문장(_ask_which 등) — 모델이 지은 문장과 다를 때만
        ok = {d.name: d for d in destinations if d.name and d.is_approachable}
        options = list(dict.fromkeys(n for n in (getattr(draft, "clarify_options", None) or []) if n in ok))
        wanted = getattr(draft, "clarify_wanted", "") or ""
        if len(options) >= 2:
            kind = "several"
        elif getattr(draft, "clarify_kind", "") == "not_available" or draft.intent == "navigate":
            kind = "not_available"             # 목록에 없는 곳(navigate 가 코드에서 clarify 로 낮춰진 경우 포함)
            wanted = wanted or (draft.destination_candidate or "")
        else:
            kind = "no_destination"
        return result.model_copy(update={"reply": clarify_reply(kind, wanted, options)})
    if result.intent == "unknown":
        if result.reply == LLM_UNAVAILABLE:
            return result
        wake = (result.reply or "").strip() == WAKE_GREETING or _is_wake(heard or "")
        return result.model_copy(update={"reply": WAKE_GREETING if wake else RETRY_PROMPT})
    return result

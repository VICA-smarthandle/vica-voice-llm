"""의도 재단(2026-11-20 voice-principles-intent-recut) — 순수 함수만 둔다.

되묻는 문장을 LLM 이 짓지 않고 코드의 틀에서 만든다(원칙 4·11), [지금 상황]에 실을 단서 줄을 만든다(원칙 7·19).
전부 `.env` VICA_RECUT=on 일 때만 움직인다(기본 꺼짐 = 지금과 같다). ROS·LLM 을 모른다 — 장치 없이 시험된다.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

from . import answer_field
from .question_record import OpenQuestion
from .replies import ASK_DESTINATION, choose_between, not_available

NOT_FOR_ROBOT = "not_for_robot"
DISMISS = "dismiss"
CLARIFY_KINDS = ("several", "not_available", "no_destination")

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

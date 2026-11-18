"""LLM 도구 정의(설계 2026-11-18 5.1).

도구는 외부 API 두 경로(gpt-5.4-mini 글자, Realtime 소리)에만 준다. 로컬 모델에는 주지 않는다(D11).
LLM 은 발화마다 도구 하나를 부른다: set_intent(지금의 의도 해석) 또는 reserve_facility(예약 요청과 그 대답).
저장하는 도구는 없다 — 저장은 사용자의 승낙 뒤에 코드가 한다(reservation_flow).

언제 무엇을 부르는지는 지시문이 아니라 아래 도구 설명에 적는다. 지시문은 로컬 폴백도 같이 쓴다.
이 모듈의 최상위 import 는 가볍게 둔다 — TTS 노드도 스위치를 보려고 import 한다.
"""
from __future__ import annotations

import os
from typing import Sequence

SET_INTENT = "set_intent"
RESERVE = "reserve_facility"

RESERVE_DESCRIPTION = (
    "시설을 예약해 달라는 말, 그 시간에 쓸 수 있는지 묻는 말, 진행 중인 예약에 대한 대답(이름·날짜·시각·시설)에 호출한다. "
    "사용자가 이번에 말한 값만 넣는다. 어디로 가자, 안내해 달라는 말은 이동이므로 set_intent 를 호출한다. "
    "실제 예약은 사용자 확인 뒤 시스템이 한다.")
SET_INTENT_DESCRIPTION = "시설 예약이 아닌 모든 말의 의도를 정한다."
_RESERVE_KEYS = ("facility", "date", "start_time", "end_time", "name")


def _on(key: str) -> bool:
    return os.environ.get(key, "").strip().lower() == "on"


def text_tools_on() -> bool:
    """글자 경로(gpt-5.4-mini)의 도구 출력. `.env` VICA_TOOLS_TEXT=on, 기본 꺼짐."""
    return _on("VICA_TOOLS_TEXT")


def audio_tools_on() -> bool:
    """소리 경로(Realtime)의 도구 출력. `.env` VICA_TOOLS_AUDIO=on, 기본 꺼짐."""
    return _on("VICA_TOOLS_AUDIO")


def reserve_tool(facility_names: Sequence[str], with_heard_text: bool = False) -> dict:
    """reserve_facility 도구(OpenAI 함수 꼴). Realtime 은 with_heard_text 로 들린 말을 함께 받는다."""
    properties = {
        "facility": {"type": "string", "enum": list(facility_names), "description": "예약할 시설"},
        "date": {"type": "string",
                 "description": "사용자가 말한 날짜 표현 그대로. 예: 오늘, 내일, 모레, 다음 주 월요일, 11월 20일"},
        "start_time": {"type": "string", "description": "시작 시각, 24시간 HH:MM"},
        "end_time": {"type": "string", "description": "끝 시각이나 쓰는 시간을 말했을 때만, 끝 시각을 24시간 HH:MM 으로"},
        "name": {"type": "string", "description": "예약자 이름. 사용자가 말했을 때만"},
    }
    required: list[str] = []
    if with_heard_text:
        properties["heard_text"] = {"type": "string", "description": "들린 말을 한국어로 그대로 적는다"}
        required = ["heard_text"]
    return {"type": "function", "function": {
        "name": RESERVE, "description": RESERVE_DESCRIPTION,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


def set_intent_tool(parameters: dict) -> dict:
    """set_intent 도구. parameters 는 지금 구조화 출력(_IntentDraft)의 JSON 스키마."""
    return {"type": "function", "function": {
        "name": SET_INTENT, "description": SET_INTENT_DESCRIPTION, "parameters": parameters}}


def reserve_call(args: dict):
    """reserve_facility 인자 → ReserveCall. 모르는 키는 버리고 글자가 아닌 값은 빈 값으로."""
    from .reservation_flow import ReserveCall   # 지연 import — 이 모듈을 가볍게 둔다

    return ReserveCall(**{k: (args.get(k).strip() if isinstance(args.get(k), str) else "") for k in _RESERVE_KEYS})

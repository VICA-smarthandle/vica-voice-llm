"""긴급 명령어를 LLM 호출 '이전'에 규칙 기반으로 감지한다."""
from __future__ import annotations

import re
from typing import Optional

EMERGENCY_KEYWORDS = [
    "멈춰",
    "정지",
    "스탑",
    "스톱",
    "안돼",
    "위험해",
]

SOFT_KEYWORDS = [
    "잠깐",
    "천천히",
    "느리게",
]

from .replies import EMERGENCY_REPLY  # noqa: E402  (문서 흐름상 여기에 둔다)

__all__ = [
    "EMERGENCY_KEYWORDS",
    "SOFT_KEYWORDS",
    "EMERGENCY_REPLY",
    "PAUSE_PHRASES",
    "PAUSABLE_DIALOG_STATES",
    "detect_emergency",
    "detect_stop_request",
    "stop_pause_intent_for",
]

_TOKEN_SPLIT = re.compile(r"[\s,.!?~…·\"'()\[\]{}<>:;]+")


def _starts_at_token_boundary(text: str, keyword: str) -> bool:
    """긴급어가 어절 경계에서 시작하는가."""
    tokens = [token for token in _TOKEN_SPLIT.split(text) if token]
    for start in range(len(tokens)):
        joined = ""
        for token in tokens[start:]:
            joined += token
            if len(joined) >= len(keyword):
                break
        if joined.startswith(keyword):
            return True
    return False


def detect_emergency(text: str) -> Optional[str]:
    """발화에 긴급어가 있으면 매칭된 키워드를, 없으면 None 을 돌려준다."""
    if not text:
        return None
    for keyword in EMERGENCY_KEYWORDS:
        if _starts_at_token_boundary(text, keyword):
            return keyword
    return None


PAUSE_PHRASES = ("일시정지",)

PAUSABLE_DIALOG_STATES = ("navigating", "paused_handle")


def detect_stop_request(text: str) -> Optional[str]:
    """멈춤 말이면 그 말을, 아니면 None. 하드 긴급어 + 붙여 쓴 '일시정지'."""
    keyword = detect_emergency(text)
    if keyword:
        return keyword
    joined = re.sub(r"\s+", "", text or "")
    for phrase in PAUSE_PHRASES:
        if phrase in joined:
            return phrase
    return None


def stop_pause_intent_for(text: str, dialog_state: str):
    """청취 창 안 말이 멈춤 요청이고 지금 안내 주행 중이면 일시정지 제안을, 아니면 None."""
    if dialog_state not in PAUSABLE_DIALOG_STATES or not detect_stop_request(text):
        return None
    from .schema import VicaIntent

    return VicaIntent(intent="pause", reply="", need_confirm=False, safety_flag="emergency")

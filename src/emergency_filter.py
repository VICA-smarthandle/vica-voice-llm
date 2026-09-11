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
    "detect_emergency",
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

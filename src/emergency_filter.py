"""긴급 명령어를 LLM 호출 '이전'에 규칙 기반으로 감지한다.

안전 원칙 (CLAUDE.md): 긴급 정지는 LLM 을 거치지 않는다.
이 모듈은 '감지'만 한다. 실제 정지는 Safety Supervisor / State Machine 이 한다.
"""
from __future__ import annotations

import re
from typing import Optional

# CLAUDE.md 의 긴급어 후보를 그대로 채택.
EMERGENCY_KEYWORDS = [
    "멈춰",
    "정지",
    "스탑",
    "스톱",
    "안돼",
    "위험해",
    "잠깐",
    "천천히",
    "느리게",
]


def detect_emergency(text: str) -> Optional[str]:
    """발화에 긴급어가 있으면 매칭된 키워드를, 없으면 None 을 돌려준다."""
    if not text:
        return None
    norm = re.sub(r"\s+", "", text)
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in norm:
            return keyword
    return None

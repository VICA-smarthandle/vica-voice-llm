"""긴급 명령어를 LLM 호출 '이전'에 규칙 기반으로 감지한다.

안전 원칙 (CLAUDE.md): 긴급 정지는 LLM 을 거치지 않는다.
이 모듈은 '감지'만 한다. 실제 정지는 Safety Supervisor / State Machine 이 한다.

목록 정본은 vica_ros2_ws 의 mission_logic.HARD_EMERGENCY_KEYWORDS 다.
emergency_estop_bridge 가 그 목록으로 최종 판정하므로, 여기 목록이 더 넓으면
"멈췄다고 말했는데 실제로는 안 멈추는" 어긋남이 생긴다. 두 목록을 일치시킨다.
"""
from __future__ import annotations

import re
from typing import Optional

# 즉시 정지로 이어지는 하드 긴급어. mission_logic.HARD_EMERGENCY_KEYWORDS 와 동일.
EMERGENCY_KEYWORDS = [
    "멈춰",
    "정지",
    "스탑",
    "스톱",
    "안돼",
    "위험해",
]

# 정지가 아니라 '속도를 줄여 달라'는 요청. 예전에는 긴급어로 묶여 있었으나,
# 이 말들은 E-stop 대상이 아니어서 로봇이 "멈추겠습니다"라고 답하고도 계속 가는
# 어긋남을 만들었다. 지금은 LLM 이 일반 발화로 해석한다.
# 감속 intent 로 연결하는 작업은 별도 설계 항목이다.
SOFT_KEYWORDS = [
    "잠깐",
    "천천히",
    "느리게",
]

# 긴급어 감지 시의 응답. 문구 정본은 replies.py 에 있고, 기존 import 경로를
# 유지하려고 여기서 다시 내보낸다.
from .replies import EMERGENCY_REPLY  # noqa: E402  (문서 흐름상 여기에 둔다)

__all__ = [
    "EMERGENCY_KEYWORDS",
    "SOFT_KEYWORDS",
    "EMERGENCY_REPLY",
    "detect_emergency",
]

# 어절 구분자 (공백과 문장부호).
_TOKEN_SPLIT = re.compile(r"[\s,.!?~…·\"'()\[\]{}<>:;]+")


def _is_hangul(char: str) -> bool:
    return "가" <= char <= "힣"


def _starts_token(text: str, keyword: str) -> bool:
    """어절 하나가 긴급어로 시작하는가. 예) "멈춰줘", "정지해 주세요"."""
    return any(token.startswith(keyword) for token in _TOKEN_SPLIT.split(text) if token)


def _starts_word(text: str, keyword: str) -> bool:
    """공백을 지운 문자열에서 긴급어가 낱말 첫머리에 오는가.

    앞 글자가 한글이면 더 긴 낱말의 일부로 본다. "행정지원실"의 "정지",
    "감정지수"의 "정지" 같은 오탐을 막는다. 반대로 "안 돼요"처럼 띄어 쓴 경우는
    공백을 지운 뒤 첫머리에 오므로 잡힌다.
    """
    norm = re.sub(r"\s+", "", text)
    start = 0
    while True:
        index = norm.find(keyword, start)
        if index < 0:
            return False
        if index == 0 or not _is_hangul(norm[index - 1]):
            return True
        start = index + 1


def detect_emergency(text: str) -> Optional[str]:
    """발화에 긴급어가 있으면 매칭된 키워드를, 없으면 None 을 돌려준다.

    낱말 첫머리에서만 인정한다. 단순 부분 문자열로 보면 "행정지원실"(행정+지원실)
    같은 일반 낱말이 "정지"로 잡혀 엉뚱한 비상정지가 걸린다.
    """
    if not text:
        return None
    for keyword in EMERGENCY_KEYWORDS:
        if _starts_token(text, keyword) or _starts_word(text, keyword):
            return keyword
    return None

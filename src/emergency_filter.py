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


def _starts_at_token_boundary(text: str, keyword: str) -> bool:
    """긴급어가 어절 경계에서 시작하는가.

    어절(공백·문장부호로 나뉜 조각)을 start 번째부터 이어 붙여, 그 결과가 긴급어로
    시작하는지 본다. 두 가지를 동시에 만족해야 하기 때문이다.

    1. STT 는 띄어쓰기를 제멋대로 낸다. 특히 "안돼"를 거의 항상 "안 돼"로 적으므로
       어절 하나만 봐서는 못 잡는다. 그래서 뒤 어절까지 이어 붙인다.
    2. 그렇다고 공백을 통째로 지우고 아무 위치나 인정하면 "행정지원실"의 "정지"가
       잡혀 엉뚱한 비상정지가 걸린다. 그래서 '어절이 시작하는 자리'에서만 인정한다.

    실측 근거: 예전 규칙(공백 제거 후 앞 글자가 한글이면 무시)은 whisper 가 "아 안 돼"
    라고 정확히 받아쓴 5회를 전부 걸러냈다. 앞의 "아" 때문이다. 위급할 때 감탄사를
    붙여 외치는 것은 자연스러우므로 놓치면 안 된다.
    (docs/measurements/emergency-20260725-1600.md)
    """
    tokens = [token for token in _TOKEN_SPLIT.split(text) if token]
    for start in range(len(tokens)):
        joined = ""
        for token in tokens[start:]:
            joined += token
            if len(joined) >= len(keyword):
                break  # 긴급어 길이만큼만 모으면 판정에 충분하다
        if joined.startswith(keyword):
            return True
    return False


def detect_emergency(text: str) -> Optional[str]:
    """발화에 긴급어가 있으면 매칭된 키워드를, 없으면 None 을 돌려준다.

    어절 첫머리에서만 인정한다. 단순 부분 문자열로 보면 "행정지원실"(행정+지원실)
    같은 일반 낱말이 "정지"로 잡혀 엉뚱한 비상정지가 걸린다.
    """
    if not text:
        return None
    for keyword in EMERGENCY_KEYWORDS:
        if _starts_at_token_boundary(text, keyword):
            return keyword
    return None

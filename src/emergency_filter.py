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
    "PAUSE_PHRASES",
    "PAUSABLE_DIALOG_STATES",
    "detect_emergency",
    "detect_stop_request",
    "stop_pause_intent_for",
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


# ---- 청취 창 안 멈춤 말 → 주행 중이면 일시정지 (2026-10-06 사용자 결정 (가)) ----
# '비카야' 뒤 말에 멈춤 말이 있으면 안내 주행 중에는 일시정지(pause) 제안으로 바꾼다.
# 비상정지(래치·관리자 해제)는 '비카야' 없이 외친 긴급어(웨이크워드 노드 긴급 모델)
# 몫으로 남긴다 — 불러 놓고 한 요청은 말로 다시 출발할 수 있는 정지가 맞다.
# 전에는 이 말이 safety_flag 만 단 unknown 으로 나가 미션이 아무 데도 쓰지 않았다
# (10-05 21:40 실기: '이동을 멈춰'에도 계속 주행).

# 어절 첫머리 규칙(detect_emergency)으로는 붙여 쓴 '정지'를 못 잡는 멈춤 말.
PAUSE_PHRASES = ("일시정지",)

# 일시정지가 받아들여지는 미션 대화 단계(RobotState.dialog_state). 미션의
# check_pause_gate 와 같은 뜻이다 — 주행 중, 그리고 손 놓침으로 선 PAUSED(말로 재개하는
# 보통 일시정지로 바뀐다). 그 밖에서는 미션이 '지금은 안내 중이 아닙니다'로 거절하므로
# 바꾸지 않고 예전처럼 둔다.
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
    """청취 창 안 말이 멈춤 요청이고 지금 안내 주행 중이면 일시정지 제안을, 아니면 None.

    reply 는 비운다 — "잠시 멈추겠습니다" 는 일시정지를 실제로 건 미션이 말한다.
    safety_flag 는 emergency 로 둬 멈춤 말에서 왔다는 흔적을 남긴다(미션의 pause
    경로는 이 값을 보지 않는다).
    """
    if dialog_state not in PAUSABLE_DIALOG_STATES or not detect_stop_request(text):
        return None
    from .schema import VicaIntent

    return VicaIntent(intent="pause", reply="", need_confirm=False, safety_flag="emergency")

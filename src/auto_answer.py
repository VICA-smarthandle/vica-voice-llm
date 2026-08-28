"""접근 질문 뒤 '엔터 없는' 답변 창 판정 — 순수 로직.

로봇이 다가가 묻는 상대는 터미널 앞에 있지 않은 시각장애인이다. push-to-talk 의
엔터를 눌러 줄 사람이 없으므로, 접근 질문의 재생이 끝난 직후에만 마이크를
자동으로 연다. 상시 청취(귀 계층 재설계)가 오기 전까지의 다리다.

ROS 도 마이크도 모르는 순수 함수만 둔다 — pytest 만으로 끝난다.
"""
from __future__ import annotations

# mission_logic.py 의 MSG_APPROACH_QUESTION ("안내가 필요하신가요?") 과 맞춘다.
# 문구가 바뀌면 여기도 같이 바꾼다 — /vica/tts_done 은 재생한 문장을 그대로 싣는다.
# 다른 멘트(수락 "도와드리겠습니다"·무응답 "실례했습니다" 등)에는 이 조각이 없다.
AUTO_ANSWER_HINTS = ("안내가 필요",)

# 자동 녹음 길이[초]. mission 의 응답 대기는 재생 종료부터 8초다
# (APPROACH_RESPONSE_TIMEOUT_SEC). 녹음 4초 + whisper 약 2초 = 6초 < 8초.
RECORD_SECONDS = 4.0


def should_auto_listen(spoken_text: str) -> bool:
    """방금 끝까지 재생된 문장이 답변을 기다리는 접근 질문인가."""
    return any(hint in spoken_text for hint in AUTO_ANSWER_HINTS)

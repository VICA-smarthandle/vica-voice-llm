"""장면 분석기가 예상 로봇 문장을 만들 때 쓰는 미션 고정 문장 사본 (R2: src/mission_phrases.py 는 건드리지 않는다).

정본은 vica_ros2_ws 의 mission_logic.py MSG_* 이고, tests/test_scene_phrases.py 가 그 파일을 읽어 글자를 대조한다.
src/mission_phrases.py 에 이미 있는 문장은 거기서 가져온다.
"""
from __future__ import annotations

from src.mission_phrases import ASK_ENTRANCE, ASK_WAIT_TIME, DOOR_SIDE, HELP_CANCEL_CONFIRM, HELP_CONFIRM, WAIT_NEED_ASK  # noqa: F401
from src.replies import RESUME_CONFIRM  # noqa: F401  (LLM 쪽 문장 — 미션 상수 아님)

START = "{name}{josa} 안내를 시작합니다."            # MSG_START
CANCEL_CONFIRM = "안내를 취소할까요?"                 # MSG_CANCEL_CONFIRM
CANCELED = "안내를 취소했습니다."                     # MSG_CANCELED
CANCEL_KEPT = "안내를 계속하겠습니다."                # MSG_CANCEL_KEPT
PAUSED = "잠시 멈추겠습니다. 다시 출발하려면 말씀해 주세요."  # MSG_PAUSED
NOT_NAVIGATING = "지금은 안내 중이 아닙니다."          # MSG_NOT_NAVIGATING
FINISH = "안내를 종료합니다."                         # MSG_FINISH

# 미션 파일에 글자 그대로 있어야 하는 것들 (RESUME_CONFIRM 은 미션 상수가 아니라 제외)
MISSION_COPIES = (ASK_ENTRANCE, START, CANCEL_CONFIRM, CANCELED, CANCEL_KEPT, PAUSED, NOT_NAVIGATING, FINISH)

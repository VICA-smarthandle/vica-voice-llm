"""청각 안내의 판정 로직 (소리·ROS 없이 검증되는 순수 로직).

두 가지를 정한다.

1. 회전 신호를 받았을 때 무엇을 말할지 (TurnAnnouncer)

소리를 내는 것은 audio_cue.py, 토픽을 다루는 것은 ros_audio_cue_node.py 다.
여기는 "언제 무엇을" 만 정한다.
"""
from __future__ import annotations

import json
from typing import Optional

from .replies import TURN_LEFT, TURN_RIGHT

# TurnGuide.msg 의 상수와 같은 값. 메시지를 import 하면 ROS 없이 시험할 수 없어
# 숫자를 그대로 둔다 (정본은 vica_ros2_ws/src/vica_interfaces/msg/TurnGuide.msg).
DIRECTION_NONE = 0
DIRECTION_LEFT = 1
DIRECTION_RIGHT = 2

PHASE_IDLE = 0
PHASE_PREPARE = 1
PHASE_NOW = 2
PHASE_COMPLETE = 3
PHASE_CANCELED = 4

# 안내가 끝났다고 보는 /vica_goal_event 이벤트 이름.
# 정본은 vica_ros2_ws 의 mission_manager_node._publish_goal_event 다.
GUIDANCE_END_EVENTS = frozenset({"goal_succeeded", "goal_canceled", "goal_failed"})
GUIDANCE_ARRIVED_EVENT = "goal_succeeded"

def parse_goal_event(payload: str) -> Optional[str]:
    """/vica_goal_event JSON 에서 event 문자열만 꺼낸다.

    이 토픽의 계약은 JSON 이다 (guideline/vica_architecture.md 토픽 표,
    vica_user_guidance.guidance_priority.parse_goal_event 와 같은 판정).
    평문 이벤트 이름은 계약이 아니므로 받지 않는다 — 두 형식을 다 허용하면
    계약이 다시 흐려진다. 파싱 실패나 event 키 부재는 None 이다. 예외를
    던지지 않는다 — 잘못된 payload 하나가 안내 노드를 죽이면 안 된다.
    """
    try:
        data = json.loads(payload)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    event = data.get("event")
    return event if isinstance(event, str) else None


class TurnAnnouncer:
    """회전 신호를 발화 문구로 바꾼다. 같은 회전은 한 번만 말한다.

    안전 직결 안내라 축약하지 않는다 — 매번 같은 문장을 말해야 사용자가 매번
    같은 판단을 할 수 있다 (2026-08-05 사용자 결정).
    """

    def __init__(self) -> None:
        self._last_sequence: Optional[int] = None

    def on_turn(
        self,
        direction: int,
        phase: int,
        sequence_id: int,
        source_stale: bool = False,
    ) -> Optional[str]:
        """말할 문구를 돌려준다. 말하지 않을 상황이면 None.

        sequence_id 는 회전 진입마다 증가한다(TurnGuide.msg). 같은 회전 안에서
        신호가 여러 번 와도 첫 번째만 말한다.
        """
        if source_stale:
            # /odom 미수신·stale. direction 이 판단 불가라는 뜻이므로 말하지 않는다.
            return None
        if phase != PHASE_NOW:
            # PREPARE 는 2단계(경로 예고) 몫이고 1단계는 발행하지 않는다.
            # COMPLETE/CANCELED 는 끝났다는 뜻이라 안내할 것이 없다.
            return None
        if sequence_id == self._last_sequence:
            return None

        if direction == DIRECTION_LEFT:
            text = TURN_LEFT
        elif direction == DIRECTION_RIGHT:
            text = TURN_RIGHT
        else:
            return None

        self._last_sequence = sequence_id
        return text

    def reset(self) -> None:
        """안내가 끝나면 호출한다. 다음 안내의 첫 회전을 놓치지 않기 위해서다."""
        self._last_sequence = None

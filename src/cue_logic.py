"""청각 안내의 판정 로직 (소리·ROS 없이 검증되는 순수 로직).

두 가지를 정한다.

1. 회전 신호를 받았을 때 무엇을 말할지 (TurnAnnouncer)
2. 호출에 "네?" 로 답할지, 짧은 음으로 답할지 (GreetingState)

소리를 내는 것은 audio_cue.py, 토픽을 다루는 것은 ros_audio_cue_node.py 다.
여기는 "언제 무엇을" 만 정한다.
"""
from __future__ import annotations

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

# 대화가 끊겼다고 보는 시간. src/history.py 의 DEFAULT_IDLE_RESET_SEC 와 같은 값이며,
# 같은 뜻으로 쓴다 — "이전 사용자는 떠났다". 인사 주기를 따로 정한 것이 아니다.
DEFAULT_IDLE_RESET_SEC = 180.0


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


class GreetingState:
    """호출에 "네?" 로 답할지 정한다 — 안내 한 건마다 첫 호출에만.

    공용 로봇이라 사용자가 계속 바뀐다. 짧은 음만으로는 처음 쓰는 사용자가 무슨
    뜻인지 모르므로, 안내 한 건의 첫 호출에는 말로 답한다 (2026-08-05 결정).

    "부르고 아무 말도 하지 않은 것"은 인사로 치지 않는다. 잘못 부르고 떠난
    사람 뒤에 온 다음 사용자가 인사를 못 받는 것을 막는다.
    """

    def __init__(self, idle_reset_sec: float = DEFAULT_IDLE_RESET_SEC) -> None:
        self.idle_reset_sec = idle_reset_sec
        self._greeted = False
        self._last_activity: Optional[float] = None

    def on_wake(self, now: float) -> bool:
        """호출을 받았다. True 면 "네?", False 면 짧은 음."""
        if (
            self._last_activity is not None
            and now - self._last_activity >= self.idle_reset_sec
        ):
            # 대화가 끊겼다 = 이전 사용자는 떠났다. history.py 와 같은 판정이다.
            self._greeted = False
        self._last_activity = now
        return not self._greeted

    def on_user_spoke(self, now: float) -> None:
        """호출 뒤 실제로 발화가 이어졌다. 이때 비로소 인사가 성립한다."""
        self._last_activity = now
        self._greeted = True

    def on_guidance_ended(self) -> None:
        """도착·취소·실패. 다음 사용자를 위해 인사를 되살린다."""
        self._greeted = False

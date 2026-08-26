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
    """회전 신호를 발화 문구로 바꾼다 — 신중 모드 (2026-08-26 실기 재설계).

    실주행에서 잔 보정 흔들림마다 "우회전할게요"가 나와 부정확한 정보가
    쌓였다. 두 게이트를 거친 회전만 말한다:

    1. 지속 확인: NOW 신호 후 hold_sec 동안 COMPLETE 가 오지 않아야 발화.
       잔 보정은 금방 끝나므로 여기서 걸러진다.
    2. 도착 근접 억제: 잔여거리(신선한 값)가 near_goal_m 이하면 침묵 —
       도착 정렬 회전이 도착 멘트를 밀어내지 않게 한다.

    문구를 축약하지 않는 원칙(2026-08-05)은 유지한다. 안전 직결 안내라
    말할 때는 매번 같은 문장이다.
    """

    def __init__(
        self,
        hold_sec: float = 0.8,
        near_goal_m: float = 5.0,
        distance_fresh_sec: float = 3.0,
    ) -> None:
        self.hold_sec = hold_sec
        self.near_goal_m = near_goal_m
        self.distance_fresh_sec = distance_fresh_sec
        self._announced: set = set()          # 소진된 회차 (발화·억제·조기종료)
        self._pending: Optional[tuple] = None  # (sequence_id, text, armed_at)
        self._distance: Optional[float] = None
        self._distance_at: Optional[float] = None

    def set_distance(self, meters: Optional[float], now: float) -> None:
        """Nav2 잔여거리 갱신 (노드가 action feedback 토픽에서 먹인다)."""
        if meters is None or meters <= 0.0:
            return
        self._distance = float(meters)
        self._distance_at = now

    def _near_goal(self, now: float) -> bool:
        if self._distance is None or self._distance_at is None:
            return False
        if now - self._distance_at > self.distance_fresh_sec:
            return False  # 낡은 값으로는 침묵하지 않는다 — 모르는 것과 같다
        return self._distance <= self.near_goal_m

    def on_turn(
        self,
        direction: int,
        phase: int,
        sequence_id: int,
        now: float,
        source_stale: bool = False,
    ) -> None:
        """회전 신호를 먹인다. 발화 여부는 poll() 이 정한다."""
        if source_stale:
            # /odom 미수신 — 방향 판단 불가. 회차는 소진하지 않는다.
            return
        if phase in (PHASE_COMPLETE, PHASE_CANCELED):
            # hold 안에 끝난 회전 = 잔 보정. 침묵하고 회차를 소진한다 —
            # 같은 회차의 늦은 NOW 신호가 되살아나면 안 된다.
            if self._pending is not None and self._pending[0] == sequence_id:
                self._pending = None
                self._announced.add(sequence_id)
            return
        if phase != PHASE_NOW:
            return  # PREPARE 는 2단계(경로 예고) 몫이다
        if sequence_id in self._announced:
            return
        if self._pending is not None and self._pending[0] == sequence_id:
            return  # 이미 대기 중 — armed_at 을 되돌리지 않는다

        if direction == DIRECTION_LEFT:
            text = TURN_LEFT
        elif direction == DIRECTION_RIGHT:
            text = TURN_RIGHT
        else:
            return
        self._pending = (sequence_id, text, now)

    def poll(self, now: float) -> Optional[str]:
        """말할 때가 됐으면 문구를 돌려준다 (회차당 최대 한 번)."""
        if self._pending is None:
            return None
        sequence_id, text, armed_at = self._pending
        if now - armed_at < self.hold_sec:
            return None
        self._pending = None
        self._announced.add(sequence_id)
        if self._near_goal(now):
            return None  # 도착 근접 — 이 회전은 말하지 않고 소진한다
        return text

    def reset(self) -> None:
        """안내가 끝나면 호출한다. 다음 안내의 첫 회전을 놓치지 않기 위해서다."""
        self._announced = set()
        self._pending = None

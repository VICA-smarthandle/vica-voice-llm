"""[SIM ONLY] 가상 로봇 주행 시뮬레이터 — 순수 로직.

로봇 없이 음성/LLM 파이프라인을 "주행하는 것처럼" 개발·계측하기 위한 모형이다.
실제 로봇의 안전 의미론을 그대로 흉내 낸다:

  - 이동 확정 조건: intent==navigate + 목적지 확정 + 확인 완료 + safety normal
    (docs/ros2-interface.md 의 state machine 조건과 동일)
  - 긴급(E-stop): 즉시 정지 + **래치**. 새 이동은 거부되고, 자동 재개는 없다.
    reset 은 명시적으로만 (실제 규칙: 관리자 앱 단일 reset — GOVERNANCE 5절)
  - E-stop 해제 뒤에도 이전 목적지는 폐기된다 (자동 재개 금지 규칙)

주행 시간은 목적지 pose 거리 / 보행 속도로 계산한다. pose 가 자리표시자(원점)면
기본 시간을 쓰고, 층이 다르면 승강기 시간을 더한다.

시각(now)을 인자로 받는 순수 로직이라 ROS·시계 없이 단위 테스트할 수 있다.
"""
from __future__ import annotations

import math
from typing import Optional

SPEED_MPS = 0.8            # 안내 보행 속도
MIN_TRAVEL_SEC = 3.0
DEFAULT_TRAVEL_SEC = 8.0   # pose 가 자리표시자(≈원점)일 때
FLOOR_CHANGE_SEC = 5.0     # 층 이동(승강기) 가산


class SimRobot:
    """상태: idle / moving / estopped(래치)."""

    def __init__(self, destinations, start_building: str = "starlight_building",
                 start_floor: int = 1):
        self._dests = {d.id: d for d in destinations}
        self.state = "idle"
        self.building = start_building
        self.floor = start_floor
        self.x = 0.0
        self.y = 0.0
        self._goal = None          # 이동 중인 목적지
        self._arrive_at = 0.0

    # ---------------------------------------------------------------- 입력
    def handle_intent(self, intent, now: float) -> Optional[str]:
        """VicaIntent(pydantic 또는 유사 객체)를 받는다. 일어난 일을 돌려준다."""
        if self.state == "estopped":
            return "blocked_estop"     # 래치 중 — 새 이동 거부
        if (intent.intent != "navigate" or not intent.matched_destination_id
                or intent.need_confirm or intent.safety_flag != "normal"):
            return None
        dest = self._dests.get(intent.matched_destination_id)
        if dest is None:
            return None
        self._goal = dest
        self._arrive_at = now + self._travel_sec(dest)
        self.state = "moving"
        return "move_started"

    def handle_emergency(self, now: float) -> str:
        """긴급 수신 — 상태와 무관하게 즉시 래치. 목적지는 폐기된다."""
        self.state = "estopped"
        self._goal = None
        return "estopped"

    def reset(self, now: float) -> str:
        """명시적 reset (실제로는 관리자 앱만 가능). 목적지 자동 재개 없음."""
        self.state = "idle"
        self._goal = None
        return "reset"

    # ---------------------------------------------------------------- 진행
    def tick(self, now: float) -> Optional[dict]:
        """주기 호출. 도착했으면 도착 이벤트를 돌려준다."""
        if self.state != "moving" or now < self._arrive_at:
            return None
        dest = self._goal
        self.state = "idle"
        self._goal = None
        self.building = dest.building
        self.floor = dest.floor
        if dest.pose is not None:
            self.x, self.y = dest.pose.x, dest.pose.y
        return {
            "kind": "arrived",
            "dest_id": dest.id,
            "message": dest.arrival_message or f"{dest.name}에 도착했습니다.",
        }

    # ---------------------------------------------------------------- 상태
    @property
    def is_moving(self) -> bool:
        return self.state == "moving"

    def _travel_sec(self, dest) -> float:
        pose = getattr(dest, "pose", None)
        if pose is None:
            base = DEFAULT_TRAVEL_SEC
        else:
            dist = math.hypot(pose.x - self.x, pose.y - self.y)
            # 자리표시자 pose(제자리)면 거리 기반이 무의미 — 기본 시간 사용
            base = DEFAULT_TRAVEL_SEC if dist < 0.5 else max(MIN_TRAVEL_SEC,
                                                             dist / SPEED_MPS)
        if dest.floor != self.floor:
            base += FLOOR_CHANGE_SEC
        return base

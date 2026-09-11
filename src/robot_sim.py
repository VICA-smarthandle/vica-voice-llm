"""[SIM ONLY] 가상 로봇 주행 시뮬레이터 — 순수 로직."""
from __future__ import annotations

import math
from typing import Optional

SPEED_MPS = 0.8
MIN_TRAVEL_SEC = 3.0
DEFAULT_TRAVEL_SEC = 8.0
FLOOR_CHANGE_SEC = 5.0


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
        self._goal = None
        self._arrive_at = 0.0

    def handle_intent(self, intent, now: float) -> Optional[str]:
        """VicaIntent(pydantic 또는 유사 객체)를 받는다. 일어난 일을 돌려준다."""
        if self.state == "estopped":
            return "blocked_estop"
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

    @property
    def is_moving(self) -> bool:
        return self.state == "moving"

    def _travel_sec(self, dest) -> float:
        pose = getattr(dest, "pose", None)
        if pose is None:
            base = DEFAULT_TRAVEL_SEC
        else:
            dist = math.hypot(pose.x - self.x, pose.y - self.y)
            base = DEFAULT_TRAVEL_SEC if dist < 0.5 else max(MIN_TRAVEL_SEC,
                                                             dist / SPEED_MPS)
        if dest.floor != self.floor:
            base += FLOOR_CHANGE_SEC
        return base

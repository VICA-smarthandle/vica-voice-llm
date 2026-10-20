"""사람 접근 차임을 울릴 때를 정한다 — ROS·소리 장치 없는 순수 로직."""
from __future__ import annotations

APPROACHING = "approaching"


class ApproachChime:
    """다가가는 동안 period_sec 마다 울린다."""

    def __init__(self, period_sec: float = 2.0, stale_sec: float = 3.0,
                 after_speech_sec: float = 1.0, enabled: bool = True) -> None:
        self.period_sec = period_sec
        self.stale_sec = stale_sec
        self.after_speech_sec = after_speech_sec
        self.enabled = enabled
        self._approaching = False
        self._state_at = float("-inf")
        self._next_at = float("inf")

    def on_robot_state(self, dialog_state: str, now: float) -> None:
        approaching = dialog_state == APPROACHING
        if approaching and not self._approaching:
            self._next_at = now
        self._approaching = approaching
        self._state_at = now

    def on_speech_end(self, now: float) -> None:
        self._next_at = max(self._next_at, now + self.after_speech_sec)

    def due(self, now: float) -> bool:
        return (self.enabled and self._approaching
                and now - self._state_at <= self.stale_sec
                and now >= self._next_at)

    def played(self, now: float) -> None:
        self._next_at = now + self.period_sec

"""위치 비프를 울릴 때를 정한다 — ROS·소리 장치 없는 순수 로직."""
from __future__ import annotations

from typing import Optional

from .tts_queue import EMERGENCY

GRIP_WAIT = "grip_wait"
ESTOPPED = "estopped"
_SECTIONS = (GRIP_WAIT, ESTOPPED)


class LocatorBeep:
    """구간마다 한 번, 그 구간의 멘트가 끝난 뒤에 울린다."""

    def __init__(self, after_speech_sec: float = 0.5, stale_sec: float = 3.0) -> None:
        self.after_speech_sec = after_speech_sec
        self.stale_sec = stale_sec
        self._state = ""
        self._state_at = float("-inf")
        self._section = 0
        self._played_section = -1
        self._due_at: Optional[float] = None

    @property
    def state(self) -> str:
        return self._state

    def on_robot_state(self, dialog_state: str, now: float) -> None:
        if dialog_state != self._state:
            self._state = dialog_state
            self._section += 1
            self._due_at = None
        self._state_at = now

    def on_speech_end(self, priority: str, now: float) -> None:
        if self._state not in _SECTIONS or self._played_section == self._section:
            return
        if (self._state == GRIP_WAIT or self._due_at is not None
                or priority == EMERGENCY):
            self._due_at = now + self.after_speech_sec

    def _fresh(self, now: float) -> bool:
        return now - self._state_at <= self.stale_sec

    def due(self, now: float) -> bool:
        return (self._due_at is not None and now >= self._due_at
                and self._played_section != self._section and self._fresh(now))

    def played(self, now: float) -> None:
        self._played_section = self._section
        self._due_at = None

    def still_wanted(self, now: float) -> bool:
        """울리던 비프를 이어도 되는가 — 같은 구간이 이어지고 상태 알림이 살아 있다."""
        return (self._played_section == self._section
                and self._state in _SECTIONS and self._fresh(now))

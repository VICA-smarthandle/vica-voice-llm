"""위치 비프를 울릴 때를 정한다 — ROS·소리 장치 없는 순수 로직.

2026-10-23 사용자 결정: 시각장애인이 로봇 위치를 소리로 찾게, 두 때에만 4초 비프
(audio_cue.locator_beep)를 한 번 울린다. 미션이 1초마다(대화 단계가 바뀌면 즉시)
보내는 /vica/robot_state 의 dialog_state 로 구간을 안다.

- 손잡이 잡기 요청(dialog_state "grip_wait"): 미션이 "손잡이가 진동하고 있습니다.
  잡아주세요."를 말하고 잡기를 기다리는 동안. 그 멘트가 끝나고 after_speech_sec 뒤에
  울린다. 잡음 판정이 나면 미션이 진동 반복을 멈추는 같은 tick 에 단계를 바꿔 즉시
  알리므로, 단계가 바뀌는 순간 끊으면 진동과 같이 멈춘다.
- 비상정지(dialog_state "estopped"): 미션이 비상정지를 말로 알린 경우만(emergency 등급
  멘트). 서 있을 때 조용히 걸리는 비상정지(버튼·앱·통신·부팅 래치)는 미션도 말하지
  않으므로 울리지 않는다 — 울리면 부팅·통신 순단마다 4초 비프가 난다.

구간(같은 dialog_state 가 이어지는 동안)마다 한 번만 울린다. 끊겨도 다시 울리지 않는다 —
넘어짐 멘트는 10초마다 되풀이되지만 비프는 한 번이다. 목적지 도착·대기 중 호출 위치 진동·
해제 멘트에는 울리지 않는다.
"""
from __future__ import annotations

from typing import Optional

from .tts_queue import EMERGENCY

GRIP_WAIT = "grip_wait"
ESTOPPED = "estopped"
_SECTIONS = (GRIP_WAIT, ESTOPPED)


class LocatorBeep:
    """구간마다 한 번, 그 구간의 멘트가 끝난 뒤에 울린다.

    - 상태 알림이 stale_sec 넘게 끊기면 울리지 않고, 울리던 것도 끊는다(미션이 죽어도
      비프가 이어지지 않게).
    - 비상정지 구간은 emergency 등급 멘트가 끝나야 울릴 차례가 된다. 차례가 된 뒤 끝난
      말은 등급과 상관없이 비프를 그 말 뒤로 민다.
    """

    def __init__(self, after_speech_sec: float = 0.5, stale_sec: float = 3.0) -> None:
        self.after_speech_sec = after_speech_sec
        self.stale_sec = stale_sec
        self._state = ""
        self._state_at = float("-inf")
        self._section = 0               # dialog_state 가 바뀔 때마다 1씩 는다
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

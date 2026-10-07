"""사람 접근 차임을 울릴 때를 정한다 — ROS·소리 장치 없는 순수 로직.

2026-10-07 사용자 결정: 시각장애인에게 다가가기 시작할 때 미션이 "안내로봇 비카가
다가가고 있어요"를 한 번 말하고, 다가가는 동안 TTS 노드가 2초마다 종 두 음을 울려
위치를 알린다. 다가가는 중인지는 미션이 1초마다 보내는 /vica/robot_state 의
dialog_state("approaching")로 안다 — 접근이 어떤 길로 끝나든(도착·취소·비상정지)
상태가 바뀌면 멈추므로 끝내는 신호를 따로 둘 필요가 없다.
"""
from __future__ import annotations

APPROACHING = "approaching"


class ApproachChime:
    """다가가는 동안 period_sec 마다 울린다.

    - 상태 알림이 stale_sec 넘게 끊기면 멈춘다(미션이 죽어도 계속 울리지 않게).
    - 말이 끝난 직후 after_speech_sec 동안은 쉰다(첫 인사에 겹치지 않게).
    - enabled=False 면 울리지 않는다(.env VICA_APPROACH_CHIME=off).
    """

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
            self._next_at = now          # 새 접근 — 말이 없으면 바로 울린다
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

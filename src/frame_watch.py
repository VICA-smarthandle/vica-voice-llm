"""reSpeaker 소리 끊김 기록 — 감시 루프가 소리 조각을 오래 못 받으면 터미널에 남길 문장을 만든다.

2026-11-11 사용자: "usb 가 끊기면 끊겼다고 음성 노드 터미널에 남겨야". 11-04 전시회에서 USB 가 끊기자 감시
스레드가 `q.get()` 에서 무기한 멈춰 아무 기록도 없이 듣기를 멈췄다(CPU 0 %). 복구(재시작)는 하지 않는다 —
기록만 한다(설계 2026-11-11-speaker-follow-design.md G3). 시각 규칙만 있는 순수 모듈이다.
"""
from __future__ import annotations

from typing import Dict, Optional

GAP_SEC = 1.0          # 이만큼 조각이 없으면 끊김 의심
REPEAT_SEC = 10.0      # 끊김이 이어지면 이 간격으로 다시 남긴다 · 같은 상태 문장도 이 간격으로만


class FrameWatch:
    def __init__(self, gap_sec: float = GAP_SEC, repeat_sec: float = REPEAT_SEC):
        self.gap_sec = gap_sec
        self.repeat_sec = repeat_sec
        self._last_frame: Optional[float] = None
        self._reported_at: Optional[float] = None     # 이번 끊김을 마지막으로 남긴 시각(None = 안 남김)
        self._status_at: Dict[str, float] = {}

    def on_frame(self, now: float) -> Optional[str]:
        """소리 조각이 들어왔다. 끊김을 남긴 뒤 처음 들어온 것이면 회복 문장."""
        line = None
        if self._reported_at is not None and self._last_frame is not None:
            line = f"[USB] 소리 다시 들어옴(끊김 {now - self._last_frame:.1f} s)"
        self._last_frame = now
        self._reported_at = None
        return line

    def on_idle(self, now: float) -> Optional[str]:
        """조각을 기다리다 시간이 지났다. 끊김이 gap_sec 를 넘으면 처음 한 번, 그 뒤 repeat_sec 마다."""
        if self._last_frame is None:
            return None
        gap = now - self._last_frame
        if gap < self.gap_sec:
            return None
        if self._reported_at is not None and now - self._reported_at < self.repeat_sec:
            return None
        self._reported_at = now
        return f"[USB] reSpeaker 소리가 {int(gap)} s 째 안 들어옴 — 끊김 의심"

    def on_status(self, text: str, now: float) -> Optional[str]:
        """sounddevice 콜백 상태(overflow 등). 같은 문장은 repeat_sec 에 한 번."""
        last = self._status_at.get(text)
        if last is not None and now - last < self.repeat_sec:
            return None
        self._status_at[text] = now
        return f"[USB] 입력 상태: {text}"

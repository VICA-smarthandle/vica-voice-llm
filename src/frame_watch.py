"""reSpeaker 소리 끊김 기록 — 감시 루프가 소리 조각을 오래 못 받으면 터미널에 남길 문장을 만든다."""
from __future__ import annotations

from typing import Dict, Optional

GAP_SEC = 1.0
REPEAT_SEC = 10.0


class FrameWatch:
    def __init__(self, gap_sec: float = GAP_SEC, repeat_sec: float = REPEAT_SEC):
        self.gap_sec = gap_sec
        self.repeat_sec = repeat_sec
        self._last_frame: Optional[float] = None
        self._reported_at: Optional[float] = None
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

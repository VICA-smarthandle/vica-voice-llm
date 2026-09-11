"""상시(Always-on) 긴급어 감지 모니터 (CLAUDE.md Phase 4)."""
from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional

import numpy as np

from .emergency_filter import detect_emergency
from .schema import EmergencyEvent

SAMPLE_RATE = 16000


class EmergencyMonitor:
    """마이크를 상시 감시하며 긴급어를 감지한다."""

    def __init__(
        self,
        on_event: Callable[[EmergencyEvent], None],
        transcribe: Optional[Callable[[np.ndarray], str]] = None,
        window_sec: float = 2.0,
        hop_sec: float = 0.5,
        rms_threshold: float = 0.01,
        cooldown_sec: float = 2.0,
        max_mute_sec: float = 8.0,
    ):
        self._on_event = on_event
        self._transcribe = transcribe
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.rms_threshold = rms_threshold
        self.cooldown_sec = cooldown_sec
        self.max_mute_sec = max_mute_sec

        self._last_event_at = 0.0
        self._muted = False
        self._muted_at = 0.0
        self._resume_at = 0.0
        self._buffer = np.zeros(self._window_samples, dtype=np.float32)
        self._buffer_lock = threading.Lock()

    @property
    def _window_samples(self) -> int:
        return int(self.window_sec * SAMPLE_RATE)

    def set_muted(self, muted: bool, now: Optional[float] = None) -> None:
        """TTS 재생 상태를 반영한다 (/vica/tts_state 구독자가 호출)."""
        now = time.time() if now is None else now
        if muted:
            if not self._muted:
                self._muted_at = now
            self._muted = True
            return

        if self._muted:
            self._muted = False
            self._clear_buffer()
            self._resume_at = now + self.window_sec

    def is_muted(self, now: Optional[float] = None) -> bool:
        """현재 감시를 쉬는 중인지. max_mute_sec 를 넘기면 스스로 해제한다."""
        if not self._muted:
            return False
        now = time.time() if now is None else now
        if now - self._muted_at >= self.max_mute_sec:
            self.set_muted(False, now=now)
            return False
        return True

    def _clear_buffer(self) -> None:
        with self._buffer_lock:
            self._buffer = np.zeros(self._window_samples, dtype=np.float32)

    def _get_transcribe(self) -> Callable[[np.ndarray], str]:
        """STT 를 늦게 로드한다 (테스트에서는 주입된 가짜를 쓰므로 로드 안 함)."""
        if self._transcribe is None:
            from .stt import VicaSTT

            model = os.environ.get("VICA_EMERGENCY_STT_MODEL", "medium")
            self._transcribe = VicaSTT(model_size=model).transcribe
        return self._transcribe

    def process_window(self, audio: np.ndarray, now: Optional[float] = None) -> Optional[EmergencyEvent]:
        """오디오 창 하나를 검사한다. 긴급어가 있으면 이벤트를 만들어 콜백까지 부른다."""
        now = time.time() if now is None else now

        if self.is_muted(now):
            return None

        if now < self._resume_at:
            return None

        if now - self._last_event_at < self.cooldown_sec:
            return None

        if audio.size == 0:
            return None
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < self.rms_threshold:
            return None

        text = self._get_transcribe()(audio)
        keyword = detect_emergency(text)
        if keyword is None:
            return None

        event = EmergencyEvent(keyword=keyword, source_text=text, detected_at=now)
        self._last_event_at = now
        self._on_event(event)
        return event

    def run(self) -> None:
        """마이크 상시 감시 루프 (blocking). Ctrl+C 로 종료."""
        import sounddevice as sd

        self._get_transcribe()
        window = self._window_samples
        self._clear_buffer()

        def callback(indata, _frames, _time, _status):
            chunk = indata[:, 0]
            with self._buffer_lock:
                self._buffer = np.concatenate([self._buffer, chunk])[-window:]

        print(f"긴급어 상시 감시 시작 (창 {self.window_sec}초 / 간격 {self.hop_sec}초, 종료: Ctrl+C)")
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
            while True:
                time.sleep(self.hop_sec)
                with self._buffer_lock:
                    snapshot = self._buffer.copy()
                self.process_window(snapshot)


def _print_event(event: EmergencyEvent) -> None:
    print(f"🚨 [긴급] '{event.keyword}' 감지! (인식: {event.source_text!r})")


if __name__ == "__main__":
    EmergencyMonitor(on_event=_print_event).run()

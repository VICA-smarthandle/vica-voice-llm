"""상시(Always-on) 긴급어 감지 모니터 (CLAUDE.md Phase 4).

push-to-talk 대화와 별개로, 마이크를 계속 들으며 "멈춰" 같은 긴급어를
LLM 없이 감지해 EmergencyEvent 를 만든다.

  마이크 상시 스트림 (window_sec 창을 hop_sec 간격으로 슬라이딩)
    -> 음량 게이트 (조용하면 STT 생략)
    -> whisper STT (짧은 창이라 빠름)
    -> emergency_filter.detect_emergency
    -> EmergencyEvent -> on_event 콜백

안전 원칙: 이 모듈은 '감지'만 한다. 실제 정지는 Safety Supervisor /
State Machine 이 한다. 로봇 이동 명령은 어디에도 없다.

CLI 데모 실행:
    .venv/bin/python -m src.emergency_monitor
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional

import numpy as np

from .emergency_filter import detect_emergency
from .schema import EmergencyEvent

SAMPLE_RATE = 16000  # whisper 표준 입력 샘플레이트


class EmergencyMonitor:
    """마이크를 상시 감시하며 긴급어를 감지한다.

    transcribe 를 주입식으로 받아서, 테스트에서는 가짜 STT 를 넣을 수 있다.
    """

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
        # mute 가 걸린 채 TTS 노드가 죽으면 감시가 영구히 멈춘다. 그 상태를 막는
        # fail-safe: 이 시간이 지나면 해제 신호가 없어도 스스로 감시를 재개한다.
        self.max_mute_sec = max_mute_sec

        self._last_event_at = 0.0  # 마지막 이벤트 시각 (쿨다운용)
        self._muted = False
        self._muted_at = 0.0  # mute 가 걸린 시각 (fail-safe 계산용)
        self._resume_at = 0.0  # 이 시각 전까지는 판정을 보류한다
        self._buffer = np.zeros(self._window_samples, dtype=np.float32)
        self._buffer_lock = threading.Lock()

    @property
    def _window_samples(self) -> int:
        return int(self.window_sec * SAMPLE_RATE)

    # -- TTS 재생 중 감시 억제 -------------------------------------------------

    def set_muted(self, muted: bool, now: Optional[float] = None) -> None:
        """TTS 재생 상태를 반영한다 (/vica/tts_state 구독자가 호출).

        로봇이 말하는 동안 감시를 쉬어, 스피커로 나간 자기 목소리를 긴급어로
        오인하는 자가 트리거를 막는다.

        해제할 때가 중요하다. 창(window_sec)만큼의 오디오를 계속 담아 두므로
        해제 직후 버퍼에는 아직 로봇 목소리가 남아 있다. 그래서 버퍼를 비우고,
        새 소리가 창만큼 찰 때까지 판정을 보류한다.
        """
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

            # 짧은 창의 키워드 감지에는 small 이면 충분하다 (medium 은 대화용).
            model = os.environ.get("VICA_EMERGENCY_STT_MODEL", "small")
            self._transcribe = VicaSTT(model_size=model).transcribe
        return self._transcribe

    def process_window(self, audio: np.ndarray, now: Optional[float] = None) -> Optional[EmergencyEvent]:
        """오디오 창 하나를 검사한다. 긴급어가 있으면 이벤트를 만들어 콜백까지 부른다.

        순수 로직이라 마이크 없이 unit test 로 검증할 수 있다.
        """
        now = time.time() if now is None else now

        # 1) TTS 재생 중에는 쉰다 (자가 트리거 차단). set_muted 참고.
        if self.is_muted(now):
            return None

        # 2) 해제 직후 보류: 버퍼가 새 소리로 다시 찰 때까지 판정하지 않는다.
        if now < self._resume_at:
            return None

        # 3) 쿨다운: 방금 이벤트를 냈으면 같은 외침을 중복 감지하지 않는다.
        if now - self._last_event_at < self.cooldown_sec:
            return None

        # 4) 음량 게이트: 조용한 창은 STT 를 돌리지 않는다 (GPU 절약 + 환청 방지).
        if audio.size == 0:
            return None
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < self.rms_threshold:
            return None

        # 5) STT -> 긴급어 필터 (LLM 없음).
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

        self._get_transcribe()  # 루프 시작 전에 모델을 미리 로드
        window = self._window_samples
        self._clear_buffer()

        # 버퍼는 오디오 콜백 스레드가 쓰고, set_muted(ROS 콜백)와 판정 루프가 읽는다.
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

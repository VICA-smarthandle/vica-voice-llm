"""청각 안내음 생성·재생.

파형을 만드는 부분은 순수 함수라 소리 장치 없이 시험된다. 재생만 sounddevice 를
쓰고, 실패해도 예외를 밖으로 내지 않는다 — 소리가 안 나는 것이 파이프라인을
멈출 이유는 되지 않는다.

TTS 를 거치지 않는 이유:
    큐에 줄을 서지 않아 즉시 난다. 그리고 /vica/tts_state 를 켜지 않으므로
    긴급어 상시 감시가 쉬지 않는다. 0.15초짜리 순음이 "멈춰"·"비카야" 로
    오인될 여지는 사실상 없다.

소리 설계 (2026-08-05):
    호출 응답   880Hz 단음      — 기존 _ack_beep 과 같은 소리
    좌회전      660Hz 단음      — 낮은 음
    우회전      990Hz 단음      — 높은 음
    도착        784→1047Hz 2음  — 상행. 회전음과 뚜렷이 구분된다

좌우를 음높이로 나눈 것은 1차 안이다. 스피커가 스테레오면 좌/우 채널로 나누는
편이 직관적이다(소리 나는 쪽으로 몸이 반응한다). AEC 배선 후 출력 특성을 확인하고
정한다.
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 44100
DEFAULT_VOLUME = 0.4

WAKE_ACK_HZ = 880.0
TURN_LEFT_HZ = 660.0
TURN_RIGHT_HZ = 990.0
ARRIVED_HZ = (784.0, 1047.0)

CUE_SEC = 0.15
WAKE_SEC = 0.12


def tone(freq_hz: float, duration_sec: float, volume: float = DEFAULT_VOLUME) -> np.ndarray:
    """한 음의 파형. 창(Hanning)을 씌워 시작·끝의 딱 소리를 없앤다."""
    if duration_sec <= 0 or freq_hz <= 0:
        return np.zeros(0, dtype=np.float32)
    samples = int(duration_sec * SAMPLE_RATE)
    t = np.arange(samples) / SAMPLE_RATE
    wave = volume * np.sin(2 * np.pi * freq_hz * t) * np.hanning(samples)
    return wave.astype(np.float32)


def sequence(freqs_hz, duration_sec: float, volume: float = DEFAULT_VOLUME) -> np.ndarray:
    """여러 음을 이어 붙인다 (도착음처럼 두 음 이상인 경우)."""
    parts = [tone(f, duration_sec, volume) for f in freqs_hz]
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)


def wake_ack() -> np.ndarray:
    return tone(WAKE_ACK_HZ, WAKE_SEC)


def turn_left() -> np.ndarray:
    return tone(TURN_LEFT_HZ, CUE_SEC)


def turn_right() -> np.ndarray:
    return tone(TURN_RIGHT_HZ, CUE_SEC)


def arrived() -> np.ndarray:
    return sequence(ARRIVED_HZ, CUE_SEC)


def play(wave: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bool:
    """파형을 재생한다. 재생 장치가 없거나 실패해도 예외를 내지 않는다.

    blocking 하지 않는다 — 호출한 콜백을 붙잡아 두면 안 된다.
    """
    if wave is None or len(wave) == 0:
        return False
    try:
        from . import audio_out

        audio_out.play(wave, sample_rate)
        return True
    except Exception:
        return False

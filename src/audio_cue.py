"""청각 안내음 생성·재생."""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 44100
DEFAULT_VOLUME = 0.4


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


THINKING_NOTES_HZ = (523.25, 659.25, 783.99, 659.25)
THINKING_NOTE_SEC = 0.42
THINKING_GAP_SEC = 0.10
THINKING_VOLUME = 0.16


def thinking_loop() -> np.ndarray:
    parts: list[np.ndarray] = []
    gap = np.zeros(int(THINKING_GAP_SEC * SAMPLE_RATE), dtype=np.float32)
    for f in THINKING_NOTES_HZ:
        parts.append(tone(f, THINKING_NOTE_SEC, THINKING_VOLUME))
        parts.append(gap)
    return np.concatenate(parts)

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


APPROACH_CHIME_NOTES_HZ = (880.0, 1174.66)
APPROACH_CHIME_GAP_SEC = 0.24
APPROACH_CHIME_NOTE_SEC = 1.0
APPROACH_CHIME_PARTIALS = ((1, 1.0, 0.45), (2, 0.28, 0.25), (3, 0.10, 0.14))
APPROACH_CHIME_ATTACK_SEC = 0.004


def _bell_note(freq_hz: float) -> np.ndarray:
    samples = int(APPROACH_CHIME_NOTE_SEC * SAMPLE_RATE)
    t = np.arange(samples) / SAMPLE_RATE
    wave = np.zeros(samples, dtype=np.float64)
    for multiple, amp, fade_sec in APPROACH_CHIME_PARTIALS:
        wave += amp * np.exp(-t * 3.0 / fade_sec) * np.sin(2 * np.pi * freq_hz * multiple * t)
    attack = max(1, int(APPROACH_CHIME_ATTACK_SEC * SAMPLE_RATE))
    wave[:attack] *= np.linspace(0.0, 1.0, attack)
    return wave


def approach_chime() -> np.ndarray:
    """종 두 음 "딩—동↗" 한 번. 재생 크기는 부르는 쪽이 정한다(audio_out peak_dbfs)."""
    gap = int(APPROACH_CHIME_GAP_SEC * SAMPLE_RATE)
    first, second = (_bell_note(f) for f in APPROACH_CHIME_NOTES_HZ)
    wave = np.zeros(gap + len(second), dtype=np.float64)
    wave[: len(first)] += first
    wave[gap:] += second
    peak = float(np.max(np.abs(wave)))
    return (wave / peak).astype(np.float32) if peak > 0 else wave.astype(np.float32)

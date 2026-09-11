"""안내음 파형 검증 (소리 장치 없이)."""
from __future__ import annotations

import numpy as np

from src import audio_cue


def _peak_freq(wave: np.ndarray, sample_rate: int = audio_cue.SAMPLE_RATE) -> float:
    """파형에서 가장 센 주파수를 찾는다."""
    spectrum = np.abs(np.fft.rfft(wave))
    return float(np.fft.rfftfreq(len(wave), 1 / sample_rate)[np.argmax(spectrum)])


def test_tone_has_requested_frequency_and_length():
    wave = audio_cue.tone(880.0, 0.2)
    assert len(wave) == int(0.2 * audio_cue.SAMPLE_RATE)
    assert abs(_peak_freq(wave) - 880.0) < 15.0


def test_tone_starts_and_ends_at_silence():
    """창을 씌우지 않으면 시작·끝에서 '딱' 소리가 난다."""
    wave = audio_cue.tone(880.0, 0.2)
    assert abs(wave[0]) < 1e-3
    assert abs(wave[-1]) < 1e-3


def test_tone_does_not_clip():
    wave = audio_cue.tone(880.0, 0.2)
    assert float(np.max(np.abs(wave))) <= 1.0


def test_arrived_is_two_rising_tones():
    """도착음은 단음이 아닌 상행 2음이다 — 다른 알림음과 구분된다."""
    wave = audio_cue.arrived()
    half = len(wave) // 2
    assert _peak_freq(wave[:half]) < _peak_freq(wave[half:])


def test_degenerate_input_returns_empty_wave():
    assert len(audio_cue.tone(880.0, 0.0)) == 0
    assert len(audio_cue.tone(0.0, 0.2)) == 0
    assert len(audio_cue.sequence([], 0.1)) == 0


def test_play_is_safe_without_audio_device():
    """소리가 안 나는 것이 파이프라인을 멈출 이유는 되지 않는다."""
    assert audio_cue.play(np.zeros(0, dtype=np.float32)) is False
    assert audio_cue.play(None) is False


def test_thinking_loop_is_a_quiet_seamless_cycle():
    """"생각 중" 운율(2026-09-01): 말 대신 배경으로 반복되는 한 바퀴.

    조건 셋 — ① 말보다 조용해야 하고(음량 상한) ② 끝이 무음이라 반복
    이음새에 딱 소리가 없어야 하며 ③ 반복해 깔 만한 길이(1초 이상)여야 한다.
    """
    wave = audio_cue.thinking_loop()
    assert wave.dtype == np.float32
    assert len(wave) >= audio_cue.SAMPLE_RATE  # ③
    assert np.abs(wave).max() <= audio_cue.THINKING_VOLUME + 1e-6  # ①
    tail = wave[-int(0.05 * audio_cue.SAMPLE_RATE):]
    assert np.abs(tail).max() == 0.0  # ②

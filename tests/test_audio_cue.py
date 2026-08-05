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


def test_left_is_lower_than_right():
    """좌우를 음높이로 구분한다 — 낮으면 왼쪽, 높으면 오른쪽."""
    assert _peak_freq(audio_cue.turn_left()) < _peak_freq(audio_cue.turn_right())


def test_arrived_is_two_rising_tones():
    """도착음은 회전음과 뚜렷이 구분돼야 한다 (단음 아닌 상행 2음)."""
    wave = audio_cue.arrived()
    half = len(wave) // 2
    assert _peak_freq(wave[:half]) < _peak_freq(wave[half:])


def test_arrived_is_longer_than_a_turn_cue():
    assert len(audio_cue.arrived()) > len(audio_cue.turn_left())


def test_degenerate_input_returns_empty_wave():
    assert len(audio_cue.tone(880.0, 0.0)) == 0
    assert len(audio_cue.tone(0.0, 0.2)) == 0
    assert len(audio_cue.sequence([], 0.1)) == 0


def test_play_is_safe_without_audio_device():
    """소리가 안 나는 것이 파이프라인을 멈출 이유는 되지 않는다."""
    assert audio_cue.play(np.zeros(0, dtype=np.float32)) is False
    assert audio_cue.play(None) is False

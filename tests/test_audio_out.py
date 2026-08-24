"""audio_out 순수 변환 검증 — 소리 장치 없이 돈다.

-3dBFS 정규화가 볼륨의 유일한 손잡이다(reSpeaker 재생부에 ALSA 믹서 없음).
그래서 정규화가 틀리면 소리가 안 들리거나 클리핑으로 AEC 가 망가진다.
"""
import numpy as np
import pytest

from src import audio_out

MINUS_3DBFS = 10.0 ** (-3.0 / 20.0)  # ≈ 0.7079


def test_normalize_raises_quiet_wave_to_target():
    wave = np.full(100, 0.01, dtype=np.float32)
    out = audio_out.normalize_peak(wave, peak_dbfs=-3.0)
    assert np.max(np.abs(out)) == pytest.approx(MINUS_3DBFS, rel=1e-4)


def test_normalize_lowers_loud_wave_to_target():
    wave = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    out = audio_out.normalize_peak(wave, peak_dbfs=-3.0)
    assert np.max(np.abs(out)) == pytest.approx(MINUS_3DBFS, rel=1e-4)
    assert np.max(np.abs(out)) < 1.0  # 클리핑 없음 (AEC 보호)


def test_normalize_keeps_silence_untouched():
    wave = np.zeros(50, dtype=np.float32)
    out = audio_out.normalize_peak(wave)
    assert np.all(out == 0)


def test_normalize_keeps_waveform_shape():
    """정규화는 크기만 바꾸고 파형(비율)은 보존해야 한다."""
    wave = np.array([0.1, -0.2, 0.4], dtype=np.float32)
    out = audio_out.normalize_peak(wave, peak_dbfs=-3.0)
    assert out[1] / out[0] == pytest.approx(-2.0, rel=1e-4)
    assert out[2] / out[0] == pytest.approx(4.0, rel=1e-4)


def test_peak_env_override(monkeypatch):
    monkeypatch.setenv("VICA_TTS_PEAK_DBFS", "-6.0")
    wave = np.full(10, 0.5, dtype=np.float32)
    out = audio_out.normalize_peak(wave)
    assert np.max(np.abs(out)) == pytest.approx(10.0 ** (-6.0 / 20.0), rel=1e-4)


def test_peak_env_garbage_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("VICA_TTS_PEAK_DBFS", "시끄럽게")
    assert audio_out.target_peak_dbfs() == audio_out.DEFAULT_PEAK_DBFS


def test_resample_44k_to_16k_length():
    one_sec = np.random.default_rng(0).standard_normal(44100).astype(np.float32)
    out = audio_out.resample(one_sec, 44100, 16000)
    assert abs(len(out) - 16000) <= 1
    assert out.dtype == np.float32


def test_resample_same_rate_is_noop():
    wave = np.ones(160, dtype=np.float32)
    assert audio_out.resample(wave, 16000, 16000) is wave


def test_resample_stereo_keeps_channels():
    stereo = np.zeros((44100, 2), dtype=np.float32)
    out = audio_out.resample(stereo, 44100, 16000)
    assert out.ndim == 2 and out.shape[1] == 2


def test_to_stereo_duplicates_mono():
    mono = np.array([0.1, 0.2], dtype=np.float32)
    out = audio_out.to_stereo(mono)
    assert out.shape == (2, 2)
    assert np.array_equal(out[:, 0], out[:, 1])


def test_to_stereo_passes_stereo_through():
    stereo = np.zeros((5, 2), dtype=np.float32)
    assert audio_out.to_stereo(stereo) is stereo


def test_prepare_full_pipeline_for_respeaker():
    """44.1kHz 모노 → 16kHz 스테레오, 최고점 -3dBFS (reSpeaker 재생 조건)."""
    one_sec = (0.05 * np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)).astype(
        np.float32
    )
    out = audio_out.prepare(one_sec, 44100, device_rate=16000, device_channels=2)
    assert out.ndim == 2 and out.shape[1] == 2
    assert abs(out.shape[0] - 16000) <= 1
    assert np.max(np.abs(out)) == pytest.approx(MINUS_3DBFS, rel=1e-3)


def test_prepare_default_device_only_normalizes():
    wave = np.full(100, 0.2, dtype=np.float32)
    out = audio_out.prepare(wave, 44100, device_rate=None, device_channels=1)
    assert out.shape == wave.shape
    assert np.max(np.abs(out)) == pytest.approx(MINUS_3DBFS, rel=1e-4)

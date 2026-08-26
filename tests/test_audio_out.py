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


# ---- 장치 선택: 무조건 reSpeaker ---------------------------------------------
#
# 이 로봇의 소리 출구는 reSpeaker 하나뿐이다 (2026-08-26 사용자 확정).
# 노드가 켜질 때 pulse 기본 출력을 reSpeaker 로 강제(+음소거 해제)하고,
# 직통 장치가 목록에서 사라져도(마이크 감시가 카드를 잡는 8/12 함정) pulse
# 경유로 같은 곳에 낸다. reSpeaker 가 아예 없으면 폴백 없이 실패한다.


class _FakeSd:
    def __init__(self, devices):
        self._devices = devices

    def query_devices(self):
        return self._devices


RESPEAKER_OUT = {"name": "ReSpeaker 4 Mic Array: USB Audio (hw:2,0)",
                 "max_output_channels": 2, "default_samplerate": 16000.0}
PULSE_OUT = {"name": "pulse", "max_output_channels": 32, "default_samplerate": 44100.0}
HDMI_OUT = {"name": "hdmi", "max_output_channels": 2, "default_samplerate": 44100.0}


def _with_devices(monkeypatch, devices, routed):
    import sys
    monkeypatch.setitem(sys.modules, "sounddevice", _FakeSd(devices))
    monkeypatch.setattr(audio_out, "_ensure_respeaker_route", lambda: routed)
    monkeypatch.delenv("VICA_TTS_DEVICE", raising=False)


def test_direct_respeaker_output_wins(monkeypatch):
    _with_devices(monkeypatch, [HDMI_OUT, RESPEAKER_OUT, PULSE_OUT], routed=True)
    assert audio_out._find_device() == (1, 16000, 2)


def test_hidden_direct_goes_through_pulse(monkeypatch):
    """마이크 감시가 카드를 잡으면 직통이 목록에서 사라진다(8/12 실측).
    기본 출력이 reSpeaker 로 강제돼 있으므로 pulse 경유 = 같은 스피커다."""
    _with_devices(monkeypatch, [HDMI_OUT, PULSE_OUT], routed=True)
    index, rate, ch = audio_out._find_device()
    assert index == 1 and ch == 32


def test_no_respeaker_at_all_refuses(monkeypatch):
    """reSpeaker 가 시스템에 없으면(강제 실패) 다른 스피커로 새지 않는다 —
    소리는 나는데 AEC 참조가 깨진 채 도는 것이 가장 위험하다."""
    _with_devices(monkeypatch, [HDMI_OUT, PULSE_OUT], routed=False)
    assert audio_out._find_device() is None

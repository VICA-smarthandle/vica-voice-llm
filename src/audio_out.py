"""로봇 소리의 단일 출구 — reSpeaker 재생 경로 (AEC 참조 신호)."""
from __future__ import annotations

import os
import threading
from math import gcd
from typing import Optional

import numpy as np

DEFAULT_PEAK_DBFS = -3.0

CHUNK = 1600

_stop_flag = threading.Event()

_device_cache: Optional[tuple[int, int, int]] = None
_device_searched = False


def target_peak_dbfs() -> float:
    try:
        return float(os.environ.get("VICA_TTS_PEAK_DBFS", DEFAULT_PEAK_DBFS))
    except ValueError:
        return DEFAULT_PEAK_DBFS


def normalize_peak(wave: np.ndarray, peak_dbfs: Optional[float] = None) -> np.ndarray:
    """최고점을 peak_dbfs 로 맞춘다 (작으면 키우고 크면 줄인다). 무음은 그대로."""
    wave = np.asarray(wave, dtype=np.float32)
    if wave.size == 0:
        return wave
    peak = float(np.max(np.abs(wave)))
    if peak <= 0.0:
        return wave
    if peak_dbfs is None:
        peak_dbfs = target_peak_dbfs()
    target = 10.0 ** (peak_dbfs / 20.0)
    return (wave * (target / peak)).astype(np.float32)


def resample(wave: np.ndarray, rate: int, target_rate: int) -> np.ndarray:
    """샘플레이트 변환. 2차원(스테레오)이면 시간축(axis 0) 기준."""
    wave = np.asarray(wave, dtype=np.float32)
    if wave.size == 0 or int(rate) == int(target_rate):
        return wave
    from scipy.signal import resample_poly

    g = gcd(int(rate), int(target_rate))
    out = resample_poly(wave, int(target_rate) // g, int(rate) // g, axis=0)
    return out.astype(np.float32)


def to_stereo(wave: np.ndarray) -> np.ndarray:
    """모노를 좌우 동일한 스테레오로 확장한다 (3.5mm 양쪽 채널을 채운다)."""
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim == 1:
        return np.column_stack([wave, wave])
    return wave


def prepare(wave: np.ndarray, rate: int,
            device_rate: Optional[int], device_channels: int) -> np.ndarray:
    """재생 직전 변환 파이프라인. 정규화는 리샘플 뒤에 한다 —"""
    out = np.asarray(wave, dtype=np.float32)
    if device_rate is not None:
        out = resample(out, rate, device_rate)
    out = normalize_peak(out)
    if device_channels >= 2:
        out = to_stereo(out)
    return out


_alsa_handler_ref = None


def _silence_alsa_errors() -> None:
    global _alsa_handler_ref
    try:
        from ctypes import CDLL, CFUNCTYPE, c_char_p, c_int

        handler_type = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
        _alsa_handler_ref = handler_type(lambda *_args: None)
        CDLL("libasound.so.2").snd_lib_error_set_handler(_alsa_handler_ref)
    except Exception:
        pass


_silence_alsa_errors()


def _ensure_respeaker_route() -> bool:
    """pulse 기본 출력을 reSpeaker 로 강제하고 음소거를 푼다. 성공 여부 반환."""
    try:
        import subprocess

        sinks = subprocess.run(
            ["pactl", "list", "sinks", "short"],
            capture_output=True, text=True, timeout=3,
        ).stdout
        name = ""
        for line in sinks.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and (
                "seeed" in parts[1].lower() or "respeaker" in parts[1].lower()
            ):
                name = parts[1]
                break
        if not name:
            return False
        subprocess.run(["pactl", "set-default-sink", name], timeout=3)
        subprocess.run(["pactl", "set-sink-mute", name, "0"], timeout=3)
        return True
    except Exception:
        return False


def _find_device() -> Optional[tuple[int, int, int]]:
    import sounddevice as sd

    devices = sd.query_devices()
    want = os.environ.get("VICA_TTS_DEVICE", "").strip()

    if want:
        if want.isdigit():
            index = int(want)
            d = devices[index]
            return index, int(d["default_samplerate"]), int(d["max_output_channels"])
        for i, d in enumerate(devices):
            if want.lower() in d["name"].lower() and d["max_output_channels"] >= 1:
                return i, int(d["default_samplerate"]), int(d["max_output_channels"])
        return None

    routed = _ensure_respeaker_route()

    for i, d in enumerate(devices):
        if "respeaker" in d["name"].lower() and d["max_output_channels"] >= 1:
            return i, int(d["default_samplerate"]), int(d["max_output_channels"])

    if routed:
        for i, d in enumerate(devices):
            if d["name"].strip().lower() == "pulse" and d["max_output_channels"] >= 1:
                return i, int(d["default_samplerate"]), int(d["max_output_channels"])

    return None


def output_device() -> Optional[tuple[int, int, int]]:
    """(장치 번호, 샘플레이트, 채널 수) 또는 None(기본 장치). 첫 호출 때 1회 탐색."""
    global _device_cache, _device_searched
    if not _device_searched:
        try:
            _device_cache = _find_device()
        except Exception:
            _device_cache = None
        _device_searched = True
    return _device_cache


def reset_device_cache() -> None:
    """장치를 꽂거나 뺀 뒤 재탐색이 필요할 때 (그리고 시험용)."""
    global _device_cache, _device_searched, _out_stream, _out_key
    _device_cache = None
    _device_searched = False
    if _out_stream is not None:
        try:
            _out_stream.close()
        except Exception:
            pass
        _out_stream = None
        _out_key = None


_out_stream = None
_out_key: Optional[tuple] = None


def _persistent_stream(index: Optional[int], rate: int, channels: int):
    global _out_stream, _out_key
    import sounddevice as sd

    key = (index, rate, channels)
    if _out_stream is not None and _out_key == key:
        return _out_stream
    if _out_stream is not None:
        try:
            _out_stream.close()
        except Exception:
            pass
    _out_stream = sd.OutputStream(samplerate=rate, device=index,
                                  channels=channels, dtype="float32")
    _out_stream.start()
    _out_key = key
    return _out_stream


def play(wave: np.ndarray, rate: int, blocking: bool = False) -> None:
    """정규화·리샘플 후 재생한다. 실패 처리는 호출자 몫(예외 그대로 전파) —"""
    import sounddevice as sd

    wave = np.asarray(wave, dtype=np.float32)
    if wave.size == 0:
        return

    device = output_device()
    if device is None:
        raise RuntimeError(
            "reSpeaker 재생 장치를 찾을 수 없다 (연결 또는 VICA_TTS_DEVICE 확인)")
    index, out_rate, channels = device
    out = prepare(wave, rate, out_rate, channels)

    if not blocking:
        sd.play(out, out_rate, device=index)
        return

    if out.ndim == 1:
        out = out.reshape(-1, 1)
    _stop_flag.clear()
    stream = _persistent_stream(index, out_rate, out.shape[1])
    for i in range(0, len(out), CHUNK):
        if _stop_flag.is_set():
            break
        stream.write(np.ascontiguousarray(out[i:i + CHUNK]))


def stop() -> None:
    """재생 중인 소리를 즉시 끊는다 (긴급 선점·barge-in 용). 스레드 안전."""
    _stop_flag.set()
    try:
        import sounddevice as sd

        sd.stop()
    except Exception:
        pass

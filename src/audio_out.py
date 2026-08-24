"""로봇 소리의 단일 출구 — reSpeaker 재생 경로 (AEC 참조 신호).

XVF-3000 의 AEC 는 "자기가 재생한 소리"만 참조로 알고 마이크 입력에서 뺀다.
다른 장치(HDMI 등)로 나간 소리는 빼야 할 목록에 없어서 마이크에 그대로
되돌아온다. 그래서 TTS·효과음·인사 등 로봇이 내는 모든 소리는 이 모듈을
지나 reSpeaker 로 나가야 한다 (vica-wakeword/docs/integration-design.md 8절 D6).

- 재생 경로는 장치 기본 샘플레이트(16kHz) 전용이라 리샘플한다.
- 최고점(peak)을 -3dBFS 로 정규화한다. reSpeaker 재생부에는 ALSA 볼륨
  조절이 없어(믹서 컨트롤 0개) 이 정규화가 유일한 볼륨 수단이다. 클리핑은
  AEC 성능을 떨어뜨리므로 0dBFS 까지 올리지 않는다.
- reSpeaker 가 없으면(개발 PC) 기본 장치로 폴백한다 — 정규화만 적용.

순수 변환(정규화·리샘플·스테레오 확장)은 소리 장치 없이 시험된다.

환경변수:
    VICA_TTS_DEVICE     출력 장치 강제 지정 (번호 또는 이름 일부). 미지정 시
                        reSpeaker 자동 탐색.
    VICA_TTS_PEAK_DBFS  정규화 목표 (기본 -3.0)
"""
from __future__ import annotations

import os
import threading
from math import gcd
from typing import Optional

import numpy as np

DEFAULT_PEAK_DBFS = -3.0

# blocking 재생의 조각 크기(샘플). 조각 사이마다 중단 깃발을 확인하므로
# stop() 의 반응 지연 상한이 이 값이다 (16kHz 기준 0.1초).
CHUNK = 1600

# 재생 중단 깃발. sd.wait() 중인 스레드에 다른 스레드가 sd.stop() 을 걸면
# PortAudio 가 교착할 수 있어(2026-08-24 barge-in 데모에서 행 실측),
# blocking 재생은 조각 쓰기 + 깃발 방식으로 끊는다.
_stop_flag = threading.Event()

# 장치 탐색 결과 캐시: (index, samplerate, channels) 또는 None(기본 장치).
# 미탐색 상태와 "없음"을 구분하려고 별도 플래그를 쓴다.
_device_cache: Optional[tuple[int, int, int]] = None
_device_searched = False


def target_peak_dbfs() -> float:
    try:
        return float(os.environ.get("VICA_TTS_PEAK_DBFS", DEFAULT_PEAK_DBFS))
    except ValueError:
        return DEFAULT_PEAK_DBFS


# ---------------------------------------------------------------- 순수 변환
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
    """재생 직전 변환 파이프라인. 정규화는 리샘플 뒤에 한다 —
    리샘플이 최고점을 미세하게 바꿀 수 있기 때문이다."""
    out = np.asarray(wave, dtype=np.float32)
    if device_rate is not None:
        out = resample(out, rate, device_rate)
    out = normalize_peak(out)
    if device_channels >= 2:
        out = to_stereo(out)
    return out


# ---------------------------------------------------------------- 장치 탐색
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
        return None  # 지정했는데 못 찾음 → 기본 장치 (경고는 호출자 로그 몫)

    for i, d in enumerate(devices):
        if "respeaker" in d["name"].lower() and d["max_output_channels"] >= 1:
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
    global _device_cache, _device_searched
    _device_cache = None
    _device_searched = False


# ---------------------------------------------------------------- 재생
def play(wave: np.ndarray, rate: int, blocking: bool = False) -> None:
    """정규화·리샘플 후 재생한다. 실패 처리는 호출자 몫(예외 그대로 전파) —
    TTS 는 원인을 로그에 남기고, 효과음은 조용히 넘어가는 식으로 서로 다르다.

    blocking=True(TTS)는 조각 단위로 쓰며 조각 사이에 stop() 깃발을 본다 —
    다른 스레드가 언제든 0.1초 안에 끊을 수 있다 (긴급 선점·barge-in).
    blocking=False(효과음, 0.2초 내외)는 짧아서 끊을 일이 없다.
    """
    import sounddevice as sd

    wave = np.asarray(wave, dtype=np.float32)
    if wave.size == 0:
        return

    device = output_device()
    if device is None:
        index, out_rate = None, int(rate)
        out = normalize_peak(wave)
    else:
        index, out_rate, channels = device
        out = prepare(wave, rate, out_rate, channels)

    if not blocking:
        sd.play(out, out_rate, device=index)
        return

    if out.ndim == 1:
        out = out.reshape(-1, 1)
    _stop_flag.clear()
    with sd.OutputStream(samplerate=out_rate, device=index,
                         channels=out.shape[1], dtype="float32") as stream:
        for i in range(0, len(out), CHUNK):
            if _stop_flag.is_set():
                break
            stream.write(np.ascontiguousarray(out[i:i + CHUNK]))


def stop() -> None:
    """재생 중인 소리를 즉시 끊는다 (긴급 선점·barge-in 용). 스레드 안전."""
    _stop_flag.set()
    try:
        import sounddevice as sd

        sd.stop()   # 비차단(sd.play) 효과음 쪽도 끊는다
    except Exception:
        pass

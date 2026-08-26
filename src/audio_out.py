"""로봇 소리의 단일 출구 — reSpeaker 재생 경로 (AEC 참조 신호).

XVF-3000 의 AEC 는 "자기가 재생한 소리"만 참조로 알고 마이크 입력에서 뺀다.
다른 장치(HDMI 등)로 나간 소리는 빼야 할 목록에 없어서 마이크에 그대로
되돌아온다. 그래서 TTS·효과음·인사 등 로봇이 내는 모든 소리는 이 모듈을
지나 reSpeaker 로 나가야 한다 (vica-wakeword/docs/integration-design.md 8절 D6).

- 재생 경로는 장치 기본 샘플레이트(16kHz) 전용이라 리샘플한다.
- 최고점(peak)을 -3dBFS 로 정규화한다. reSpeaker 재생부에는 ALSA 볼륨
  조절이 없어(믹서 컨트롤 0개) 이 정규화가 유일한 볼륨 수단이다. 클리핑은
  AEC 성능을 떨어뜨리므로 0dBFS 까지 올리지 않는다.
- reSpeaker 는 항상 있다 — 없으면 폴백 없이 실패한다 (다른 장치로 소리만
  나가고 AEC 가 조용히 깨지는 것이 더 위험하다).

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


# ---------------------------------------------------------------- ALSA 잡음기
#
# ALSA C 라이브러리는 파이썬 로깅을 거치지 않고 stderr 에 직접 뿌린다
# ("ALSA lib pcm.c: underrun occurred" 등). pulse 경유 재생은 조각 쓰기
# 사이마다 이 메시지가 나와 터미널을 덮는다(2026-08-26 실기). 재생 자체에는
# 무해한 알림이라 핸들러를 비워 끈다 — 진짜 재생 실패는 예외로 따로 잡힌다.
_alsa_handler_ref = None  # ctypes 콜백이 GC 되면 세그폴트 — 참조를 붙잡아 둔다


def _silence_alsa_errors() -> None:
    global _alsa_handler_ref
    try:
        from ctypes import CDLL, CFUNCTYPE, c_char_p, c_int

        handler_type = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
        _alsa_handler_ref = handler_type(lambda *_args: None)
        CDLL("libasound.so.2").snd_lib_error_set_handler(_alsa_handler_ref)
    except Exception:
        pass  # 못 꺼도 동작에는 지장 없다 — 시끄러울 뿐


_silence_alsa_errors()


# ---------------------------------------------------------------- 장치 탐색
def _ensure_respeaker_route() -> bool:
    """pulse 기본 출력을 reSpeaker 로 강제하고 음소거를 푼다. 성공 여부 반환.

    이 로봇의 소리 출구는 reSpeaker 하나뿐이다 (2026-08-26 사용자 확정).
    재부팅·재장착마다 하던 수동 pactl 설정과 음소거 사고(2026-08-26 실기:
    장치 재장착 후 음소거로 인식돼 TTS 전체가 무음)를 노드가 스스로 치운다.
    """
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
        return None  # 지정했는데 못 찾음 → 기본 장치 (경고는 호출자 로그 몫)

    # 출구는 무조건 reSpeaker 다. 먼저 pulse 배선을 reSpeaker 로 강제해 두고
    # (기본 출력 지정 + 음소거 해제), 직통 장치가 보이면 직통을 쓴다.
    routed = _ensure_respeaker_route()

    for i, d in enumerate(devices):
        if "respeaker" in d["name"].lower() and d["max_output_channels"] >= 1:
            return i, int(d["default_samplerate"]), int(d["max_output_channels"])

    # 직통이 목록에서 사라지는 경우: 마이크 감시가 카드를 잡고 있으면 PortAudio
    # 목록에서 카드가 통째로 빠진다 (2026-08-12 실측, 2026-08-26 로봇 재현).
    # 위에서 기본 출력을 reSpeaker 로 강제했으므로 pulse 경유 = 같은 스피커다.
    if routed:
        for i, d in enumerate(devices):
            if d["name"].strip().lower() == "pulse" and d["max_output_channels"] >= 1:
                return i, int(d["default_samplerate"]), int(d["max_output_channels"])

    # reSpeaker 가 시스템에 아예 없다 — 다른 스피커로 새지 않는다 (AEC 참조가
    # 깨진 채 소리만 나는 것이 가장 위험하다). 호출자가 큰 소리로 실패한다.
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


# 블로킹 재생용 상시 출력 스트림. 발화마다 스트림을 여닫으면 열기 순간의
# USB 제어 트래픽이 칩 상태 폴링(barge-in VAD 조회)과 충돌해 장치가 열기를
# 거부할 수 있다 (2026-08-24 실측: Device unavailable). 한 번 열어 유지한다.
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
        # 다른 장치로 조용히 폴백하지 않는다 — 소리는 나는데 AEC 참조가
        # 깨진 채 도는 것이 가장 위험하다 (정책: docs/respeaker-v3-capabilities.md).
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

        sd.stop()   # 비차단(sd.play) 효과음 쪽도 끊는다
    except Exception:
        pass

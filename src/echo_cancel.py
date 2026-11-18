"""소프트웨어 에코 제거 — speexdsp(ctypes) 감싸개와 기준 신호 지연.

스피커 소리가 마이크에 닿기까지 약 118 ms 걸려, 기준(ch5)을 100 ms 늦춘 뒤
256 ms 필터에 넣는다(설계 4.1). 새 의존성 없이 설치된 libspeexdsp.so.1 만 쓴다.
"""
from __future__ import annotations

import ctypes

import numpy as np

BLOCK = 1280          # wakeword_monitor.FRAME 과 같다
SPEEX_FRAME = 160
_SET_SAMPLING_RATE = 24   # SPEEX_ECHO_SET_SAMPLING_RATE


class EchoCancelUnavailable(RuntimeError):
    """라이브러리를 못 열었거나 상태 초기화에 실패했다."""


class EchoCanceller:
    def __init__(self, delay_ms: int = 100, tail_ms: int = 256, rate: int = 16000,
                 lib_name: str = "libspeexdsp.so.1") -> None:
        self._st = None
        try:
            lib = ctypes.CDLL(lib_name)
            p16 = ctypes.POINTER(ctypes.c_int16)
            lib.speex_echo_state_init.restype = ctypes.c_void_p
            lib.speex_echo_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
            lib.speex_echo_ctl.restype = ctypes.c_int
            lib.speex_echo_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
            lib.speex_echo_cancellation.restype = None
            lib.speex_echo_cancellation.argtypes = [ctypes.c_void_p, p16, p16, p16]
            lib.speex_echo_state_destroy.restype = None
            lib.speex_echo_state_destroy.argtypes = [ctypes.c_void_p]
        except (OSError, AttributeError) as e:
            raise EchoCancelUnavailable(f"speexdsp 를 열지 못했다: {e}") from e
        st = lib.speex_echo_state_init(SPEEX_FRAME, tail_ms * rate // 1000)
        if not st:
            raise EchoCancelUnavailable("speex_echo_state_init 실패")
        self._lib = lib
        self._st = st
        self._p16 = p16
        lib.speex_echo_ctl(st, _SET_SAMPLING_RATE, ctypes.byref(ctypes.c_int(rate)))
        self._delay = np.zeros(delay_ms * rate // 1000, dtype=np.int16)

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """mic·ref 는 같은 길이의 int16 1차원(160 의 배수). 에코를 뺀 mic 를 돌려준다."""
        if self._st is None:
            raise RuntimeError("이미 닫힌 EchoCanceller")
        mic = np.ascontiguousarray(mic, dtype=np.int16)
        ref = np.ascontiguousarray(ref, dtype=np.int16)
        if mic.ndim != 1 or ref.ndim != 1 or len(mic) != len(ref):
            raise ValueError("mic·ref 는 같은 길이의 1차원이어야 한다")
        if len(mic) % SPEEX_FRAME:
            raise ValueError(f"길이는 {SPEEX_FRAME} 의 배수여야 한다")
        joined = np.concatenate([self._delay, ref])
        play = np.ascontiguousarray(joined[:len(ref)])
        self._delay = joined[len(ref):].copy()
        out = np.zeros_like(mic)
        for i in range(0, len(mic), SPEEX_FRAME):
            s = slice(i, i + SPEEX_FRAME)
            self._lib.speex_echo_cancellation(
                self._st,
                mic[s].ctypes.data_as(self._p16),
                play[s].ctypes.data_as(self._p16),
                out[s].ctypes.data_as(self._p16))
        return out

    def close(self) -> None:
        st, self._st = self._st, None
        if st:
            self._lib.speex_echo_state_destroy(st)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

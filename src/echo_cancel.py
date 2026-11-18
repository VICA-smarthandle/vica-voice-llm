"""소프트웨어 에코 제거 — speexdsp(ctypes) 감싸개와 기준 신호 지연.

스피커 소리가 마이크에 닿기까지 약 118 ms 걸려, 기준(ch5)을 100 ms 늦춘 뒤
256 ms 필터에 넣는다(설계 4.1). 새 의존성 없이 설치된 libspeexdsp.so.1 만 쓴다.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np

from .echo_meter import AEC_LOG_DIR, EchoMeter

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


_MODES = ("off", "log", "on")


@dataclass(frozen=True)
class EchoConfig:
    mode: str = "off"
    delay_ms: int = 100
    tail_ms: int = 256
    warnings: tuple = ()


def _int_in_range(env: Mapping[str, str], key: str, default: int, lo: int, hi: int, warnings: list) -> int:
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(str(raw).strip())
    except ValueError:
        value = None
    if value is None or not lo <= value <= hi:
        warnings.append(f"{key}={raw!r} 를 못 읽음 — 기본값 {default} 로 돈다")
        return default
    return value


def echo_config_from_env(env: Mapping[str, str]) -> EchoConfig:
    """VICA_SW_AEC 계열 .env 키를 읽는다. 잘못된 값은 기본값 + 경고."""
    warnings: list = []
    raw = env.get("VICA_SW_AEC")
    mode = "off" if raw is None else str(raw).strip().lower()
    if mode == "":
        mode = "off"
    elif mode not in _MODES:
        warnings.append(f"VICA_SW_AEC={raw!r} 는 off|log|on 이 아님 — off 로 돈다")
        mode = "off"
    delay_ms = _int_in_range(env, "VICA_SW_AEC_DELAY_MS", 100, 0, 400, warnings)
    tail_ms = _int_in_range(env, "VICA_SW_AEC_TAIL_MS", 256, 32, 512, warnings)
    return EchoConfig(mode, delay_ms, tail_ms, tuple(warnings))


class EchoStage:
    """마이크 루프 단계. 어떤 실패도 원래 ch0 를 막지 않는다(설계 4.3)."""

    def __init__(self, config: EchoConfig, warn: Callable[[str], None],
                 meter: Optional["EchoMeter"] = None,
                 factory: Callable[..., EchoCanceller] = EchoCanceller) -> None:
        self._config = config
        self._warn = warn
        self._meter = meter
        self._factory = factory
        self._mode = config.mode
        self._ec = None
        if self._mode == "off":
            return
        try:
            self._ec = self._build()
        except Exception as exc:
            self._die(exc)

    @property
    def mode(self) -> str:
        return self._mode

    def _build(self):
        return self._factory(delay_ms=self._config.delay_ms, tail_ms=self._config.tail_ms)

    def _die(self, exc: Exception) -> None:
        ec, self._ec = self._ec, None
        self._mode = "off"
        try:
            if ec is not None:
                ec.close()
        except Exception:
            pass
        try:
            self._warn(f"소프트웨어 에코 제거 중단 — 원래 소리로 계속 감시: {exc}")
        except Exception:
            pass

    def feed(self, ch0: np.ndarray, ch5: np.ndarray) -> np.ndarray:
        if self._mode == "off" or self._ec is None:
            return ch0
        try:
            if len(ch0) != BLOCK or len(ch5) != len(ch0):
                old = self._ec
                self._ec = self._build()
                try:
                    old.close()
                except Exception:
                    pass
                return ch0
            out = self._ec.process(ch0, ch5)
            if self._meter is not None:
                self._meter.update(ch0, out, ch5)
        except Exception as exc:
            self._die(exc)
            return ch0
        return out if self._mode == "on" else ch0


def _tell(fn: Callable[[str], None], line: str) -> None:
    """알림 콜백이 예외를 내도 기동을 막지 않는다."""
    try:
        fn(line)
    except Exception:
        pass


def build_echo_stage(env: Mapping[str, str], info: Callable[[str], None], warn: Callable[[str], None],
                     log_dir: Optional[Path] = None, factory=EchoCanceller) -> Optional[EchoStage]:
    """노드 기동 때 단계를 만든다. 예외를 내지 않는다 — 못 만들면 경고 한 번 뒤 None(원래 ch0 로 감시)."""
    try:
        config = echo_config_from_env(env)
        for line in config.warnings:
            _tell(warn, line)
        if config.mode == "off":
            return None
        folder = Path(AEC_LOG_DIR).expanduser() if log_dir is None else Path(log_dir)
        stage = EchoStage(config, warn, meter=EchoMeter(config.mode, folder, warn), factory=factory)
    except Exception as exc:
        _tell(warn, f"소프트웨어 에코 제거를 켤 수 없음 — 원래 소리로 계속 감시: {exc}")
        return None
    if stage.mode == "off":
        return None
    _tell(info, f"소프트웨어 에코 제거: {config.mode} (기준 늦춤 {config.delay_ms} ms · 필터 {config.tail_ms} ms)")
    return stage


def clip_info_with_aec(info: dict, mode: str) -> dict:
    return info if mode == "off" else {**info, "aec": mode}

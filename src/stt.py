"""faster-whisper 기반 한국어 음성 인식(STT).

마이크로 녹음한 음성을 한국어 텍스트로 변환한다.
- record_until_enter(): 엔터를 누를 때까지 마이크로 녹음
- transcribe():         오디오(또는 wav 경로) -> 텍스트
- listen():             녹음 + 변환을 한 번에

입력 방식은 push-to-talk: '엔터로 녹음 시작 -> 말하기 -> 엔터로 종료'.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np
from dotenv import load_dotenv

load_dotenv()  # VICA_STT_* 환경변수 (.env) — 어떤 진입점에서 실행해도 적용되도록 여기서 로드

SAMPLE_RATE = 16000  # whisper 표준 입력 샘플레이트


def _preload_cuda_ctranslate2() -> None:
    """Jetson CUDA용 libctranslate2 를 미리 적재한다 (있을 때만).

    pip 의 ctranslate2 ARM64 휠은 CPU 전용이라 Jetson GPU 빌드를 따로 둔다.
    코어 라이브러리를 .venv/ct2lib 에 두고 RTLD_GLOBAL 로 먼저 올리면
    ctranslate2 _ext 가 LD_LIBRARY_PATH 설정 없이 링크된다 (2026-07-19).
    라이브러리가 없으면 아무것도 하지 않고 pip 기본(CPU) 빌드로 동작한다.
    """
    import ctypes
    import sys

    lib = Path(sys.prefix) / "ct2lib" / "libctranslate2.so.4"
    if lib.exists():
        try:
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass  # CUDA 미탑재 환경(PC 등)에선 CPU 빌드로 폴백


_preload_cuda_ctranslate2()


class VicaSTT:
    """한국어 음성 인식기. 모델은 생성 시 한 번만 로드한다."""

    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        language: str = "ko",
    ):
        from faster_whisper import WhisperModel

        # Jetson 온디바이스 기본은 GPU(cuda/float16). CUDA 초기화가 안 되는 환경에서는
        # CPU(int8) 로 자동 폴백한다. VICA_STT_DEVICE/COMPUTE 로 명시하면 그 값을 쓴다.
        # 대화용 STT 는 medium. 발화 전체를 한 번에 옮기므로 정확도를 우선한다.
        # (상시 긴급어 감시는 0.5초마다 짧은 창을 돌려야 해서 small 을 따로 쓴다 —
        #  emergency_monitor.VICA_EMERGENCY_STT_MODEL 참고.)
        model_size = model_size or os.environ.get("VICA_STT_MODEL", "medium")
        device = device or os.environ.get("VICA_STT_DEVICE", "cuda")
        compute_type = compute_type or os.environ.get("VICA_STT_COMPUTE", "float16")
        try:
            self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as exc:  # CUDA 미가용 등 -> CPU 폴백 (이미 cpu 면 그대로 실패)
            if device == "cpu":
                raise
            print(f"[STT] {device} 로드 실패({exc}) -> CPU(int8) 폴백")
            device, compute_type = "cpu", "int8"
            self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = language

    def transcribe(self, audio: Union[np.ndarray, str, Path]) -> str:
        """오디오(numpy float32 16kHz) 또는 wav 파일 경로 -> 텍스트."""
        source = str(audio) if isinstance(audio, Path) else audio
        # beam_size=1: 짧은 명령 발화엔 greedy 로 충분하고 기본값(5)보다 수 배 빠르다
        # (Jetson 실측 2026-07-19).
        #
        # vad_filter 는 일부러 켜지 않는다. push-to-talk 시절에는 엔터까지의 긴 무음을
        # 잘라내는 효과가 컸지만, 지금 입력은 웨이크워드가 골라낸 2~3초 클립이라
        # 잘라낼 무음이 거의 없고, 작게 말한 긴급어를 무음으로 오판해 지울 위험이 있다.
        segments, _info = self._model.transcribe(
            source, language=self.language, beam_size=1
        )
        return "".join(seg.text for seg in segments).strip()

    def record_until_enter(self) -> np.ndarray:
        """엔터를 누를 때까지 마이크로 녹음해 1차원 float32 파형을 돌려준다."""
        import sounddevice as sd

        frames: list[np.ndarray] = []

        def callback(indata, _frames, _time, _status):
            frames.append(indata.copy())

        print("🎤 녹음 중... 말씀하세요 (끝나면 엔터)")
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback
        ):
            input()  # 엔터를 누를 때까지 녹음 지속

        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(frames, axis=0).flatten()

    def listen(self) -> str:
        """녹음 후 한국어 텍스트로 변환해 돌려준다."""
        audio = self.record_until_enter()
        if audio.size == 0:
            return ""
        return self.transcribe(audio)

"""faster-whisper 기반 한국어 음성 인식(STT)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np
from dotenv import load_dotenv

load_dotenv()

SAMPLE_RATE = 16000


def _preload_cuda_ctranslate2() -> None:
    """Jetson CUDA용 libctranslate2 를 미리 적재한다 (있을 때만)."""
    import ctypes
    import sys

    lib = Path(sys.prefix) / "ct2lib" / "libctranslate2.so.4"
    if lib.exists():
        try:
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


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

        model_size = model_size or os.environ.get("VICA_STT_MODEL", "medium")
        device = device or os.environ.get("VICA_STT_DEVICE", "cuda")
        compute_type = compute_type or os.environ.get("VICA_STT_COMPUTE", "float16")
        try:
            self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as exc:
            if device == "cpu":
                raise
            print(f"[STT] {device} 로드 실패({exc}) -> CPU(int8) 폴백")
            device, compute_type = "cpu", "int8"
            self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = language

    def transcribe(self, audio: Union[np.ndarray, str, Path]) -> str:
        """오디오(numpy float32 16kHz) 또는 wav 파일 경로 -> 텍스트."""
        source = str(audio) if isinstance(audio, Path) else audio
        segments, _info = self._model.transcribe(
            source, language=self.language, beam_size=1,
            temperature=0.0, condition_on_previous_text=False,
        )
        from .stt_guard import accept_segments

        return accept_segments(segments)

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
            input()

        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(frames, axis=0).flatten()

    def record_seconds(self, seconds: float) -> np.ndarray:
        """정해진 시간만큼 녹음해 1차원 float32 파형을 돌려준다."""
        import sounddevice as sd

        frames = int(seconds * SAMPLE_RATE)
        audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        return audio.flatten()

    def listen(self) -> str:
        """녹음 후 한국어 텍스트로 변환해 돌려준다."""
        audio = self.record_until_enter()
        if audio.size == 0:
            return ""
        return self.transcribe(audio)

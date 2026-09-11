"""supertonic 기반 한국어 음성 출력(TTS)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import supertonic
import supertonic.loader

supertonic.loader.DEFAULT_ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


class VicaTTS:
    """한국어 음성 합성기. 모델은 생성 시 한 번만 로드한다."""

    def __init__(self, model: str = "supertonic-3", voice: str = "F2", lang: str = "ko"):
        self._tts = supertonic.TTS(model=model, auto_download=True)
        self._style = self._tts.get_voice_style(voice)
        self.lang = lang

    def _synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """텍스트 -> (1차원 오디오 파형, 샘플레이트)."""
        wav, _duration = self._tts.synthesize(text, self._style, lang=self.lang)
        return np.asarray(wav).squeeze(), int(self._tts.sample_rate)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """합성만 하고 재생하지 않는다 — 합성 캐시(synth_cache)용 공개 API."""
        return self._synthesize(text)

    def save(self, text: str, path: str | Path) -> None:
        """합성 결과를 wav 파일로 저장한다."""
        wav, _ = self._tts.synthesize(text, self._style, lang=self.lang)
        self._tts.save_audio(wav, str(path))

    def speak(self, text: str) -> bool:
        """합성 후 스피커로 재생한다. 오디오 장치가 없으면 False 를 돌려준다."""
        if not text:
            return False
        wav, sample_rate = self._synthesize(text)
        try:
            from . import audio_out

            audio_out.play(wav, sample_rate, blocking=True)
            return True
        except Exception as exc:
            import sys

            print(f"[TTS] 재생 실패: {exc}", file=sys.stderr)
            return False

    def play_audio(self, wav, sample_rate: int) -> bool:
        """이미 만들어 둔 파형을 재생한다 (고정 멘트 캐시용 — 합성 생략)."""
        try:
            from . import audio_out

            audio_out.play(wav, sample_rate, blocking=True)
            return True
        except Exception as exc:
            import sys

            print(f"[TTS] 재생 실패: {exc}", file=sys.stderr)
            return False

    def stop(self) -> None:
        """재생 중인 소리를 즉시 끊는다 (긴급 발화 선점용)."""
        from . import audio_out

        audio_out.stop()

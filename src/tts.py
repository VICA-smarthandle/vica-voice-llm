"""supertonic 기반 한국어 음성 출력(TTS).

VICA 의 reply 텍스트를 음성으로 합성한다. (ONNX 기반이라 빠르고 온디바이스에 적합)
- speak(): 합성 후 스피커로 재생 (오디오 장치가 없으면 조용히 건너뜀)
- save():  합성 결과를 wav 파일로 저장

supertonic-3 내장 목소리: 남성 M1~M5, 여성 F1~F5.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import supertonic
import supertonic.loader

# supertonic 은 provider 가 CPU 로 하드코딩되어 있어 모듈 변수를 바꿔 GPU 를 켠다.
# Jetson 온디바이스 기본은 CUDA. CUDA 가 없으면 supertonic 이 자동으로 CPU 로 폴백한다.
# 합성 3.0초 -> 0.3초 (onnxruntime-gpu 필요, docs/jetson-setup.md 참고)
supertonic.loader.DEFAULT_ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# 재생 음량 목표(최대진폭).
#
# 합성 원본은 최대진폭이 0.3 안팎이라 디지털 여유의 30 % 도 쓰지 않는다. 스피커가
# 고정 게인이면 그 손실이 그대로 체감 음량이 된다 — 2026-08-13 실측에서 음성이
# 안내음(DEFAULT_VOLUME=0.4)보다 11.7 dB 작았다.
#
# 1.0 에 딱 맞추면 반올림으로 클리핑이 나므로 0.95 로 둔다.
PEAK_TARGET = 0.95


def normalize_peak(wav: np.ndarray, target: float = PEAK_TARGET) -> np.ndarray:
    """최대진폭을 target 에 맞춘다.

    무음이거나 정수형이면 그대로 돌려준다 — 정수형은 스케일 규약이 달라
    여기서 건드리지 않는다.
    """
    if wav.size == 0 or not np.issubdtype(wav.dtype, np.floating):
        return wav
    peak = float(np.abs(wav).max())
    if peak <= 0.0:
        return wav
    return (wav * (target / peak)).astype(wav.dtype, copy=False)


class VicaTTS:
    """한국어 음성 합성기. 모델은 생성 시 한 번만 로드한다."""

    def __init__(self, model: str = "supertonic-3", voice: str = "F2", lang: str = "ko"):
        self._tts = supertonic.TTS(model=model, auto_download=True)
        self._style = self._tts.get_voice_style(voice)
        self.lang = lang

    def _synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """텍스트 -> (1차원 오디오 파형, 샘플레이트)."""
        # synthesize 의 두 번째 반환값은 '오디오 길이(초)'다. 샘플레이트가 아니다.
        # 실제 샘플레이트는 모델 속성(self._tts.sample_rate)에 들어 있다.
        wav, _duration = self._tts.synthesize(text, self._style, lang=self.lang)
        return normalize_peak(np.asarray(wav).squeeze()), int(self._tts.sample_rate)

    def save(self, text: str, path: str | Path) -> None:
        """합성 결과를 wav 파일로 저장한다.

        재생과 같은 정규화를 거친다 — 미리 만들어 둔 안내 음성(assets/)이
        실시간 합성보다 작게 나오면 안 된다.
        """
        wav, _ = self._tts.synthesize(text, self._style, lang=self.lang)
        self._tts.save_audio(normalize_peak(np.asarray(wav)), str(path))

    def speak(self, text: str) -> bool:
        """합성 후 스피커로 재생한다. 오디오 장치가 없으면 False 를 돌려준다."""
        if not text:
            return False
        wav, sample_rate = self._synthesize(text)
        try:
            import sounddevice as sd

            sd.play(wav, samplerate=sample_rate)
            sd.wait()
            return True
        except Exception as exc:
            # 헤드리스/오디오 장치 없음 등 -> 텍스트 흐름은 막지 않되, 원인은 알린다.
            import sys

            print(f"[TTS] 재생 실패: {exc}", file=sys.stderr)
            return False

    def stop(self) -> None:
        """재생 중인 소리를 즉시 끊는다 (긴급 발화 선점용).

        다른 스레드에서 불러도 된다. speak() 안의 sd.wait() 가 곧바로 돌아온다.
        """
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass  # 오디오 장치가 없으면 끊을 것도 없다

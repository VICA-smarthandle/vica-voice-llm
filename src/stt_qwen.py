"""Qwen3-ASR(transcribe.cpp) 받아쓰기 — 귀의 whisper 자리(int16 소리 → 글자)에 그대로 끼운다.

2026-10-15 사용자 결정: 개인정보(목소리 국외 전송)를 피하려고 귀의 받아쓰기를 로컬 Qwen3-ASR 1.7B 로 바꿔 본다.
대화 창(1단계)과 '멈춰' 확인(3단계)을 역할별 `.env` 스위치로 고른다 — 기본은 whisper.
설계: docs/superpowers/specs/2026-10-15-stt-qwen-asr-design.md

문지기(오늘 실측의 핵심): Qwen 은 말이 없으면 'language None'(빈 글자)을 내도록 학습됐는데, 한국어로 고정하면
transcribe.cpp 가 답 머리를 'language Korean' 으로 미리 채워 그 말을 못 한다 — 그래서 침묵에 '네.'·'응.' 을 지어냈다.
언어를 비우면 침묵은 빈칸이지만 짧은 한국어를 중국어·태국어로 착각한다. 그래서 1차는 언어를 비우고, 빈칸이면
말 없음, 한국어면 그대로, 다른 언어면 2차로 한국어 고정을 한 번 더 한다(정답 216개: turbo 160 · 1.7B 문지기 174,
침묵 7/7 빈칸).

이 모듈은 transcribe_cpp 를 쓸 때만 부른다(지연 import) — 라이브러리가 없는 PC 에서도 시험이 돈다.
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

RATE = 16000
ENGINES = ("whisper", "qwen")
_PUNCT_ONLY = re.compile(r"[\s.,!?~…'\"·\-。、，？！]*")


class QwenUnavailable(RuntimeError):
    """Qwen 을 쓸 수 없다 — 설정·라이브러리·모델 적재 실패, 또는 엔진 오류가 잇따름.

    귀는 whisper 로 되돌아가지 않고 멈춘다(10-15 사용자 결정: 바꾼 줄 모르고 옛 귀로 시험하는 일이 없게).
    """


def ear_stt_engines() -> tuple[str, str]:
    """귀의 두 받아쓰기 엔진 — (대화 창, 긴급 확인). 기본 whisper.

    모르는 값은 오류다 — 잘못 적은 설정으로 몰래 whisper 가 돌면 시험 결과를 엉뚱한 엔진에 붙이게 된다.
    """
    out = []
    for key in ("VICA_STT_ENGINE", "VICA_VERIFY_STT_ENGINE"):
        value = (os.environ.get(key) or "whisper").strip().lower()
        if value not in ENGINES:
            raise QwenUnavailable(f"{key}={value!r} — whisper 또는 qwen 만 됩니다")
        out.append(value)
    return out[0], out[1]


def _deadline(audio_sec: float) -> float:
    """한 번 실행의 시한(초). 1.7B 는 16 s 말에 2.6 s 걸렸다 — 넉넉히 잡되 귀가 오래 묶이지 않게."""
    return 2.0 + 0.6 * audio_sec


def load_binding(lib_path: Optional[str] = None, py_path: Optional[str] = None):
    """transcribe_cpp 를 부른다. 경로는 .env 로 받는다 — 개인 경로를 코드에 박지 않는다(AGENTS.md)."""
    if lib_path:
        os.environ["TRANSCRIBE_LIBRARY"] = lib_path
    if py_path and py_path not in sys.path:
        sys.path.insert(0, py_path)
    try:
        import transcribe_cpp
    except Exception as exc:  # noqa: BLE001 — ImportError·라이브러리 ABI 오류 모두 '못 씀'
        raise QwenUnavailable(f"transcribe_cpp 를 불러오지 못함: {exc} "
                              f"(VICA_TRANSCRIBE_LIB={lib_path!r}, VICA_TRANSCRIBE_PY={py_path!r})") from exc
    return transcribe_cpp


class QwenTranscriber:
    """Qwen3-ASR 모델 하나 — 대화 창과 긴급 확인이 같이 쓴다(감시 루프가 스레드 하나라 겹쳐 부를 일이 없다)."""

    N_CTX = 1024                 # 청취 상한 15 s ≈ 오디오 토큰 190 + 지시·답. KV 메모리를 묶어 둔다
    FATAL_ERRORS_IN_ROW = 5      # 엔진 오류가 이만큼 잇따르면 귀를 멈춘다

    def __init__(self, model_path: str, *, threads: int = 4, binding=None,
                 lib_path: Optional[str] = None, py_path: Optional[str] = None,
                 log: Callable[[str], None] = print,
                 deadline: Callable[[float], float] = _deadline):
        self._log = log
        self._deadline = deadline
        self._tc = binding if binding is not None else load_binding(lib_path, py_path)
        self.name = Path(model_path).stem
        try:
            self._model = self._tc.Model(str(model_path))
            self._session = self._model.session(n_threads=threads, n_ctx=self.N_CTX)
        except Exception as exc:  # noqa: BLE001 — 파일 없음·적재 오류·장치 없음 모두
            raise QwenUnavailable(f"Qwen 모델을 올리지 못함 ({model_path}): {exc}") from exc
        self.errors_in_row = 0
        self.last: dict = {}
        # 예열 — 첫 '멈춰' 확인이 초기화 비용을 내지 않게. 결과는 버린다.
        self.listen(np.zeros(RATE, dtype=np.int16))

    # ---------------------------------------------------------------- 실행 한 번
    def _run(self, pcm: np.ndarray, language: Optional[str]) -> tuple[str, str]:
        """(글자, 언어). 내용 문제(반복·잘림·취소)는 빈칸, 엔진 오류는 빈칸 + 오류 셈."""
        timer = threading.Timer(self._deadline(len(pcm) / RATE), self._session.cancel)
        timer.daemon = True
        timer.start()
        try:
            res = self._session.run(pcm, language=language)
        except (self._tc.OutputTruncated, self._tc.Aborted) as exc:
            self._log(f"[Qwen] {type(exc).__name__} — 빈칸으로 처리")
            self.errors_in_row = 0
            return "", ""
        except Exception as exc:  # noqa: BLE001 — 어떤 오류도 귀 감시 스레드를 죽이지 않는다
            self.errors_in_row += 1
            self._log(f"[Qwen] 오류 {self.errors_in_row}/{self.FATAL_ERRORS_IN_ROW}: {type(exc).__name__}: {exc}")
            if self.errors_in_row >= self.FATAL_ERRORS_IN_ROW:
                raise QwenUnavailable(f"Qwen 오류가 {self.errors_in_row}번 잇따름 — 귀를 멈춘다") from exc
            return "", ""
        finally:
            timer.cancel()
        self.errors_in_row = 0
        return (res.text or "").strip(), (res.language or "")

    @staticmethod
    def _pcm(audio: np.ndarray) -> Optional[np.ndarray]:
        if audio is None or len(audio) == 0:
            return None
        return np.ascontiguousarray(np.asarray(audio).reshape(-1), dtype=np.float32) / 32768.0

    @staticmethod
    def _clean(text: str) -> str:
        return "" if _PUNCT_ONLY.fullmatch(text or "") else text.strip()

    # ---------------------------------------------------------------- 역할
    def listen(self, audio: np.ndarray) -> str:
        """대화·질문 창 — 문지기를 거친 글자. 말이 없으면 빈칸."""
        pcm = self._pcm(audio)
        if pcm is None:
            return ""
        t0 = time.monotonic()
        text, lang = self._run(pcm, None)
        second = False
        if self._clean(text) and lang != "ko":
            text, _ = self._run(pcm, "ko")
            second = True
        self.last = {"lang": lang or "none", "second_pass": second, "sec": time.monotonic() - t0}
        return self._clean(text)

    def verify(self, audio: np.ndarray, is_emergency: Callable[[str], bool]) -> str:
        """'멈춰' 확인 — 한국어 고정 한 번(빠르게). 긴급어가 아니면 '말 없음' 인지 한 번 더 본다.

        감시기는 긴급어가 아닌 확인 전사를 질문 답으로 넘길 수 있다(wakeword_monitor 의 followup_answer).
        한국어 고정은 잡음에도 '네.' 를 적으므로, 말소리가 없었으면 빈칸을 돌려줘 그 길을 막는다.
        """
        pcm = self._pcm(audio)
        if pcm is None:
            return ""
        t0 = time.monotonic()
        text = self._clean(self._run(pcm, "ko")[0])
        checked = False
        if text and not is_emergency(text):
            checked = True
            if not self._clean(self._run(pcm, None)[0]):
                text = ""
        self.last = {"lang": "ko", "second_pass": checked, "sec": time.monotonic() - t0}
        return text


def qwen_from_env(log: Callable[[str], None] = print) -> QwenTranscriber:
    """`.env` 값으로 Qwen 을 올린다. 모델 경로가 없으면 QwenUnavailable."""
    def _path(key: str) -> Optional[str]:
        value = (os.environ.get(key) or "").strip()
        return os.path.expanduser(value) if value else None

    model = _path("VICA_QWEN_MODEL")
    if not model:
        raise QwenUnavailable("VICA_QWEN_MODEL(gguf 경로)이 비어 있습니다")
    try:
        threads = int(os.environ.get("VICA_QWEN_THREADS") or 4)
    except ValueError:
        threads = 4
    return QwenTranscriber(model, threads=threads, lib_path=_path("VICA_TRANSCRIBE_LIB"),
                           py_path=_path("VICA_TRANSCRIBE_PY"), log=log)

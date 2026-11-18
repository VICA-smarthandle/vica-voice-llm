"""Qwen3-ASR 받아쓰기 감싸개(src/stt_qwen.py) — 가짜 transcribe_cpp 로 문지기·긴급 확인·오류 흡수·실패 정책을 본다.

2026-10-15 사용자 결정: 귀 whisper 를 Qwen3-ASR 1.7B 로 바꿔 시험, 적재에 실패하면 whisper 로 되돌아가지 않는다.
설계: docs/superpowers/specs/2026-10-15-stt-qwen-asr-design.md
"""
import time
import types

import numpy as np
import pytest

from src.stt_qwen import QwenTranscriber, QwenUnavailable, ear_stt_engines, qwen_from_env


def _binding(script, load_error=None):
    """script(pcm, language) -> (text, lang) 또는 예외. 호출 기록은 b.calls 에 남는다."""
    b = types.SimpleNamespace(calls=[], cancels=[])

    class TranscribeError(RuntimeError):
        pass

    class OutputTruncated(TranscribeError):
        pass

    class OutputRepetition(OutputTruncated):
        pass

    class Aborted(TranscribeError):
        pass

    class Session:
        def run(self, pcm, language=None):
            assert pcm.dtype == np.float32 and pcm.ndim == 1
            b.calls.append(language)
            text, lang = script(pcm, language)
            return types.SimpleNamespace(text=text, language=lang)

        def cancel(self):
            b.cancels.append(True)

    class Model:
        def __init__(self, path):
            if load_error is not None:
                raise load_error(f"못 읽음: {path}")
            self.path = path

        def session(self, n_threads=0, n_ctx=0):
            b.session_args = (n_threads, n_ctx)
            return Session()

    b.TranscribeError, b.OutputTruncated, b.OutputRepetition, b.Aborted = (
        TranscribeError, OutputTruncated, OutputRepetition, Aborted)
    b.Model = Model
    return b


def _make(script, **kw):
    b = _binding(script)
    q = QwenTranscriber("/m/Qwen3-ASR-1.7B-Q8_0.gguf", binding=b, log=lambda *a: None, **kw)
    b.calls.clear()                    # 예열 호출은 빼고 센다
    return q, b


SPEECH = np.full(16000, 3000, dtype=np.int16)


def _is_stop(text):
    return "멈춰" in text


# ---------------------------------------------------------------- 대화 창 — 문지기
def test_no_speech_in_auto_pass_gives_empty_without_second_pass():
    q, b = _make(lambda pcm, lang: ("", ""))
    assert q.listen(SPEECH) == ""
    assert b.calls == [None]


def test_korean_in_auto_pass_is_used_directly():
    q, b = _make(lambda pcm, lang: ("그래.", "ko"))
    assert q.listen(SPEECH) == "그래."
    assert b.calls == [None]


def test_other_language_reruns_forced_korean():
    q, b = _make(lambda pcm, lang: ("嗯。", "zh") if lang is None else ("응.", "ko"))
    assert q.listen(SPEECH) == "응."
    assert b.calls == [None, "ko"]


def test_punctuation_only_is_empty():
    q, _ = _make(lambda pcm, lang: (".", "zh") if lang is None else (".", "ko"))
    assert q.listen(SPEECH) == ""


def test_empty_audio_never_reaches_the_model():
    q, b = _make(lambda pcm, lang: ("네.", "ko"))
    assert q.listen(np.zeros(0, dtype=np.int16)) == ""
    assert b.calls == []


# ---------------------------------------------------------------- 긴급 확인
def test_verify_stop_word_takes_one_forced_korean_pass():
    q, b = _make(lambda pcm, lang: ("멈춰.", "ko"))
    assert q.verify(SPEECH, _is_stop) == "멈춰."
    assert b.calls == ["ko"]


def test_verify_ghost_on_no_speech_becomes_empty():
    """한국어 고정은 잡음에도 '네.' 를 적는다 — 긴급어가 아니면 '말 없음' 인지 한 번 더 본다."""
    q, b = _make(lambda pcm, lang: ("네.", "ko") if lang == "ko" else ("", ""))
    assert q.verify(SPEECH, _is_stop) == ""
    assert b.calls == ["ko", None]


def test_verify_real_non_stop_speech_survives():
    q, _ = _make(lambda pcm, lang: ("그래 가자.", "ko"))
    assert q.verify(SPEECH, _is_stop) == "그래 가자."


# ---------------------------------------------------------------- 오류 흡수·실패
def test_repetition_is_swallowed_as_empty():
    """반복·잘림은 엔진 고장이 아니라 내용 문제 — 빈칸으로 돌려주고 오류로 세지 않는다."""
    b = _binding(lambda pcm, lang: ("", ""))
    q = QwenTranscriber("/m/x.gguf", binding=b, log=lambda *a: None)

    def rep(pcm, language=None):
        raise b.OutputRepetition("반복")
    q._session.run = rep
    assert q.listen(SPEECH) == ""
    assert q.errors_in_row == 0


def test_engine_errors_are_swallowed_until_too_many_in_a_row():
    b = _binding(lambda pcm, lang: ("", ""))
    q = QwenTranscriber("/m/x.gguf", binding=b, log=lambda *a: None)

    def boom(pcm, language=None):
        raise b.TranscribeError("백엔드 오류")
    q._session.run = boom
    for _ in range(QwenTranscriber.FATAL_ERRORS_IN_ROW - 1):
        assert q.listen(SPEECH) == ""
    with pytest.raises(QwenUnavailable):
        q.listen(SPEECH)


def test_success_resets_the_error_count():
    b = _binding(lambda pcm, lang: ("네.", "ko"))
    q = QwenTranscriber("/m/x.gguf", binding=b, log=lambda *a: None)
    real = q._session.run

    def flaky(pcm, language=None):
        flaky.n += 1
        if flaky.n % 2:
            raise b.TranscribeError("가끔")
        return real(pcm, language=language)
    flaky.n = 0
    q._session.run = flaky
    for _ in range(QwenTranscriber.FATAL_ERRORS_IN_ROW * 2):
        q.listen(SPEECH)
    assert q.errors_in_row <= 1


def test_load_failure_raises_unavailable():
    b = _binding(lambda pcm, lang: ("", ""), load_error=FileNotFoundError)
    with pytest.raises(QwenUnavailable):
        QwenTranscriber("/m/없음.gguf", binding=b, log=lambda *a: None)


def test_warm_up_runs_once_and_session_is_bounded():
    b = _binding(lambda pcm, lang: ("", ""))
    QwenTranscriber("/m/x.gguf", binding=b, threads=3, log=lambda *a: None)
    assert len(b.calls) >= 1                       # 예열
    assert b.session_args == (3, QwenTranscriber.N_CTX)


def test_slow_run_is_cancelled_at_the_deadline():
    b = _binding(lambda pcm, lang: ("", ""))
    q = QwenTranscriber("/m/x.gguf", binding=b, log=lambda *a: None, deadline=lambda sec: 0.05)

    def slow(pcm, language=None):
        time.sleep(0.2)
        if b.cancels:
            raise b.Aborted("취소됨")
        return types.SimpleNamespace(text="늦은 글자", language="ko")
    q._session.run = slow
    assert q.listen(SPEECH) == ""
    assert b.cancels


# ---------------------------------------------------------------- 설정
def test_engines_default_to_whisper(monkeypatch):
    monkeypatch.delenv("VICA_STT_ENGINE", raising=False)
    monkeypatch.delenv("VICA_VERIFY_STT_ENGINE", raising=False)
    assert ear_stt_engines() == ("whisper", "whisper")


def test_engines_read_each_role(monkeypatch):
    monkeypatch.setenv("VICA_STT_ENGINE", "qwen")
    monkeypatch.setenv("VICA_VERIFY_STT_ENGINE", "Whisper")
    assert ear_stt_engines() == ("qwen", "whisper")


def test_unknown_engine_is_an_error_not_a_silent_whisper(monkeypatch):
    monkeypatch.setenv("VICA_STT_ENGINE", "qwem")
    with pytest.raises(QwenUnavailable):
        ear_stt_engines()


def test_qwen_needs_a_model_path(monkeypatch):
    monkeypatch.delenv("VICA_QWEN_MODEL", raising=False)
    with pytest.raises(QwenUnavailable):
        qwen_from_env(log=lambda *a: None)


def test_paths_from_env_expand_home(monkeypatch):
    """.env 예시처럼 ~ 로 적어도 된다."""
    import src.stt_qwen as sq
    seen = {}

    class Spy:
        def __init__(self, model, **kw):
            seen["model"], seen["kw"] = model, kw
    monkeypatch.setattr(sq, "QwenTranscriber", Spy)
    monkeypatch.setenv("VICA_QWEN_MODEL", "~/m/q.gguf")
    monkeypatch.setenv("VICA_TRANSCRIBE_LIB", "~/lib/libtranscribe.so")
    monkeypatch.delenv("VICA_TRANSCRIBE_PY", raising=False)
    monkeypatch.setenv("VICA_QWEN_THREADS", "2")
    sq.qwen_from_env(log=lambda *a: None)
    assert not seen["model"].startswith("~")
    assert not seen["kw"]["lib_path"].startswith("~")
    assert seen["kw"]["py_path"] is None and seen["kw"]["threads"] == 2

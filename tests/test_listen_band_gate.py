"""말소리 대역 관문 — 칩이 말이라고 해도 400~3000 Hz 가 조용하면 침묵으로 친다(2026-10-21 실기).

10-21 11:45 "네" 한마디 뒤 바람 소리(150~400 Hz)를 칩 말소리 감지가 4.7초 동안 말로 봐 창이 안 닫혔다.
저장 소리 54개 재생: 꼬리 소리의 대역 크기는 95 %가 0.0035 아래, 실제 말은 99 %가 0.0058 위.
귀 모듈의 기본은 끔(None)이고 노드가 VICA_LISTEN_BAND_MIN(기본 0.004)으로 켠다.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.wakeword_monitor import (
    LISTEN_BAND_MIN_DEFAULT,
    WakewordMonitor,
    listen_band_min_from_env,
    speech_band_rms,
)

N = 1280
T = np.arange(N) / 16000.0


def tone(freq: float, amp: float) -> np.ndarray:
    return (np.sin(2 * np.pi * freq * T) * amp * 32767).astype(np.int16)


SPEECH = tone(1000, 0.1)      # 말소리 대역 — 대역 크기 약 0.07
WEAK = tone(1000, 0.008)      # 약한 어미 — 대역 크기 약 0.0057
WIND = tone(220, 0.03)        # 바람·기계음 — 대역 밖
QUIET = np.zeros(N, dtype=np.int16)


class Fake:
    def __init__(self, text="네."):
        self.text = text

    def predict(self, _frame):
        return {"a": 0.0, "b": 0.0}

    def transcribe(self, _audio):
        return self.text


def followup(band_min, texts, states):
    m = WakewordMonitor(on_emergency=lambda e: None, on_user_text=texts.append, on_wake=lambda: None,
                        predict=Fake().predict, transcribe=Fake().transcribe,
                        on_listen_state=states.append, listen_band_min=band_min)
    m.arm_followup(now=0.0)
    m.set_muted(False, now=0.0)
    return m


def feed(m, frames, t0, vad=True):
    for i, f in enumerate(frames):
        m.process_frame(f, now=t0 + i * 0.08, vad=vad)
    return t0 + len(frames) * 0.08


class TestBandRms:
    def test_tone_inside_band(self):
        assert speech_band_rms(SPEECH) == pytest.approx(0.1 / np.sqrt(2), rel=0.1)

    def test_tone_below_band_is_near_zero(self):
        assert speech_band_rms(WIND) < 0.002

    def test_silence(self):
        assert speech_band_rms(QUIET) == 0.0


class TestWindowEnd:
    def test_wind_after_answer_closes_the_window(self):
        """칩이 바람을 말로 봐도 0.8초 뒤 창이 닫힌다 — 10-21 "네" 뒤 4.7초."""
        texts, states = [], []
        m = followup(LISTEN_BAND_MIN_DEFAULT, texts, states)
        t = feed(m, [SPEECH] * 6, 0.1)
        feed(m, [WIND] * 12, t)
        assert texts == ["네."]

    def test_without_the_gate_wind_holds_the_window(self):
        texts, states = [], []
        m = followup(None, texts, states)
        t = feed(m, [SPEECH] * 6, 0.1)
        feed(m, [WIND] * 12, t)
        assert texts == []

    def test_weak_ending_above_the_gate_keeps_listening(self):
        """약한 어미(대역 크기 0.0057)는 말로 남는다 — 예전 소리 크기 말끝의 잘림을 되풀이하지 않는다."""
        texts, states = [], []
        m = followup(LISTEN_BAND_MIN_DEFAULT, texts, states)
        t = feed(m, [SPEECH] * 6, 0.1)
        feed(m, [WEAK] * 12, t)
        assert texts == []

    def test_chip_silence_still_ends_speech(self):
        """칩이 침묵이라 하면 대역과 상관없이 침묵이다 — 지금 동작 그대로."""
        texts, states = [], []
        m = followup(LISTEN_BAND_MIN_DEFAULT, texts, states)
        t = feed(m, [SPEECH] * 6, 0.1)
        feed(m, [SPEECH] * 12, t, vad=False)
        assert texts == ["네."]


class TestWindowStart:
    def test_wind_alone_does_not_start_speech(self):
        texts, states = [], []
        m = followup(LISTEN_BAND_MIN_DEFAULT, texts, states)
        feed(m, [WIND] * 30, 0.1)
        assert "speech" not in states and texts == []

    def test_loud_wind_does_not_start_speech_by_loudness(self):
        """칩이 말이 아니라 했거나(False) 판정이 없을 때(None) 소리 크기로 시작을 보강하는 길도 대역을 본다."""
        for vad in (False, None):
            texts, states = [], []
            m = followup(LISTEN_BAND_MIN_DEFAULT, texts, states)
            feed(m, [WIND] * 30, 0.1, vad=vad)
            assert "speech" not in states, vad

    def test_without_the_gate_wind_starts_speech(self):
        texts, states = [], []
        m = followup(None, texts, states)
        feed(m, [WIND] * 30, 0.1)
        assert "speech" in states


def test_band_min_from_env(monkeypatch):
    monkeypatch.delenv("VICA_LISTEN_BAND_MIN", raising=False)
    assert listen_band_min_from_env() == LISTEN_BAND_MIN_DEFAULT
    for raw, want in (("0.006", 0.006), ("off", None), ("0", None), ("abc", LISTEN_BAND_MIN_DEFAULT),
                      ("2", LISTEN_BAND_MIN_DEFAULT)):
        monkeypatch.setenv("VICA_LISTEN_BAND_MIN", raw)
        assert listen_band_min_from_env() == want

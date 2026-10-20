"""수음 최소·짧은 답 최소를 .env·인자로, 실행 중 문턱 바꾸기, 조각 점수 알림(2026-10-12 마이크 점검기)."""
from __future__ import annotations

import numpy as np
import pytest

from src.wakeword_monitor import (LISTEN_MIN_RMS, SHORT_ANSWER_MIN_RMS, WakewordMonitor,
                                  listen_thresholds_from_env)
from tests.test_wakeword_monitor import LOUD, QUIET, Fake, run_frames

VOICE = np.full(1280, 420, dtype=np.int16)


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("VICA_LISTEN_MIN_RMS", raising=False)
    monkeypatch.delenv("VICA_SHORT_ANSWER_MIN_RMS", raising=False)


def make(fake=None, texts=None, **kw):
    fake = fake or Fake(scores=[])
    return WakewordMonitor(
        on_emergency=lambda e: None,
        on_user_text=(texts.append if texts is not None else (lambda t: None)),
        predict=fake.predict, transcribe=fake.transcribe,
        transcribe_confirm=lambda a: "네", **kw)


def answer(m, voiced, frame):
    """노드처럼 질문 예약 → 말 끝으로 질문 창을 열고, 말 조각 뒤 조용한 조각을 흘린다."""
    m.arm_followup(now=0.0)
    m.set_speaking(False, now=0.0)
    out = run_frames(m, voiced, frame, t0=0.08, vad=True)
    out += run_frames(m, 20, QUIET, t0=0.08 * (voiced + 1), vad=False)
    return out


def test_defaults_without_env():
    assert listen_thresholds_from_env() == (LISTEN_MIN_RMS, SHORT_ANSWER_MIN_RMS) == (0.008, 0.02)
    m = make()
    assert (m.listen_min_rms, m.short_answer_min_rms) == (0.008, 0.02)


def test_env_values(monkeypatch):
    monkeypatch.setenv("VICA_LISTEN_MIN_RMS", "0.012")
    monkeypatch.setenv("VICA_SHORT_ANSWER_MIN_RMS", "0.018")
    m = make()
    assert (m.listen_min_rms, m.short_answer_min_rms) == (0.012, 0.018)


@pytest.mark.parametrize("raw", ["", "abc", "0", "-0.01", "1", "1.5"])
def test_bad_env_falls_back(monkeypatch, raw):
    monkeypatch.setenv("VICA_LISTEN_MIN_RMS", raw)
    monkeypatch.setenv("VICA_SHORT_ANSWER_MIN_RMS", raw)
    assert listen_thresholds_from_env() == (0.008, 0.02)


def test_args_beat_env(monkeypatch):
    monkeypatch.setenv("VICA_LISTEN_MIN_RMS", "0.012")
    m = make(listen_min_rms=0.006, short_answer_min_rms=0.015)
    assert (m.listen_min_rms, m.short_answer_min_rms) == (0.006, 0.015)


def test_listen_min_rms_decides_acceptance():
    texts = []
    assert "user_text" in answer(make(texts=texts, listen_min_rms=0.008), 10, VOICE)
    assert texts == ["네"]
    texts2 = []
    assert "wake_silent" in answer(make(texts=texts2, listen_min_rms=0.010), 10, VOICE)
    assert texts2 == []


def test_short_answer_min_rms_decides_short_rescue():
    texts = []
    assert "user_text" in answer(make(texts=texts, short_answer_min_rms=0.02), 1, LOUD)
    texts2 = []
    assert "wake_silent" in answer(make(texts=texts2, short_answer_min_rms=0.03), 1, LOUD)


def test_set_thresholds_listen_values():
    texts = []
    m = make(texts=texts)
    m.set_thresholds(listen_min_rms=0.010)
    assert "wake_silent" in answer(m, 10, VOICE) and texts == []


def test_set_thresholds_wake():
    wakes = []
    fake = Fake(scores=[(0.8, 0), (0.8, 0)])
    m = WakewordMonitor(on_emergency=lambda e: None, on_user_text=lambda t: None,
                        on_wake=lambda: wakes.append(1), predict=fake.predict, transcribe=fake.transcribe)
    m.set_thresholds(wake=0.85)
    run_frames(m, 2, LOUD)
    assert wakes == []
    assert m.gate_a.threshold == m.gate_a_listen.threshold == 0.85


def test_set_thresholds_emergency_single():
    fake = Fake(scores=[(0, 0.9)] + [(0, 0)] * 10, text="멈춰")
    events = []
    m = WakewordMonitor(on_emergency=events.append, on_user_text=lambda t: None,
                        predict=fake.predict, transcribe=fake.transcribe)
    m.set_thresholds(emergency_single=0.95)
    run_frames(m, 6, LOUD)
    assert events == [] and fake.stt_calls == 0


def test_on_frame_reports_each_frame_without_changing_results():
    seq = [(0.9, 0), (0.9, 0)] + [(0, 0)] * 6
    plain = WakewordMonitor(on_emergency=lambda e: None, on_user_text=lambda t: None,
                            predict=Fake(seq).predict, transcribe=lambda a: "")
    infos = []
    watched = WakewordMonitor(on_emergency=lambda e: None, on_user_text=lambda t: None,
                              predict=Fake(seq).predict, transcribe=lambda a: "", on_frame=infos.append)
    assert run_frames(plain, 8, LOUD) == run_frames(watched, 8, LOUD)
    assert len(infos) == 8
    assert infos[0]["a"] == 0.9 and infos[0]["b"] == 0 and infos[0]["state"] == "idle"
    assert infos[2]["state"] == "listen"
    assert abs(infos[0]["rms"] - 3000 / 32768) < 1e-6
    assert infos[0]["muted"] is False and infos[0]["speaking"] is False


def test_on_frame_while_muted():
    infos = []
    m = WakewordMonitor(on_emergency=lambda e: None, on_user_text=lambda t: None,
                        predict=Fake([]).predict, transcribe=lambda a: "", on_frame=infos.append)
    m.set_muted(True, now=0.0)
    run_frames(m, 1, LOUD, t0=0.01)
    assert infos[0]["muted"] is True and infos[0]["a"] == 0.0 and infos[0]["b"] == 0.0


def test_state_property():
    m = make(Fake(scores=[(0.9, 0), (0.9, 0)]))
    assert m.state == "idle"
    run_frames(m, 2, LOUD)
    assert m.state == "listen"


def test_injected_listen_transcriber_is_used_in_free_window():
    texts = []
    fake = Fake(scores=[(0.9, 0), (0.9, 0)], text="긴급용")
    m = WakewordMonitor(on_emergency=lambda e: None, on_user_text=texts.append,
                        predict=fake.predict, transcribe=fake.transcribe,
                        transcribe_listen=lambda a: "화장실로 가자")
    run_frames(m, 2, LOUD)
    run_frames(m, 10, LOUD, t0=0.16, vad=True)
    run_frames(m, 40, QUIET, t0=0.96, vad=False)
    assert texts == ["화장실로 가자"]

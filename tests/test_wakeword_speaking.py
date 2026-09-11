"""AEC 모드(set_speaking) 검증 — TTS 재생 중에도 귀가 열려 있어야 한다."""
from __future__ import annotations

import numpy as np

from src.wakeword_monitor import (
    FOLLOWUP_ARM_TIMEOUT_SEC,
    GATE_B_SPEAKING,
    POST_ROLL_FRAMES,
    WakewordMonitor,
)

LOUD = np.full(1280, 3000, dtype=np.int16)
QUIET = np.zeros(1280, dtype=np.int16)


class Fake:
    def __init__(self, scores=(), text=""):
        self.scores = list(scores)
        self.text = text

    def predict(self, _frame):
        a, b = self.scores.pop(0) if self.scores else (0.0, 0.0)
        return {"a": a, "b": b}

    def transcribe(self, _audio):
        return self.text


def make(fake: Fake, events: list, texts: list, wakes: list):
    return WakewordMonitor(
        on_emergency=events.append,
        on_user_text=texts.append,
        on_wake=lambda: wakes.append(1),
        predict=fake.predict,
        transcribe=fake.transcribe,
    )


def run_frames(m, n, frame=QUIET, t0=0.0, vad=None):
    out = []
    for i in range(n):
        out.append(m.process_frame(frame, now=t0 + i * 0.08, vad=vad))
    return out


def speak_answer(m, t0: float):
    """발화 5프레임(칩 판정 True) + 침묵 — 말끝 감지로 청취가 닫힌다."""
    run_frames(m, 5, LOUD, t0=t0, vad=True)
    return run_frames(m, 11, QUIET, t0=t0 + 5 * 0.08)


def test_emergency_heard_while_robot_is_speaking():
    """핵심 안전 개선: 로봇이 말하는 도중의 "멈춰"가 들려야 한다."""
    fake = Fake(scores=[(0, 0.9), (0, 0.9)], text="멈춰")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)

    m.set_speaking(True, now=0.0)
    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD, t0=0.1)
    assert results[-1] == "emergency"
    assert len(events) == 1 and events[0].keyword == "멈춰"


def test_wakeword_heard_while_robot_is_speaking():
    """barge-in 의 기초: 재생 중 "비카야"가 관문을 넘어야 한다."""
    fake = Fake(scores=[(0.9, 0), (0.9, 0)], text="")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)

    m.set_speaking(True, now=0.0)
    results = run_frames(m, 2, LOUD, t0=0.1)
    assert "wake" in results
    assert wakes == [1]


def test_followup_opens_after_question_without_mute():
    """질문 예약 흐름이 뮤트 없이도 그대로 살아야 한다."""
    fake = Fake(text="응")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)

    m.arm_followup(now=0.0)
    m.set_speaking(True, now=0.1)
    m.set_speaking(False, now=1.0)

    results = speak_answer(m, t0=1.0)
    assert results[-1] == "user_text"
    assert texts == ["응"]
    assert wakes == []


def test_multi_sentence_question_reopens_at_final_sentence():
    """문장 사이에 열린 재청취 창은 다음 문장 재생이 접고, 예약을 되살린다."""
    fake = Fake(text="아니요")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)

    m.arm_followup(now=0.0)
    m.set_speaking(True, now=0.1)
    m.set_speaking(False, now=1.0)
    m.set_speaking(True, now=1.2)
    m.set_speaking(False, now=2.5)

    results = speak_answer(m, t0=2.5)
    assert results[-1] == "user_text"
    assert texts == ["아니요"]


def test_stale_arm_does_not_open_mic_much_later():
    fake = Fake(text="응")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)

    m.arm_followup(now=0.0)
    later = FOLLOWUP_ARM_TIMEOUT_SEC + 5.0
    m.set_speaking(True, now=later)
    m.set_speaking(False, now=later + 1.0)

    speak_answer(m, t0=later + 1.0)
    assert texts == []


def test_no_arm_means_no_listening_after_tts():
    fake = Fake(text="응")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)

    m.set_speaking(True, now=0.1)
    m.set_speaking(False, now=1.0)

    speak_answer(m, t0=1.0)
    assert texts == []


def test_gate_b_relaxed_only_while_speaking():
    """긴급 관문은 재생 중에만 완화(0.35)되고 끝나면 원값으로 복원돼야 한다."""
    m = make(Fake(), [], [], [])
    base = m.gate_b.threshold
    assert GATE_B_SPEAKING < base

    m.set_speaking(True, now=0.0)
    assert m.gate_b.threshold == GATE_B_SPEAKING
    m.set_speaking(False, now=1.0)
    assert m.gate_b.threshold == base


def test_emergency_fires_at_relaxed_gate_during_speech():
    """재생 중에는 0.4점짜리 외침(평시엔 미달)도 관문을 넘어야 한다 —"""
    fake = Fake(scores=[(0, 0.4), (0, 0.4)], text="멈춰")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)

    m.set_speaking(True, now=0.0)
    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD, t0=0.1)
    assert results[-1] == "emergency"
    assert len(events) == 1


def test_gate_unrelaxed_when_silent_same_score_does_nothing():
    """같은 0.4점이라도 로봇이 조용할 때는 평시 관문(0.5)이 그대로다."""
    fake = Fake(scores=[(0, 0.4), (0, 0.4)], text="멈춰")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)

    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD, t0=0.1)
    assert "emergency" not in results
    assert events == []


def test_ring_buffer_survives_tts_boundary():
    """set_speaking 은 버퍼를 비우지 않아야 한다 — TTS 직후 긴급 검증이"""
    fake = Fake(scores=[(0, 0)] * 10 + [(0, 0.9), (0, 0.9)], text="멈춰")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)

    run_frames(m, 10, LOUD, t0=0.0)
    m.set_speaking(True, now=0.9)
    m.set_speaking(False, now=1.0)
    assert len(m._ring) == 10

    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD, t0=1.1)
    assert results[-1] == "emergency"

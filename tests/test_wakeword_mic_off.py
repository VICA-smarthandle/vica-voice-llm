"""장애물 정지 안내를 말하는 동안 마이크를 닫는다(2026-10-10 사용자 결정)."""
from __future__ import annotations

import numpy as np

from src.mission_phrases import OBSTACLE_AVOID, OBSTACLE_SLOW, OBSTACLE_STOP
from src.wakeword_monitor import POST_ROLL_FRAMES, WakewordMonitor, mic_off_while_saying

LOUD = np.full(1280, 3000, dtype=np.int16)


class Fake:
    def __init__(self, scores=(), text=""):
        self.scores = list(scores)
        self.text = text

    def predict(self, _frame):
        a, b = self.scores.pop(0) if self.scores else (0.0, 0.0)
        return {"a": a, "b": b}

    def transcribe(self, _audio):
        return self.text


def make(fake: Fake, events: list):
    return WakewordMonitor(
        on_emergency=events.append,
        on_user_text=lambda _t: None,
        on_wake=lambda: None,
        predict=fake.predict,
        transcribe=fake.transcribe,
    )


def run_frames(m, n, t0):
    return [m.process_frame(LOUD, now=t0 + i * 0.08) for i in range(n)]


def start_speaking(m, text, now):
    """TTS 노드 순서 그대로: 문장 소식(tts_now) → 재생 시작(tts_state True)."""
    m.note_robot_speech(text, now=now)
    m.set_speaking(True, now=now)


def test_only_the_stop_sentence_closes_the_mic():
    assert mic_off_while_saying(OBSTACLE_STOP)
    assert mic_off_while_saying(" " + OBSTACLE_STOP + " ")
    for text in (OBSTACLE_AVOID, OBSTACLE_SLOW, "화장실로 안내해드릴까요?", "", None):
        assert not mic_off_while_saying(text)


def test_robot_saying_stop_does_not_trigger_emergency():
    fake = Fake(scores=[(0, 0.9)] * 20, text="정지")
    events = []
    m = make(fake, events)
    start_speaking(m, OBSTACLE_STOP, now=0.0)
    results = run_frames(m, 2 + POST_ROLL_FRAMES, t0=0.1)
    assert "emergency" not in results and events == []


def test_mic_opens_again_when_that_speech_ends():
    fake = Fake(scores=[(0, 0.0)] * 2 + [(0, 0.9)] * 2, text="멈춰")
    events = []
    m = make(fake, events)
    start_speaking(m, OBSTACLE_STOP, now=0.0)
    run_frames(m, 4, t0=0.1)
    assert fake.scores and events == []
    m.set_speaking(False, now=2.4)
    run_frames(m, 2, t0=2.5)
    results = run_frames(m, 1 + POST_ROLL_FRAMES, t0=2.7)
    assert results[-1] == "emergency" and events[0].keyword == "멈춰"


def test_other_robot_speech_keeps_the_mic_open():
    fake = Fake(scores=[(0, 0.9), (0, 0.9)], text="멈춰")
    events = []
    m = make(fake, events)
    start_speaking(m, OBSTACLE_SLOW, now=0.0)
    results = run_frames(m, 1 + POST_ROLL_FRAMES, t0=0.1)
    assert results[-1] == "emergency" and len(events) == 1


def test_mute_mode_end_also_reopens():
    m = make(Fake(), [])
    m.note_robot_speech(OBSTACLE_STOP, now=0.0)
    m.set_muted(True, now=0.0)
    m.set_muted(False, now=2.0)
    assert not m._is_muted(2.1) and not m._mic_off_speech


def test_lost_end_signal_reopens_by_failsafe():
    m = make(Fake(), [])
    m.note_robot_speech(OBSTACLE_STOP, now=0.0)
    assert m._is_muted(5.0)
    assert not m._is_muted(10.5) and not m._mic_off_speech

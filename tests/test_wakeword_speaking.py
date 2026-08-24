"""AEC 모드(set_speaking) 검증 — TTS 재생 중에도 귀가 열려 있어야 한다.

set_muted(뮤트)의 대체 경로다. 가짜 predict/transcribe 주입, 마이크/모델/STT
불필요 (test_wakeword_followup 과 같은 패턴). 프레임은 80ms(1280샘플) int16.
"""
from __future__ import annotations

import numpy as np

from src.wakeword_monitor import (
    FOLLOWUP_ARM_TIMEOUT_SEC,
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


def run_frames(m, n, frame=QUIET, t0=0.0):
    out = []
    for i in range(n):
        out.append(m.process_frame(frame, now=t0 + i * 0.08))
    return out


def speak_answer(m, t0: float):
    """발화 5프레임 + 침묵 0.88초(11프레임) — 말끝 감지로 청취가 닫힌다."""
    run_frames(m, 5, LOUD, t0=t0)
    return run_frames(m, 11, QUIET, t0=t0 + 5 * 0.08)


def test_emergency_heard_while_robot_is_speaking():
    """핵심 안전 개선: 로봇이 말하는 도중의 "멈춰"가 들려야 한다.

    뮤트 모드에서는 이 구간이 감시 공백이었다 (backlog 1순위 문제).
    """
    fake = Fake(scores=[(0, 0.9), (0, 0.9)], text="멈춰")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)

    m.set_speaking(True, now=0.0)   # TTS 재생 시작 — 귀는 열려 있다
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
    m.set_speaking(True, now=0.1)   # 질문 재생 시작
    m.set_speaking(False, now=1.0)  # 질문 끝 → 재청취 창

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
    m.set_speaking(False, now=1.0)  # 문장 사이 — 창이 잠깐 열림
    m.set_speaking(True, now=1.2)   # 2문장 시작 — 창을 접는다
    m.set_speaking(False, now=2.5)  # 질문 진짜 끝

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


def test_ring_buffer_survives_tts_boundary():
    """set_speaking 은 버퍼를 비우지 않아야 한다 — TTS 직후 긴급 검증이
    직전 오디오(사용자 외침의 앞부분)를 참조할 수 있어야 한다.
    (뮤트 모드의 알려진 확인 항목: jetson-handoff 5절 '1번의 확인 항목')
    """
    # 앞의 10프레임(재생 전)은 점수 0, 경계 뒤 두 프레임에서 관문을 넘는다.
    fake = Fake(scores=[(0, 0)] * 10 + [(0, 0.9), (0, 0.9)], text="멈춰")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)

    run_frames(m, 10, LOUD, t0=0.0)          # 재생 전 오디오가 링에 쌓인다
    m.set_speaking(True, now=0.9)
    m.set_speaking(False, now=1.0)
    assert len(m._ring) == 10                 # 경계에서 비워지지 않았다

    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD, t0=1.1)
    assert results[-1] == "emergency"

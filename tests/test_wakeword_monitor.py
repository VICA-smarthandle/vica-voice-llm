"""WakewordMonitor 의 순수 로직(process_frame) 검증. 마이크/모델/STT 불필요.

가짜 predict/transcribe 를 주입한다 (test_emergency_monitor 와 같은 패턴).
프레임은 80ms(1280샘플) int16.
"""
from __future__ import annotations

import numpy as np

from src.schema import EmergencyEvent
from src.wakeword_monitor import POST_ROLL_FRAMES, WakewordMonitor

LOUD = np.full(1280, 3000, dtype=np.int16)    # 발화 판정 RMS 를 넘는 프레임
QUIET = np.zeros(1280, dtype=np.int16)


class Fake:
    """프레임 순서대로 (a, b) 점수를 돌려주는 가짜 모델 + 고정 전사 STT."""

    def __init__(self, scores, text=""):
        self.scores = list(scores)
        self.text = text
        self.stt_calls = 0

    def predict(self, _frame):
        a, b = self.scores.pop(0) if self.scores else (0.0, 0.0)
        return {"a": a, "b": b}

    def transcribe(self, _audio):
        self.stt_calls += 1
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


def test_emergency_confirmed():
    # B 점수 2연속 → 포스트롤 → 전사 "멈춰!" → 이벤트 발행
    fake = Fake(scores=[(0, 0.9), (0, 0.9)] + [(0, 0)] * 10, text="멈춰!")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD)
    assert results[-1] == "emergency"
    assert len(events) == 1
    assert events[0].keyword == "멈춰"
    assert events[0].source_text == "멈춰!"
    assert isinstance(events[0], EmergencyEvent)


def test_emergency_rejected_by_stt():
    # 관문은 뚫렸지만 전사가 "멈춤" — STT 층이 막는다 (실측에서 확인된 함정)
    fake = Fake(scores=[(0, 0.9), (0, 0.9)] + [(0, 0)] * 10, text="멈춤")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD)
    assert results[-1] == "reject"
    assert events == []


def test_single_spike_no_stt_call():
    # 1프레임 스파이크는 발동하지 않고 whisper 도 부르지 않는다
    fake = Fake(scores=[(0, 0.9)] + [(0, 0)] * 10, text="멈춰")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    run_frames(m, 8, LOUD)
    assert events == [] and fake.stt_calls == 0


def test_wake_then_user_text():
    # A 2연속 → wake(응답음) → 발화(칩 판정) → 침묵 → 전사 → on_user_text
    speech_frames = 5
    silence_frames = 12          # 12×80ms ≈ 0.96초 > 0.8초
    fake = Fake(scores=[(0.9, 0), (0.9, 0)] + [(0, 0)] * 40, text="화장실 어디야")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    assert run_frames(m, 2, LOUD)[-1] == "wake"
    assert wakes == [1]
    run_frames(m, speech_frames, LOUD, t0=1.0, vad=True)  # 사용자 발화
    results = run_frames(m, silence_frames, QUIET, t0=2.0)  # 침묵 → 종료
    assert "user_text" in results
    assert texts == ["화장실 어디야"]
    assert events == []


def test_wake_silent_timeout():
    # 호출 후 아무 말 없음 → user_text 없이 복귀 (호출 오탐 기록 지점)
    fake = Fake(scores=[(0.9, 0), (0.9, 0)] + [(0, 0)] * 200, text="")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    run_frames(m, 2, LOUD)
    results = run_frames(m, 80, QUIET, t0=1.0)   # 6.4초 — 상한 초과
    assert "wake_silent" in results
    assert texts == []


def test_emergency_preempts_listen():
    # 청취 창 도중 긴급어 — 긴급이 절대 우선 (명세 11절)
    fake = Fake(
        scores=[(0.9, 0), (0.9, 0), (0, 0.9), (0, 0.9)] + [(0, 0)] * 10,
        text="정지",
    )
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    run_frames(m, 2, LOUD)                        # wake → listen
    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD, t0=1.0)
    assert results[-1] == "emergency"
    assert events[0].keyword == "정지"
    assert texts == []                            # 청취는 폐기됐다


def test_muted_suppresses_and_failsafe_unmutes():
    fake = Fake(scores=[(0, 0.9)] * 40, text="멈춰")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    m.set_muted(True, now=0.0, failsafe_sec=10.0)
    run_frames(m, 10, LOUD, t0=0.0)               # 억제 중 — 발동 없음
    assert events == []
    # fail-safe: 10초 뒤엔 해제 신호를 놓쳐도 자동 해제
    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD, t0=11.0)
    assert results[-1] == "emergency"


def test_unmute_clears_ring():
    # 해제 시 버퍼를 비운다 — 억제 중 소리가 검증 전사에 섞이지 않게
    fake = Fake(scores=[(0, 0)] * 5 + [(0, 0.9)] * 10, text="멈춰")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    m.set_muted(True, now=0.0)
    run_frames(m, 5, LOUD, t0=0.0)
    m.set_muted(False, now=0.5)
    assert len(m._ring) == 0

"""질문 후 재청취(followup) 검증 — "응" 한마디에 "비카야" 재호출이 필요 없어야 한다.

가짜 predict/transcribe 주입, 마이크/모델/STT 불필요 (test_wakeword_monitor 와
같은 패턴). 프레임은 80ms(1280샘플) int16.

흐름: 질문 노드가 /vica/listen_request → arm_followup() → 질문 TTS 종료
(set_muted False) 순간 웨이크워드 없이 청취 창이 열린다.
"""
from __future__ import annotations

import numpy as np

from src.wakeword_monitor import (
    CONFIRM_WINDOW_SEC,
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


def run_frames(m, n, frame=QUIET, t0=0.0, vad=None):
    out = []
    for i in range(n):
        out.append(m.process_frame(frame, now=t0 + i * 0.08, vad=vad))
    return out


def speak_answer(m, t0: float):
    """발화 5프레임(칩 판정 True) + 침묵 — 말끝 감지로 청취가 닫힌다."""
    run_frames(m, 5, LOUD, t0=t0, vad=True)
    return run_frames(m, 11, QUIET, t0=t0 + 5 * 0.08)


def test_followup_opens_after_question_tts_and_hears_answer():
    """예약 → 질문 TTS 종료 → 웨이크워드 없이 "응"이 들려야 한다."""
    fake = Fake(text="응")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)

    m.arm_followup(now=0.0)
    m.set_muted(True, now=0.1)    # 질문 재생 시작
    m.set_muted(False, now=1.0)   # 질문 끝 → 재청취 창

    results = speak_answer(m, t0=1.0)
    assert results[-1] == "user_text"
    assert texts == ["응"]
    assert wakes == []            # 웨이크워드를 부른 적이 없다


def test_no_arm_means_no_listening_after_tts():
    """예약이 없으면(평범한 안내 멘트) TTS 끝나도 귀가 열리면 안 된다."""
    fake = Fake(text="응")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)

    m.set_muted(True, now=0.1)
    m.set_muted(False, now=1.0)

    speak_answer(m, t0=1.0)
    assert texts == []


def test_multi_sentence_question_reopens_at_final_sentence():
    """TTS 는 문장마다 상태를 깜빡인다. 두 문장짜리 질문이면 마지막 문장 끝에
    열린 창에서 답을 들어야 한다 (문장 사이에 열렸다 닫혀도 예약이 살아야 함)."""
    fake = Fake(text="아니요")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)

    m.arm_followup(now=0.0)
    m.set_muted(True, now=0.1)    # 1문장 재생
    m.set_muted(False, now=1.0)   # 문장 사이 — 창이 잠깐 열림
    m.set_muted(True, now=1.2)    # 2문장 재생 시작 — 창을 접고 예약 부활
    m.set_muted(False, now=2.5)   # 질문 진짜 끝

    results = speak_answer(m, t0=2.5)
    assert results[-1] == "user_text"
    assert texts == ["아니요"]


def test_stale_arm_does_not_open_mic_much_later():
    """질문 TTS 가 유실되면 예약이 남는다. 한참 뒤 무관한 안내가 끝난 순간
    마이크가 열리는 오동작을 타임아웃이 막아야 한다."""
    fake = Fake(text="응")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)

    m.arm_followup(now=0.0)
    later = FOLLOWUP_ARM_TIMEOUT_SEC + 5.0
    m.set_muted(True, now=later)
    m.set_muted(False, now=later + 1.0)

    speak_answer(m, t0=later + 1.0)
    assert texts == []


def test_disarm_cancels_reservation():
    fake = Fake(text="응")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)

    m.arm_followup(now=0.0)
    m.disarm_followup()
    m.set_muted(True, now=0.1)
    m.set_muted(False, now=1.0)

    speak_answer(m, t0=1.0)
    assert texts == []


def test_emergency_still_wins_inside_followup_listen():
    """재청취 창 안에서도 긴급어가 절대 우선이어야 한다 (안전 불변)."""
    # 창이 열린 뒤 첫 두 프레임에서 모델 B 점수가 관문을 넘는다.
    fake = Fake(scores=[(0, 0.9), (0, 0.9)], text="멈춰")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)

    m.arm_followup(now=0.0)
    m.set_muted(True, now=0.1)
    m.set_muted(False, now=1.0)

    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD, t0=1.0)
    assert results[-1] == "emergency"
    assert len(events) == 1 and events[0].keyword == "멈춰"
    assert texts == []            # 답변 발화로 처리되지 않았다


def test_silent_followup_returns_quietly():
    """열렸는데 아무도 답하지 않으면 조용히 닫힌다 — 이후는 오늘과 같다
    (다시 부르면 됨). 엉뚱한 텍스트가 LLM 으로 가면 안 된다."""
    fake = Fake(text="지어낸 말")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)

    m.arm_followup(now=0.0)
    m.set_muted(True, now=0.1)
    m.set_muted(False, now=1.0)

    frames = int(CONFIRM_WINDOW_SEC / 0.08) + 2
    results = run_frames(m, frames, QUIET, t0=1.0)
    assert "wake_silent" in results      # 확인 창 상한(30초)에서 닫혔다
    assert "user_text" not in results
    assert texts == []

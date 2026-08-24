"""barge-in 검증 — 로봇이 질문하는 중에 답을 시작하면 말을 끊고 들어야 한다.

두 부품을 본다: WakewordMonitor 의 끼어들기 감지(질문 재생 중 연속 발화 프레임)
와 TtsQueue.drop_pending(하던 말과 밀린 일반 발화 버리기, 긴급은 보존).
가짜 predict/transcribe 주입 — 마이크/모델/STT 불필요.
"""
from __future__ import annotations

import numpy as np

from src.tts_queue import EMERGENCY, NARRATION, RESPONSE, TtsQueue
from src.wakeword_monitor import (
    BARGE_IN_FRAMES,
    FOLLOWUP_ARM_TIMEOUT_SEC,
    POST_ROLL_FRAMES,
    WakewordMonitor,
)

LOUD = np.full(1280, 3000, dtype=np.int16)
QUIET = np.zeros(1280, dtype=np.int16)
# barge-in 판정은 "잔여 에코 대비 배수"다. ECHO 는 AEC 수렴 전 잔여 수준
# (rms ≈ 0.03, ROS 종단 실측 0.057 근처), USER 는 그 10배짜리 사람 목소리.
ECHO = np.full(1280, 1000, dtype=np.int16)
USER = np.full(1280, 10000, dtype=np.int16)


class Fake:
    def __init__(self, scores=(), text=""):
        self.scores = list(scores)
        self.text = text
        self.heard: list[int] = []  # transcribe 가 받은 오디오 길이(샘플)

    def predict(self, _frame):
        a, b = self.scores.pop(0) if self.scores else (0.0, 0.0)
        return {"a": a, "b": b}

    def transcribe(self, audio):
        self.heard.append(len(audio))
        return self.text


def make(fake: Fake, events: list, texts: list, stops: list):
    return WakewordMonitor(
        on_emergency=events.append,
        on_user_text=texts.append,
        on_barge_in=lambda: stops.append(1),
        predict=fake.predict,
        transcribe=fake.transcribe,
    )


def run_frames(m, n, frame=QUIET, t0=0.0):
    out = []
    for i in range(n):
        out.append(m.process_frame(frame, now=t0 + i * 0.08))
    return out


# ---------------------------------------------------------------- monitor
def test_barge_in_cuts_question_and_hears_full_answer():
    """질문 재생 중 발화 시작 → 끊기 콜백 + 청취. 말머리도 놓치지 않아야 한다."""
    fake = Fake(text="네 화장실이요")
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops)

    m.arm_followup(now=0.0)
    m.set_speaking(True, now=0.1)         # 질문 재생 중

    run_frames(m, 3, ECHO, t0=0.2)        # 잔여 에코 — 기준선 학습
    results = run_frames(m, BARGE_IN_FRAMES, USER, t0=0.5)  # 사람이 답 시작
    assert results[-1] == "barge_in"
    assert stops == [1]

    # 이어지는 답변 3프레임 + 침묵 11프레임 → 말끝 감지로 전사
    run_frames(m, 3, USER, t0=0.9)
    results = run_frames(m, 11, QUIET, t0=1.2)
    assert results[-1] == "user_text"
    assert texts == ["네 화장실이요"]
    # 감지에 쓴 말머리(BARGE_IN_FRAMES)가 전사 오디오에 포함돼야 한다
    assert fake.heard[-1] == (BARGE_IN_FRAMES + 3 + 11) * 1280


def test_no_barge_in_without_question():
    """예약 없는 일반 안내 중에는 소리가 나도 끼어들기가 없어야 한다
    (복도 소음마다 로봇이 말을 삼키면 안 된다)."""
    fake = Fake()
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops)

    m.set_speaking(True, now=0.1)
    results = run_frames(m, 10, LOUD, t0=0.2)
    assert stops == []
    assert all(r is None for r in results)


def test_no_barge_in_when_robot_is_silent():
    """로봇이 말을 마친 뒤에는 barge-in 이 아니라 일반 재청취 흐름이다."""
    fake = Fake()
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops)

    m.arm_followup(now=0.0)
    results = run_frames(m, 10, LOUD, t0=0.2)   # speaking=False
    assert stops == []
    assert "barge_in" not in results


def test_short_noise_does_not_barge_in():
    """연속 조건: 짧은 소음(문 소리)으로 질문이 끊기면 안 된다."""
    fake = Fake()
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops)

    m.arm_followup(now=0.0)
    m.set_speaking(True, now=0.1)
    run_frames(m, 3, ECHO, t0=0.2)                      # 기준선
    run_frames(m, BARGE_IN_FRAMES - 1, USER, t0=0.5)    # 3프레임 소음
    run_frames(m, 1, ECHO, t0=0.8)                      # 끊김 → streak 리셋
    results = run_frames(m, BARGE_IN_FRAMES - 1, USER, t0=0.9)
    assert stops == []
    assert "barge_in" not in results


def test_steady_echo_alone_does_not_barge_in():
    """자기 잔여 에코만으로 질문이 끊기면 안 된다 — ROS 종단 시험에서 질문
    재생 0.3초 만에 자기 목소리 잔여(rms 0.057)로 스스로 barge-in 한 회귀의
    재발 방지 (2026-08-24). 잔여는 일정하거나 서서히 줄므로 '기준선 대비
    배수' 조건을 절대 못 넘는다."""
    fake = Fake()
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops)

    m.arm_followup(now=0.0)
    m.set_speaking(True, now=0.1)
    results = run_frames(m, 30, ECHO, t0=0.2)   # 2.4초 내내 잔여 에코
    assert stops == []
    assert "barge_in" not in results


def test_soft_answer_falls_back_to_followup_window():
    """기준선 3배에 못 미치는 작은 답은 barge-in 을 안 내지만, 질문이 끝나면
    재청취 창(안전망)이 열려 결국 들린다 — 보수적 판정이 안전한 이유."""
    fake = Fake(text="응")
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops)

    m.arm_followup(now=0.0)
    m.set_speaking(True, now=0.1)
    run_frames(m, 3, ECHO, t0=0.2)
    soft = np.full(1280, 2000, dtype=np.int16)   # 에코의 2배 — 3배 미만
    results = run_frames(m, 6, soft, t0=0.5)
    assert stops == [] and "barge_in" not in results

    m.set_speaking(False, now=1.2)               # 질문 끝 → 안전망 창
    run_frames(m, 5, USER, t0=1.3)
    results = run_frames(m, 11, QUIET, t0=1.7)
    assert results[-1] == "user_text"
    assert texts == ["응"]


def test_emergency_beats_barge_in_during_question():
    """질문 중이라도 긴급(모델 B)이 절대 우선 — barge-in 으로 소비되면 안 된다."""
    fake = Fake(scores=[(0, 0.9), (0, 0.9)], text="멈춰")
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops)

    m.arm_followup(now=0.0)
    m.set_speaking(True, now=0.1)
    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD, t0=0.2)
    assert results[-1] == "emergency"
    assert len(events) == 1
    assert stops == []          # 모니터 수준에선 미발동 (노드가 긴급 시 직접 끊는다)


def test_stale_question_does_not_barge_in():
    """오래된 예약(질문 유실)으로는 한참 뒤 안내가 끊기면 안 된다."""
    fake = Fake()
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops)

    m.arm_followup(now=0.0)
    later = FOLLOWUP_ARM_TIMEOUT_SEC + 5.0
    m.set_speaking(True, now=later)
    results = run_frames(m, 10, LOUD, t0=later)
    assert stops == []
    assert "barge_in" not in results


def test_barge_in_consumes_reservation():
    """barge-in 뒤 TTS 종료 신호가 와도 두 번째 청취 창이 열리면 안 된다."""
    fake = Fake(text="응")
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops)

    m.arm_followup(now=0.0)
    m.set_speaking(True, now=0.1)
    run_frames(m, 3, ECHO, t0=0.2)                      # 기준선
    run_frames(m, BARGE_IN_FRAMES, USER, t0=0.5)        # barge_in → listen
    run_frames(m, 3, USER, t0=0.9)
    run_frames(m, 11, QUIET, t0=1.2)                    # user_text 로 닫힘
    assert texts == ["응"]

    m.set_speaking(False, now=2.5)                      # 끊긴 TTS 의 종료 신호
    run_frames(m, 5, USER, t0=2.6)
    run_frames(m, 11, QUIET, t0=3.1)
    assert texts == ["응"]                              # 두 번째 전사가 없다


# ---------------------------------------------------------------- queue
def test_drop_pending_keeps_emergency():
    # 긴급을 먼저 넣는다 — 긴급 push 는 그 시점의 일반 대기를 밀어내기 때문
    # (기존 선점 동작). 그 뒤에 쌓인 일반 발화를 drop_pending 이 버려야 한다.
    q = TtsQueue()
    q.push(EMERGENCY, "긴급 정지했습니다", now=0.0)
    q.push(NARRATION, "복도를 지나는 중입니다", now=0.1)
    q.push(RESPONSE, "네 알겠습니다", now=0.2)

    dropped = q.drop_pending(keep_emergency=True)
    assert set(dropped) == {"복도를 지나는 중입니다", "네 알겠습니다"}
    assert len(q) == 1
    assert q.pop().priority == EMERGENCY


def test_drop_pending_all():
    q = TtsQueue()
    q.push(EMERGENCY, "긴급", now=0.0)
    dropped = q.drop_pending(keep_emergency=False)
    assert dropped == ("긴급",)
    assert len(q) == 0


def test_drop_pending_empty_queue():
    assert TtsQueue().drop_pending() == ()

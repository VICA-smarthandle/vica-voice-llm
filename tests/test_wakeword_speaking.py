"""AEC 모드(set_speaking) 검증 — TTS 재생 중에도 귀가 열려 있어야 한다.

set_muted(뮤트)의 대체 경로다. 가짜 predict/transcribe 주입, 마이크/모델/STT
불필요 (test_wakeword_followup 과 같은 패턴). 프레임은 80ms(1280샘플) int16.
"""
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


def test_gate_b_relaxed_only_while_speaking():
    """긴급 관문은 재생 중에만 완화(0.35)되고 끝나면 원값으로 복원돼야 한다.

    근거: 외침 10회 실측에서 실패 주원인이 관문 미달(근접 0.27 포함)이었다.
    """
    m = make(Fake(), [], [], [])
    base = m.gate_b.threshold
    assert GATE_B_SPEAKING < base

    m.set_speaking(True, now=0.0)
    assert m.gate_b.threshold == GATE_B_SPEAKING
    m.set_speaking(False, now=1.0)
    assert m.gate_b.threshold == base


def test_emergency_fires_at_relaxed_gate_during_speech():
    """재생 중에는 0.4점짜리 외침(평시엔 미달)도 관문을 넘어야 한다 —
    단, whisper 정확 매칭 검증은 그대로 거친다."""
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


# ---- '비카야'는 옛 질문 예약을 지운다 (2026-10-06 실기 15:21) ------------------
def _frames(m, n, frame, t0, vad=None):
    out, t = [], t0
    for _ in range(n):
        out.append(m.process_frame(frame, now=t, vad=vad))
        t += 0.08
    return out, t


def _wake_after_stale_question(m):
    """되묻기 질문 → 질문 창 → 안내('응답이 없어…')가 창을 접고 예약을 되살림
    → 그 안내 도중 '비카야' → 안내 끝 → '네?'. 반환: 마지막 시각."""
    t = 0.0
    m.arm_followup(now=t)
    m.set_speaking(True, now=t)
    _, t = _frames(m, 62, QUIET, t, vad=False)       # 질문 재생 5초
    m.set_speaking(False, now=t)                     # 질문 창이 열린다
    _, t = _frames(m, 30, QUIET, t, vad=False)
    m.set_speaking(True, now=t)                      # 안내 시작 — 창 접고 예약 되살림
    out, t = _frames(m, 2, LOUD, t, vad=False)       # '비카야'
    assert out[-1] == "wake"
    _, t = _frames(m, 58, LOUD, t, vad=False)        # 안내 나머지
    m.set_speaking(False, now=t)
    m.set_speaking(True, now=t)
    _, t = _frames(m, 6, LOUD, t, vad=False)         # '네?'
    m.set_speaking(False, now=t)
    return t


def test_wake_cancels_a_stale_question_reservation():
    """안내가 끝나는 순간 옛 질문 예약이 '질문 답' 창을 다시 열어 호출 창을
    바꿔치기했다. 질문 창은 최소 개방·반짝 무효화가 없어 로봇 소리 한 조각에
    0.8초 만에 닫혔고(empty:ghost speech=0.00s), 그 뒤 '화장실로 가자'가 사라졌다."""
    fake = Fake(scores=[(0, 0)] * 92 + [(0.9, 0), (0.9, 0)], text="화장실로 가자")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    t = _wake_after_stale_question(m)
    assert m._listen_is_followup is False            # 호출 창 그대로
    out_b, t = _frames(m, 1, LOUD, t, vad=True)      # 로봇 소리 꼬리 한 조각
    out_q, t = _frames(m, 12, QUIET, t, vad=False)   # 사용자가 숨 고르는 1초
    out_s, t = _frames(m, 15, LOUD, t, vad=True)     # '화장실로 가자'
    out_e, t = _frames(m, 12, QUIET, t, vad=False)
    assert "wake_silent" not in out_b + out_q + out_s
    assert texts == ["화장실로 가자"]

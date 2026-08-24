"""barge-in 검증 — 로봇이 질문하는 중에 답을 시작하면 말을 끊고 들어야 한다.

발동 조건은 이중 증거다 (2026-08-24 실측 3회의 결론):
  1. 칩(XVF-3000)의 발화 판정(SPEECHDETECTED) 창 과반 — RMS 는 자기 잔여
     에코와 사람을 못 가려 폐기 (vad_probe: 로봇 단독 재생 0.0% / 발화 47.6%)
  2. 사용자 방향 — "비카야" 순간 잠금(1순위) 또는 장착 보정 부채꼴(2순위).
     칩 VAD 는 화자를 못 가려 옆사람 대화에 발동했었다. 시나리오상 사용자는
     항상 핸들(고정 방향)에 있고 모든 대화는 "비카야"로 시작한다(6.1).
     방향을 모르면 발동하지 않는다 — 그래서 기본 켬이 안전하다.

가짜 predict/transcribe 주입 — 마이크/모델/STT/칩 불필요 (vad·doa 는 인자).
"""
from __future__ import annotations

import numpy as np

from src.tts_queue import EMERGENCY, NARRATION, RESPONSE, TtsQueue
from src.wakeword_monitor import (
    BARGE_VAD_MIN_HITS,
    BARGE_VAD_WINDOW,
    FOLLOWUP_ARM_TIMEOUT_SEC,
    POST_ROLL_FRAMES,
    USER_DOA_LOCK_TTL_SEC,
    WakewordMonitor,
)

LOUD = np.full(1280, 3000, dtype=np.int16)    # 사람 목소리 크기
QUIET = np.zeros(1280, dtype=np.int16)

USER_DOA = 236.0      # doa_probe 실측: 사용자 235.9°±3.8°
MIXED_DOA = 217.0     # 실측: 옆사람 발화가 이중 발화에서 만든 섞인 방향


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


def make(fake: Fake, events: list, texts: list, stops: list, **kwargs):
    return WakewordMonitor(
        on_emergency=events.append,
        on_user_text=texts.append,
        on_barge_in=lambda: stops.append(1),
        predict=fake.predict,
        transcribe=fake.transcribe,
        **kwargs,
    )


def run_frames(m, n, frame=QUIET, t0=0.0, vad=None, doa=None):
    out = []
    for i in range(n):
        out.append(m.process_frame(frame, now=t0 + i * 0.08, vad=vad, doa=doa))
    return out


def armed_speaking(m, lock=USER_DOA):
    """질문 재생 중 상태를 만든다: 방향 잠금 + 재청취 예약 + 재생 시작."""
    if lock is not None:
        m.lock_user_direction(lock, now=0.0)
    m.arm_followup(now=0.0)
    m.set_speaking(True, now=0.1)


# ---------------------------------------------------------------- 발동
def test_barge_in_fires_with_speech_and_direction_and_hears_answer():
    """이중 증거(칩 발화 + 사용자 방향)면 끊기 콜백 + 청취 + 말머리 보존."""
    fake = Fake(text="네 화장실이요")
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops)

    armed_speaking(m)
    results = run_frames(m, BARGE_VAD_WINDOW, LOUD, t0=0.2, vad=True, doa=USER_DOA)
    assert results[-1] == "barge_in"
    assert stops == [1]

    run_frames(m, 3, LOUD, t0=1.1, vad=True)
    results = run_frames(m, 11, QUIET, t0=1.4)
    assert results[-1] == "user_text"
    assert texts == ["네 화장실이요"]
    # 판정 창(BARGE_VAD_WINDOW)만큼의 말머리가 전사 오디오에 포함돼야 한다
    assert fake.heard[-1] == (BARGE_VAD_WINDOW + 3 + 11) * 1280


# ---------------------------------------------------------------- 억제 (자책골·오발동 회귀)
def test_echo_never_fires_because_chip_says_no_speech():
    """자기 잔여 에코 회귀 방지: 방향이 맞아도 칩 발화 판정(vad=False)이
    없으면 발동하지 않는다 — 로봇 단독 재생 중 SPEECHDETECTED 0.0% 실측."""
    fake = Fake()
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops)

    armed_speaking(m)
    results = run_frames(m, 30, LOUD, t0=0.2, vad=False, doa=USER_DOA)
    assert stops == []
    assert "barge_in" not in results


def test_bystander_mixed_direction_does_not_fire():
    """옆사람 오발동 회귀 방지: 발화 판정이 있어도 방향이 잠금 밖(실측 217°,
    잠금 236°±15°)이면 발동하지 않는다."""
    fake = Fake()
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops)

    armed_speaking(m)
    results = run_frames(m, 30, LOUD, t0=0.2, vad=True, doa=MIXED_DOA)
    assert stops == []
    assert "barge_in" not in results


def test_no_direction_evidence_means_dormant():
    """방향 정보가 전혀 없으면(잠금도 보정도 없음) 발화 판정만으로는 절대
    발동하지 않는다 — 기본 켬이 안전한 근거."""
    fake = Fake()
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops)

    armed_speaking(m, lock=None)
    results = run_frames(m, 30, LOUD, t0=0.2, vad=True, doa=123)
    assert stops == []
    assert "barge_in" not in results


def test_no_chip_means_dormant():
    """칩을 못 읽으면(vad=None) 잠든다 — RMS 로 대체 판정하지 않는다."""
    fake = Fake()
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops)

    armed_speaking(m)
    results = run_frames(m, 30, LOUD, t0=0.2, vad=None, doa=USER_DOA)
    assert stops == []
    assert "barge_in" not in results


def test_sparse_vad_hits_do_not_fire():
    """창 과반 조건: 짧은 소음·오판(창의 절반 미만)으로는 안 끊긴다."""
    fake = Fake()
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops)

    armed_speaking(m)
    results = []
    for i in range(20):
        vad = i % 3 == 0        # 창(10)당 최대 4개 — 과반 미달
        results.append(m.process_frame(LOUD, now=0.2 + i * 0.08,
                                       vad=vad, doa=USER_DOA))
    assert stops == []
    assert "barge_in" not in results


def test_silent_frames_do_not_fire_even_with_vad():
    """건전성 바닥: 소리 자체가 없으면(무음 프레임) 발동하지 않는다."""
    fake = Fake()
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops)

    armed_speaking(m)
    results = run_frames(m, 30, QUIET, t0=0.2, vad=True, doa=USER_DOA)
    assert stops == []
    assert "barge_in" not in results


def test_no_barge_in_without_question():
    """예약 없는 일반 안내 중에는 이중 증거가 있어도 끼어들기가 없다."""
    fake = Fake()
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops)

    m.lock_user_direction(USER_DOA, now=0.0)
    m.set_speaking(True, now=0.1)          # 예약(arm) 없음
    results = run_frames(m, 20, LOUD, t0=0.2, vad=True, doa=USER_DOA)
    assert stops == []
    assert all(r is None for r in results)


def test_stale_question_does_not_barge_in():
    """오래된 예약(질문 유실)으로는 한참 뒤 안내가 끊기면 안 된다."""
    fake = Fake()
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops)

    m.lock_user_direction(USER_DOA, now=0.0)
    m.arm_followup(now=0.0)
    later = FOLLOWUP_ARM_TIMEOUT_SEC + 5.0
    m.set_speaking(True, now=later)
    results = run_frames(m, 20, LOUD, t0=later, vad=True, doa=USER_DOA)
    assert stops == []
    assert "barge_in" not in results


# ---------------------------------------------------------------- 우선순위·흐름
def test_emergency_beats_barge_in_during_question():
    """질문 중이라도 긴급(모델 B)이 절대 우선 — 그리고 긴급에는 방향 조건이
    없다 (행인의 "멈춰"도 정지 대상)."""
    fake = Fake(scores=[(0, 0.9), (0, 0.9)], text="멈춰")
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops)

    armed_speaking(m)
    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD, t0=0.2,
                         vad=True, doa=MIXED_DOA)   # 방향 밖 발화여도
    assert results[-1] == "emergency"               # 긴급은 통한다
    assert len(events) == 1
    assert stops == []


def test_barge_in_consumes_reservation():
    """barge-in 뒤 TTS 종료 신호가 와도 두 번째 청취 창이 열리면 안 된다."""
    fake = Fake(text="응")
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops)

    armed_speaking(m)
    run_frames(m, BARGE_VAD_WINDOW, LOUD, t0=0.2, vad=True, doa=USER_DOA)
    run_frames(m, 3, LOUD, t0=1.1, vad=True)
    run_frames(m, 11, QUIET, t0=1.4)
    assert texts == ["응"]

    m.set_speaking(False, now=2.5)
    run_frames(m, 5, LOUD, t0=2.6, vad=True)
    run_frames(m, 11, QUIET, t0=3.1)
    assert texts == ["응"]


def test_vad_window_resets_at_sentence_boundary():
    """문장 경계(set_speaking 깜빡임)에서 창이 비워져, 판정이 이월되지 않는다."""
    fake = Fake()
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops)

    armed_speaking(m)
    run_frames(m, BARGE_VAD_MIN_HITS - 1, LOUD, t0=0.2, vad=True, doa=USER_DOA)
    m.set_speaking(True, now=0.6)          # 다음 문장 — 창 리셋
    results = run_frames(m, BARGE_VAD_MIN_HITS, LOUD, t0=0.7,
                         vad=True, doa=USER_DOA)
    assert "barge_in" not in results


# ---------------------------------------------------------------- 방향 잠금·보정
def test_locked_direction_beats_static_sector():
    """잠금(1순위)이 살아 있으면 장착 보정(2순위)보다 우선한다."""
    fake = Fake()
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops, user_doa_center=100.0, user_doa_width=12.0)

    m.lock_user_direction(USER_DOA, now=0.0)
    m.arm_followup(now=0.0)
    m.set_speaking(True, now=0.1)
    # 보정 부채꼴(100°) 방향은 지금 기준이 아니다 — 발동 금지
    results = run_frames(m, 30, LOUD, t0=0.2, vad=True, doa=102)
    assert "barge_in" not in results
    # 잠금 방향은 발동
    results = run_frames(m, BARGE_VAD_WINDOW, LOUD, t0=3.0, vad=True, doa=USER_DOA)
    assert "barge_in" in results


def test_expired_lock_falls_back_to_static_sector():
    """잠금이 만료되면 장착 보정 부채꼴로 돌아간다."""
    fake = Fake()
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops, user_doa_center=100.0, user_doa_width=12.0)

    m.lock_user_direction(USER_DOA, now=0.0)
    later = USER_DOA_LOCK_TTL_SEC + 10.0
    m.arm_followup(now=later)
    m.set_speaking(True, now=later)
    results = run_frames(m, 30, LOUD, t0=later, vad=True, doa=USER_DOA)
    assert "barge_in" not in results       # 만료된 잠금 방향은 무효
    results = run_frames(m, BARGE_VAD_WINDOW, LOUD, t0=later + 5,
                         vad=True, doa=102)
    assert "barge_in" in results           # 보정 부채꼴 방향은 유효


def test_direction_gate_wraps_around_zero():
    """0/359 경계: 잠금 5°, 발화 356° 는 차이 9° 로 안이다."""
    fake = Fake()
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops)

    armed_speaking(m, lock=5.0)
    results = run_frames(m, BARGE_VAD_WINDOW, LOUD, t0=0.2, vad=True, doa=356)
    assert "barge_in" in results


def test_lock_ignores_missing_doa():
    m = WakewordMonitor(on_emergency=lambda e: None, on_user_text=lambda t: None,
                        predict=lambda f: {"a": 0, "b": 0}, transcribe=lambda a: "")
    m.lock_user_direction(None)
    assert m._locked_doa is None


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

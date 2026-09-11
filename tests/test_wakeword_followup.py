"""질문 후 재청취(followup) 검증 — "응" 한마디에 "비카야" 재호출이 필요 없어야 한다."""
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
    m.set_muted(True, now=0.1)
    m.set_muted(False, now=1.0)

    results = speak_answer(m, t0=1.0)
    assert results[-1] == "user_text"
    assert texts == ["응"]
    assert wakes == []


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
    """TTS 는 문장마다 상태를 깜빡인다. 두 문장짜리 질문이면 마지막 문장 끝에"""
    fake = Fake(text="아니요")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)

    m.arm_followup(now=0.0)
    m.set_muted(True, now=0.1)
    m.set_muted(False, now=1.0)
    m.set_muted(True, now=1.2)
    m.set_muted(False, now=2.5)

    results = speak_answer(m, t0=2.5)
    assert results[-1] == "user_text"
    assert texts == ["아니요"]


def test_stale_arm_does_not_open_mic_much_later():
    """질문 TTS 가 유실되면 예약이 남는다. 한참 뒤 무관한 안내가 끝난 순간"""
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
    fake = Fake(scores=[(0, 0.9), (0, 0.9)], text="멈춰")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)

    m.arm_followup(now=0.0)
    m.set_muted(True, now=0.1)
    m.set_muted(False, now=1.0)

    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD, t0=1.0)
    assert results[-1] == "emergency"
    assert len(events) == 1 and events[0].keyword == "멈춰"
    assert texts == []


def test_silent_followup_returns_quietly():
    """열렸는데 아무도 답하지 않으면 조용히 닫힌다 — 이후는 오늘과 같다"""
    fake = Fake(text="지어낸 말")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)

    m.arm_followup(now=0.0)
    m.set_muted(True, now=0.1)
    m.set_muted(False, now=1.0)

    frames = int(CONFIRM_WINDOW_SEC / 0.08) + 2
    results = run_frames(m, frames, QUIET, t0=1.0)
    assert "wake_silent" in results
    assert "user_text" not in results
    assert texts == []


class TestShortAnswerRescue:
    """짧은 답 구제 (2026-09-01, 질문 창 한정) — "네" 0.08초가 살아난다."""

    LOUD1 = np.full(1280, 3000, dtype=np.int16)
    MID = np.full(1280, 1500, dtype=np.int16)

    def _followup(self, fake, texts, states):
        m = WakewordMonitor(
            on_emergency=lambda e: None,
            on_user_text=texts.append,
            on_wake=lambda: None,
            on_listen_state=states.append,
            predict=fake.predict,
            transcribe=fake.transcribe,
        )
        m.arm_followup(now=0.0)
        m.set_muted(False, now=0.0)
        return m

    def _short_burst(self, m, frame):
        run_frames(m, 1, frame, t0=0.1, vad=True)
        return run_frames(m, 12, QUIET, t0=0.1 + 0.08)

    def test_short_loud_yes_survives(self):
        texts, states = [], []
        m = self._followup(Fake(text="네."), texts, states)
        self._short_burst(m, self.LOUD1)
        assert texts == ["네."]

    def test_short_loud_nonanswer_is_rejected(self):
        """전사가 정답 어휘가 아니면 유령 — '방2' 둔갑 방지."""
        texts, states = [], []
        m = self._followup(Fake(text="방2"), texts, states)
        self._short_burst(m, self.LOUD1)
        assert texts == []
        assert any(s.startswith("empty:short-reject") for s in states)

    def test_short_but_faint_stays_ghost(self):
        texts, states = [], []
        m = self._followup(Fake(text="네."), texts, states)
        self._short_burst(m, self.MID)
        assert texts == []
        assert any(s.startswith("empty:ghost") for s in states)

    def test_free_window_short_burst_stays_ghost(self):
        """자유 창(호출 직후)은 짧은 답 구제 없음 + 반짝 무효화(9/1) —"""
        texts, states = [], []
        fake = Fake(scores=[(0.9, 0), (0.9, 0)] + [(0, 0)] * 220, text="네.")
        m = WakewordMonitor(
            on_emergency=lambda e: None,
            on_user_text=texts.append,
            on_wake=lambda: None,
            on_listen_state=states.append,
            predict=fake.predict,
            transcribe=fake.transcribe,
        )
        run_frames(m, 2, LOUD)
        run_frames(m, 1, self.LOUD1, t0=0.3, vad=True)
        run_frames(m, 220, QUIET, t0=0.3 + 0.08)
        assert texts == []
        assert any(st.startswith("empty") for st in states)


class TestHeadRescue:
    """말했는데 빈 창 — 두 구제 (2026-09-01)."""

    def _open_followup(self, fake, texts, states, t0=10.0):
        m = make(fake, [], texts, [])
        m._on_listen_state = states.append if states is not None else (lambda s: None)
        m.arm_followup(now=t0)
        m.set_muted(False, now=t0)
        return m

    def test_vad_missing_speech_is_rescued_by_rms(self):
        """칩 VAD 전멸(None)이어도 큰 소리 발화가 전사까지 간다 — 예전엔"""
        texts, states = [], []
        fake = Fake(text="화장실로 가자")
        m = make(fake, [], texts, [])
        m.arm_followup(now=10.0)
        m.set_muted(False, now=10.0)
        run_frames(m, 6, LOUD, t0=10.1, vad=None)
        run_frames(m, 12, QUIET, t0=10.1 + 6 * 0.08, vad=None)
        assert texts == ["화장실로 가자"]

    def test_head_before_window_is_backfilled(self):
        """창 직전에 시작한 말: 머리 3프레임 소급 + 본문 1프레임이면"""
        texts = []
        captured = {}

        class F(Fake):
            def transcribe(self, audio):
                captured["n"] = len(audio)
                return "네."

        fake = F(text="네.")
        m = make(fake, [], texts, [])
        run_frames(m, 12, QUIET, t0=5.0)
        run_frames(m, 3, LOUD, t0=5.0 + 12 * 0.08)
        t_open = 5.0 + 15 * 0.08
        m.arm_followup(now=t_open)
        m.set_muted(False, now=t_open)
        run_frames(m, 1, LOUD, t0=t_open + 0.01, vad=True)
        run_frames(m, 12, QUIET, t0=t_open + 0.09)
        assert texts == ["네."]
        assert captured["n"] >= 4 * 1280

    def test_no_backfill_when_lookback_is_all_loud(self):
        """직전이 전부 시끄러우면(로봇·비카야 소리 연속) 소급하지 않는다 —"""
        texts = []
        fake = Fake(text="유령이면안됨")
        m = make(fake, [], texts, [])
        run_frames(m, 10, LOUD, t0=5.0)
        t_open = 5.0 + 10 * 0.08
        m.arm_followup(now=t_open)
        m.set_muted(False, now=t_open)
        run_frames(m, 12, QUIET, t0=t_open + 0.01, vad=None)
        assert texts == []


class TestFreeWindowNoBackfill:
    def test_wake_tail_does_not_close_free_window_early(self):
        """자유 창은 소급 금지(9/1 실기 — '비카야' 꼬리가 말 시작으로 잡혀"""
        texts = []
        fake = Fake(scores=[(0.9, 0), (0.9, 0)] + [(0, 0)] * 60,
                    text="안내소로 가자")
        m = make(fake, [], texts, [])
        run_frames(m, 2, LOUD, t0=5.0)
        run_frames(m, 20, QUIET, t0=5.16, vad=None)
        run_frames(m, 5, LOUD, t0=5.16 + 20 * 0.08, vad=True)
        run_frames(m, 12, QUIET, t0=5.16 + 25 * 0.08)
        assert texts == ["안내소로 가자"]


class TestFreeWindowMinOpen:
    def test_vad_blip_does_not_close_before_min_open(self):
        """반짝 VAD(에코·호출 꼬리) 후 침묵이 0.88초를 넘어도, 최소 개방"""
        texts = []
        fake = Fake(scores=[(0.9, 0), (0.9, 0)] + [(0, 0)] * 70,
                    text="화장실로 가자")
        m = make(fake, [], texts, [])
        run_frames(m, 2, LOUD, t0=5.0)
        run_frames(m, 1, LOUD, t0=5.16, vad=True)
        run_frames(m, 18, QUIET, t0=5.24, vad=False)
        run_frames(m, 6, LOUD, t0=5.24 + 18 * 0.08, vad=True)
        run_frames(m, 12, QUIET, t0=5.24 + 24 * 0.08, vad=False)
        assert texts == ["화장실로 가자"]

    def test_truly_silent_call_closes_after_min_open(self):
        """정말 아무 말 없으면 최소 개방이 지나고 닫힌다 — 영구 개방은 아니다"""
        texts, states = [], []
        fake = Fake(scores=[(0.9, 0), (0.9, 0)] + [(0, 0)] * 220, text="X")
        m = WakewordMonitor(
            on_emergency=lambda e: None, on_user_text=texts.append,
            on_wake=lambda: None, on_listen_state=states.append,
            predict=fake.predict, transcribe=fake.transcribe)
        run_frames(m, 2, LOUD, t0=5.0)
        run_frames(m, 1, LOUD, t0=5.16, vad=True)
        run_frames(m, 220, QUIET, t0=5.24, vad=False)
        assert texts == []
        assert any(s.startswith("empty") for s in states)

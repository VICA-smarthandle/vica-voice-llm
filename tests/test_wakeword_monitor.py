"""WakewordMonitor 의 순수 로직(process_frame) 검증. 마이크/모델/STT 불필요."""
from __future__ import annotations

import numpy as np

from src.schema import EmergencyEvent
from src.wakeword_monitor import POST_ROLL_FRAMES, WakewordMonitor

LOUD = np.full(1280, 3000, dtype=np.int16)
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
    fake = Fake(scores=[(0, 0.9), (0, 0.9)] + [(0, 0)] * 10, text="멈춤")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD)
    assert results[-1] == "reject"
    assert events == []


def test_single_spike_no_stt_call():
    fake = Fake(scores=[(0, 0.9)] + [(0, 0)] * 10, text="멈춰")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    run_frames(m, 8, LOUD)
    assert events == [] and fake.stt_calls == 0


def test_wake_then_user_text():
    speech_frames = 5
    silence_frames = 12
    fake = Fake(scores=[(0.9, 0), (0.9, 0)] + [(0, 0)] * 40, text="화장실 어디야")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    assert run_frames(m, 2, LOUD)[-1] == "wake"
    assert wakes == [1]
    run_frames(m, speech_frames, LOUD, t0=1.0, vad=True)
    results = run_frames(m, silence_frames, QUIET, t0=2.0)
    assert "user_text" in results
    assert texts == ["화장실 어디야"]
    assert events == []


def test_wake_silent_timeout():
    fake = Fake(scores=[(0.9, 0), (0.9, 0)] + [(0, 0)] * 220, text="")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    run_frames(m, 2, LOUD)
    results = run_frames(m, 220, QUIET, t0=1.0)
    assert "wake_silent" in results
    assert texts == []


def test_emergency_preempts_listen():
    fake = Fake(
        scores=[(0.9, 0), (0.9, 0), (0, 0.9), (0, 0.9)] + [(0, 0)] * 10,
        text="정지",
    )
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    run_frames(m, 2, LOUD)
    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD, t0=1.0)
    assert results[-1] == "emergency"
    assert events[0].keyword == "정지"
    assert texts == []


def test_muted_suppresses_and_failsafe_unmutes():
    fake = Fake(scores=[(0, 0.9)] * 40, text="멈춰")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    m.set_muted(True, now=0.0, failsafe_sec=10.0)
    run_frames(m, 10, LOUD, t0=0.0)
    assert events == []
    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD, t0=11.0)
    assert results[-1] == "emergency"


def test_unmute_clears_ring():
    fake = Fake(scores=[(0, 0)] * 5 + [(0, 0.9)] * 10, text="멈춰")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    m.set_muted(True, now=0.0)
    run_frames(m, 5, LOUD, t0=0.0)
    m.set_muted(False, now=0.5)
    assert len(m._ring) == 0


def test_listen_timing_breakdown():
    """계측: 대기·발화·말끝판정·STT 를 분리 기록한다."""
    import pytest

    fake = Fake(scores=[(0.9, 0), (0.9, 0)] + [(0, 0)] * 40, text="안내소로 가자")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    run_frames(m, 2, LOUD)
    run_frames(m, 5, LOUD, t0=1.0, vad=True)
    results = run_frames(m, 20, QUIET, t0=1.4)
    assert "user_text" in results
    t = m.last_listen_timing
    assert t is not None
    assert t["wait"] == pytest.approx(0.92, abs=0.01)
    assert t["speech"] == pytest.approx(0.32, abs=0.01)
    assert t["tail"] >= 0.80
    assert t["stt"] >= 0.0


def test_listen_timing_resets_on_new_window():
    """새 청취 창이 열리면 직전 계측은 비워진다 — 낡은 수치 재사용 방지."""
    fake = Fake(scores=[(0.9, 0), (0.9, 0)] + [(0, 0)] * 60, text="안내소로 가자")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    run_frames(m, 2, LOUD)
    run_frames(m, 5, LOUD, t0=1.0, vad=True)
    run_frames(m, 20, QUIET, t0=1.4)
    assert m.last_listen_timing is not None
    m.arm_followup(now=10.0)
    m.set_muted(False, now=10.0)
    assert m.last_listen_timing is None


def test_emergency_false_fire_in_listen_keeps_utterance():
    """청취 중 긴급 게이트 오발동이 긴급어가 아니면 발화를 버리지 않는다."""
    fake = Fake(
        scores=[(0.9, 0), (0.9, 0)]
        + [(0, 0)] * 5
        + [(0, 0.9), (0, 0.9)]
        + [(0, 0)] * 10,
        text="화장실로 가줘",
    )
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    run_frames(m, 2, LOUD)
    run_frames(m, 5, LOUD, t0=1.0, vad=True)
    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD, t0=1.5, vad=True)
    assert "listen_resumed" in results
    results = run_frames(m, 14, QUIET, t0=3.0)
    assert "user_text" in results
    assert texts == ["화장실로 가줘"]
    assert events == []


def test_emergency_false_fire_in_listen_still_blocks_hallucination():
    """청취 가로챔이라도 환각 단골 문구·빈 전사는 넘기지 않는다."""
    fake = Fake(
        scores=[(0.9, 0), (0.9, 0)] + [(0, 0)] * 5
        + [(0, 0.9), (0, 0.9)] + [(0, 0)] * 10,
        text="",
    )
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    run_frames(m, 2, LOUD)
    run_frames(m, 5, LOUD, t0=1.0, vad=True)
    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD, t0=1.5, vad=True)
    assert "listen_resumed" in results
    results = run_frames(m, 14, QUIET, t0=3.0)
    assert "wake_silent" in results
    assert texts == []
    assert events == []

def test_wake_window_timeout_fires_listen_empty():
    """'비카야' 창이 빈손으로 닫히면 on_listen_empty 로 알린다."""
    fake = Fake(scores=[(0.9, 0), (0.9, 0)] + [(0, 0)] * 220, text="")
    events, texts, wakes, empties = [], [], [], []
    m = WakewordMonitor(
        on_emergency=events.append,
        on_user_text=texts.append,
        on_wake=lambda: wakes.append(1),
        on_listen_empty=lambda: empties.append(1),
        predict=fake.predict,
        transcribe=fake.transcribe,
    )
    run_frames(m, 2, LOUD)
    results = run_frames(m, 220, QUIET, t0=1.0)
    assert "wake_silent" in results
    assert empties == [1]
    assert texts == []


def test_followup_window_timeout_stays_silent():
    """질문 답변용(followup) 창의 침묵은 알리지 않는다 — 무응답 처리는"""
    fake = Fake(scores=[(0, 0)] * 500, text="")
    events, texts, wakes, empties = [], [], [], []
    m = WakewordMonitor(
        on_emergency=events.append,
        on_user_text=texts.append,
        on_wake=lambda: wakes.append(1),
        on_listen_empty=lambda: empties.append(1),
        predict=fake.predict,
        transcribe=fake.transcribe,
    )
    m.arm_followup(now=0.0)
    m.set_muted(False, now=0.0)
    results = run_frames(m, 400, QUIET, t0=0.1)
    assert "wake_silent" in results
    assert empties == []


def test_confirm_window_uses_hinted_transcriber():
    """질문 답변(followup) 창은 정답 후보를 귀띔한 전사기를 쓴다."""
    fake = Fake(scores=[(0, 0)] * 100, text="긴급전사")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    m._transcribe_listen = lambda a: "일반전사"
    m._transcribe_confirm = lambda a: "그래"
    m.arm_followup(now=0.0)
    m.set_muted(False, now=0.0)
    run_frames(m, 5, LOUD, t0=0.1, vad=True)
    results = run_frames(m, 12, QUIET, t0=0.6)
    assert "user_text" in results
    assert texts == ["그래"]


def test_wake_window_ignores_confirm_transcriber():
    """'비카야' 자유 명령 창은 귀띔 없이 일반 전사기를 쓴다 — 귀띔이"""
    fake = Fake(scores=[(0.9, 0), (0.9, 0)] + [(0, 0)] * 60, text="긴급전사")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    m._transcribe_listen = lambda a: "화장실로 가자"
    m._transcribe_confirm = lambda a: "그래"
    run_frames(m, 2, LOUD)
    run_frames(m, 5, LOUD, t0=1.0, vad=True)
    results = run_frames(m, 20, QUIET, t0=1.4)
    assert "user_text" in results
    assert texts == ["화장실로 가자"]


def test_near_silence_never_reaches_stt():
    """무음에 가까운 수음은 STT 로 보내지 않는다."""
    TINY = np.full(1280, 100, dtype=np.int16)
    fake = Fake(scores=[(0.9, 0), (0.9, 0)] + [(0, 0)] * 60, text="방2")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    run_frames(m, 2, LOUD)
    run_frames(m, 1, TINY, t0=1.0, vad=True)
    results = run_frames(m, 200, TINY, t0=1.1)
    assert "wake_silent" in results
    assert texts == []
    assert fake.stt_calls == 0


class TestAgcDesiredFromEnv:
    """AGC 목표 레벨 환경변수 해석 — 칩 쓰기는 장치 시험(자동화 제외)."""

    def test_default_string_parses(self):
        from src.dsp_state import agc_desired_from_env
        assert agc_desired_from_env("0.010") == 0.010

    def test_off_values_skip_write(self):
        from src.dsp_state import agc_desired_from_env
        for raw in ("", "0", "off", "none"):
            assert agc_desired_from_env(raw) is None

    def test_garbage_skips_write(self):
        from src.dsp_state import agc_desired_from_env
        assert agc_desired_from_env("두배로") is None

    def test_out_of_range_skips_write(self):
        """말도 안 되는 값(음수·1 초과)은 칩에 쓰지 않는다."""
        from src.dsp_state import agc_desired_from_env
        assert agc_desired_from_env("-0.01") is None
        assert agc_desired_from_env("5.0") is None


def test_confirm_hint_holds_only_answers_to_questions():
    """initial_prompt 에는 **로봇이 물어서 받는 답**만 둔다."""
    from src.wakeword_monitor import CONFIRM_HINT
    for word in ["네", "아니", "기다려", "대기해", "이십", "삼십", "반시간"]:
        assert word in CONFIRM_HINT, f"질문의 답 '{word}' 가 빠졌다"
    for word in ["됐어", "그만", "고마워", "감사", "안내 끝", "필요 없"]:
        assert word not in CONFIRM_HINT, (
            f"'{word}' 는 질문의 답이 아니다 — whisper 가 이것으로 메꾼다")
    assert len(CONFIRM_HINT) < 200


def test_listen_state_relay():
    """청취 상태 중계(2026-08-30): open → speech → closed(전사 성공)."""
    states = []
    fake = Fake(scores=[(0.9, 0), (0.9, 0)] + [(0, 0)] * 40, text="십 분만 기다려")
    events, texts, wakes = [], [], []
    m = WakewordMonitor(
        on_emergency=events.append, on_user_text=texts.append,
        on_wake=lambda: wakes.append(1), predict=fake.predict,
        transcribe=fake.transcribe, on_listen_state=states.append)
    run_frames(m, 2, LOUD)
    run_frames(m, 5, LOUD, t0=1.0, vad=True)
    run_frames(m, 12, QUIET, t0=2.0)
    assert states == ["open", "speech", "closed"]


def test_listen_state_empty_on_silence():
    states = []
    fake = Fake(scores=[(0.9, 0), (0.9, 0)] + [(0, 0)] * 200, text="")
    events, texts, wakes = [], [], []
    m = WakewordMonitor(
        on_emergency=events.append, on_user_text=texts.append,
        on_wake=lambda: wakes.append(1), predict=fake.predict,
        transcribe=fake.transcribe, on_listen_state=states.append)
    run_frames(m, 2, LOUD)
    run_frames(m, 220, QUIET, t0=1.0)
    assert states == ["open", "empty"]


def test_out_of_window_answer_rescued_when_followup_armed():
    """질문 답 대기 중(followup 예약)의 창 밖 발화는 답으로 채택한다"""
    fake = Fake(scores=[(0, 0.9), (0, 0.9)] + [(0, 0)] * 10, text="필요없다구")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    m.arm_followup(now=0.0)
    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD)
    assert "user_text" in results
    assert texts == ["필요없다구"]
    assert events == []


def test_out_of_window_speech_still_rejected_without_followup():
    """예약이 없으면(행인 대화 등) 창 밖 발화는 전처럼 기각 — 규약 유지."""
    fake = Fake(scores=[(0, 0.9), (0, 0.9)] + [(0, 0)] * 10, text="필요없다구")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD)
    assert "reject" in results
    assert texts == []

"""barge-in 검증 — 로봇이 질문하는 중에 답을 시작하면 말을 끊고 들어야 한다."""
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

LOUD = np.full(1280, 3000, dtype=np.int16)
QUIET = np.zeros(1280, dtype=np.int16)

USER_DOA = 236.0
MIXED_DOA = 217.0


class Fake:
    def __init__(self, scores=(), text=""):
        self.scores = list(scores)
        self.text = text
        self.heard: list[int] = []

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
    assert fake.heard[-1] == (BARGE_VAD_WINDOW + 3 + 11) * 1280


def test_echo_never_fires_because_chip_says_no_speech():
    """자기 잔여 에코 회귀 방지: 방향이 맞아도 칩 발화 판정(vad=False)이"""
    fake = Fake()
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops)

    armed_speaking(m)
    results = run_frames(m, 30, LOUD, t0=0.2, vad=False, doa=USER_DOA)
    assert stops == []
    assert "barge_in" not in results


def test_bystander_mixed_direction_does_not_fire():
    """옆사람 오발동 회귀 방지: 발화 판정이 있어도 방향이 잠금 밖(실측 217°,"""
    fake = Fake()
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops)

    armed_speaking(m)
    results = run_frames(m, 30, LOUD, t0=0.2, vad=True, doa=MIXED_DOA)
    assert stops == []
    assert "barge_in" not in results


def test_no_direction_evidence_means_dormant():
    """방향 정보가 전혀 없으면(잠금도 보정도 없음) 발화 판정만으로는 절대"""
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
        vad = i % 3 == 0
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
    m.set_speaking(True, now=0.1)
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


def test_emergency_beats_barge_in_during_question():
    """질문 중이라도 긴급(모델 B)이 절대 우선 — 그리고 긴급에는 방향 조건이"""
    fake = Fake(scores=[(0, 0.9), (0, 0.9)], text="멈춰")
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops)

    armed_speaking(m)
    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD, t0=0.2,
                         vad=True, doa=MIXED_DOA)
    assert results[-1] == "emergency"
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
    m.set_speaking(True, now=0.6)
    results = run_frames(m, BARGE_VAD_MIN_HITS, LOUD, t0=0.7,
                         vad=True, doa=USER_DOA)
    assert "barge_in" not in results


def test_lock_and_sector_are_a_union():
    """잠금과 장착 부채꼴은 합집합 — 어느 쪽 방향이든 발동한다."""
    fake = Fake()
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops, user_doa_center=100.0, user_doa_width=12.0)

    m.lock_user_direction(USER_DOA, now=0.0)
    m.arm_followup(now=0.0)
    m.set_speaking(True, now=0.1)
    results = run_frames(m, 30, LOUD, t0=0.2, vad=True, doa=102)
    assert "barge_in" in results
    m.arm_followup(now=3.0)
    m.set_speaking(True, now=3.0)
    results = run_frames(m, 30, LOUD, t0=3.1, vad=True, doa=USER_DOA)
    assert "barge_in" in results
    m.arm_followup(now=6.0)
    m.set_speaking(True, now=6.0)
    results = run_frames(m, 30, LOUD, t0=6.1, vad=True, doa=300)
    assert "barge_in" not in results


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
    assert "barge_in" not in results
    results = run_frames(m, BARGE_VAD_WINDOW, LOUD, t0=later + 5,
                         vad=True, doa=102)
    assert "barge_in" in results


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


def test_drop_pending_keeps_emergency():
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


def test_doa_gate_off_hears_any_direction():
    """방향 관문 해제(2026-08-30) — 잠금·부채꼴 무관, 어느 방향이든 발동."""
    fake = Fake()
    events, texts, stops = [], [], []
    m = make(fake, events, texts, stops, user_doa_center=None, doa_gate=False)
    m.lock_user_direction(USER_DOA, now=0.0)
    m.arm_followup(now=0.0)
    m.set_speaking(True, now=0.1)
    results = run_frames(m, 30, LOUD, t0=0.2, vad=True, doa=60)
    assert "barge_in" in results
    m.arm_followup(now=3.0)
    m.set_speaking(True, now=3.0)
    results = run_frames(m, 30, LOUD, t0=3.1, vad=True, doa=None)
    assert "barge_in" in results


class TestRequireBothSignals:
    """칩의 두 발화 판정(SPEECHDETECTED · VOICEACTIVITY)을 곱하는 선택지."""

    def _run(self, monkeypatch, require_both: bool, vad2, min_hits: int):
        import src.wakeword_monitor as wm
        monkeypatch.setattr(wm, "BARGE_REQUIRE_BOTH", require_both)
        monkeypatch.setattr(wm, "BARGE_VAD_MIN_HITS", min_hits)
        fake = Fake(text="네")
        events, texts, stops = [], [], []
        m = make(fake, events, texts, stops)
        armed_speaking(m)
        out = []
        for i in range(wm.BARGE_VAD_WINDOW):
            out.append(m.process_frame(LOUD, now=0.2 + i * 0.08, vad=True,
                                       doa=USER_DOA, vad2=vad2))
        return out, stops

    def test_off_ignores_second_signal(self):
        """꺼져 있으면 종전 그대로 — vad2 가 없어도(None) 발동한다."""
        fake = Fake(text="네")
        events, texts, stops = [], [], []
        m = make(fake, events, texts, stops)
        armed_speaking(m)
        out = run_frames(m, BARGE_VAD_WINDOW, LOUD, t0=0.2, vad=True,
                         doa=USER_DOA)
        assert out[-1] == "barge_in"

    def test_on_requires_both(self, monkeypatch):
        """켜면 두 번째 신호가 꺼진 프레임은 세지 않는다 — 에코가 그 자리다."""
        out, stops = self._run(monkeypatch, True, vad2=False, min_hits=3)
        assert "barge_in" not in out
        assert stops == []

    def test_on_fires_when_both_present(self, monkeypatch):
        """둘 다 켜진 진짜 발화는 그대로 발동한다."""
        out, _ = self._run(monkeypatch, True, vad2=True, min_hits=3)
        assert out[-1] == "barge_in"

    def test_listen_window_never_uses_second_signal(self):
        """청취 창의 발화 판정에는 vad2 를 섞지 않는다."""
        fake = Fake(scores=[(0.9, 0.0)] * 2, text="네")
        events, texts, stops = [], [], []
        m = make(fake, events, texts, stops)
        for i in range(2):
            m.process_frame(QUIET, now=i * 0.08, vad=False, vad2=False)
        for i in range(5):
            m.process_frame(LOUD, now=0.16 + i * 0.08, vad=True, vad2=False)
        out = None
        for i in range(40):
            r = m.process_frame(QUIET, now=0.56 + i * 0.08, vad=False,
                                vad2=False)
            if r is not None:
                out = r
                break
        assert out == "user_text"
        assert texts == ["네"]


class TestBargeMissReport:
    """미달 채증 — 발동은 로그를 남기는데 **미달은 아무 흔적도 없었다.**"""

    def _make(self, states: list, **kwargs):
        fake = Fake(text="아니")
        events, texts, stops = [], [], []
        return WakewordMonitor(
            on_emergency=events.append,
            on_user_text=texts.append,
            on_barge_in=lambda: stops.append(1),
            on_listen_state=states.append,
            predict=fake.predict,
            transcribe=fake.transcribe,
            **kwargs,
        ), stops

    def test_sound_below_threshold_is_reported(self, monkeypatch):
        """문턱에 못 미치는 발화가 있었으면 단계별 숫자를 남긴다."""
        import src.wakeword_monitor as wm
        monkeypatch.setattr(wm, "BARGE_VAD_MIN_HITS", 5)
        states = []
        m, stops = self._make(states)
        armed_speaking(m)
        for i in range(BARGE_VAD_WINDOW):
            spoke = i < 4
            m.process_frame(LOUD if spoke else QUIET, now=0.2 + i * 0.08,
                            vad=spoke, doa=USER_DOA)
        m.set_speaking(False, now=1.2)
        assert stops == []
        miss = [s for s in states if s.startswith("barge-miss")]
        assert miss, "미달이 보고돼야 한다"
        assert "S=4" in miss[0] and "문턱 5" in miss[0]

    def test_silence_is_not_reported(self):
        """아무도 말하지 않았으면 남기지 않는다 — 매 질문마다 쌓이면"""
        states = []
        m, _ = self._make(states)
        armed_speaking(m)
        for i in range(BARGE_VAD_WINDOW):
            m.process_frame(QUIET, now=0.2 + i * 0.08, vad=False, doa=USER_DOA)
        m.set_speaking(False, now=1.2)
        assert not [s for s in states if s.startswith("barge-miss")]

    def test_fired_barge_in_reports_nothing(self):
        """발동했으면 미달이 아니다 — 채증은 그 자리에서 비운다."""
        states = []
        m, stops = self._make(states)
        armed_speaking(m)
        for i in range(BARGE_VAD_WINDOW):
            m.process_frame(LOUD, now=0.2 + i * 0.08, vad=True, doa=USER_DOA)
        assert stops == [1]
        m.set_speaking(False, now=1.2)
        assert not [s for s in states if s.startswith("barge-miss")]

    def test_direction_rejection_is_visible(self, monkeypatch):
        """옆사람 발화는 S 는 켜지고 방향에서 0 이 된다 — 그 대비가 남는다."""
        import src.wakeword_monitor as wm
        monkeypatch.setattr(wm, "BARGE_VAD_MIN_HITS", 5)
        states = []
        m, stops = self._make(states, user_doa_center=USER_DOA,
                              user_doa_width=15.0)
        armed_speaking(m, lock=None)
        for i in range(BARGE_VAD_WINDOW):
            m.process_frame(LOUD, now=0.2 + i * 0.08, vad=True, doa=20.0)
        m.set_speaking(False, now=1.2)
        assert stops == []
        miss = [s for s in states if s.startswith("barge-miss")]
        assert miss and "S=10" in miss[0] and "방향=0" in miss[0]

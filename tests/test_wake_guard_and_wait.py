"""'비카야' 오감지 대책 (2026-10-08 사용자 결정)."""
from __future__ import annotations

import numpy as np

from src.wakeword_monitor import (
    LISTEN_START_SEC, SELF_WAKE_TAIL_SEC, WAKE_THRESHOLD, WakewordMonitor,
    has_wake_word, wake_threshold_from_env,
)

LOUD = np.full(1280, 3000, dtype=np.int16)
QUIET = np.zeros(1280, dtype=np.int16)
FRAME_SEC = 0.08


class Fake:
    def __init__(self, scores, text=""):
        self.scores = list(scores)
        self.text = text

    def predict(self, _frame):
        a, b = self.scores.pop(0) if self.scores else (0.0, 0.0)
        return {"a": a, "b": b}

    def transcribe(self, _audio):
        return self.text


def _make(fake, **kw):
    log = {"texts": [], "wakes": [], "scores": [], "empties": []}
    m = WakewordMonitor(
        on_emergency=lambda e: None,
        on_user_text=log["texts"].append,
        on_wake=lambda: log["wakes"].append(1),
        on_listen_empty=lambda: log["empties"].append(1),
        on_wake_score=lambda kind, peak: log["scores"].append((kind, round(peak, 2))),
        predict=fake.predict,
        transcribe=fake.transcribe,
        **kw,
    )
    return m, log


def _feed(m, frames, t0):
    """frames = [(frame, vad)] 를 80ms 간격으로 넣고 (시각, 결과) 목록을 돌려준다."""
    out = []
    for i, (frame, vad) in enumerate(frames):
        t = t0 + i * FRAME_SEC
        out.append((t, m.process_frame(frame, now=t, vad=vad)))
    return out


def test_start_wait_is_six_seconds():
    assert LISTEN_START_SEC == 6.0


def test_free_window_closes_when_nobody_starts_within_start_wait():
    m, log = _make(Fake([(0.9, 0), (0.9, 0)]))
    _feed(m, [(LOUD, None)] * 2, 0.0)
    out = _feed(m, [(QUIET, False)] * 120, 0.16)
    closed = [t for t, r in out if r == "wake_silent"]
    assert closed, "6초 동안 말이 없으면 닫혀야 한다"
    assert abs(closed[0] - (0.08 + LISTEN_START_SEC)) < 0.1
    assert log["empties"] == [1]


def test_speech_started_before_start_wait_is_heard_to_the_end():
    m, log = _make(Fake([(0.9, 0), (0.9, 0)], text="화장실로 가자"))
    _feed(m, [(LOUD, None)] * 2, 0.0)
    frames = ([(QUIET, False)] * 68 + [(LOUD, True)] * 25 + [(QUIET, False)] * 13)
    out = _feed(m, frames, 0.16)
    assert "user_text" in [r for _, r in out]
    assert log["texts"] == ["화장실로 가자"]


def test_followup_window_ignores_start_wait():
    m, log = _make(Fake([]))
    m.arm_followup(now=0.0)
    m.set_speaking(False, now=0.1)
    out = _feed(m, [(QUIET, False)] * 100, 0.2)
    assert "wake_silent" not in [r for _, r in out]
    assert m._state == "listen"


def test_default_threshold_is_point_seven():
    assert WAKE_THRESHOLD == 0.7
    m, _ = _make(Fake([]))
    assert m.gate_a.threshold == 0.7
    assert m.gate_a_listen.threshold == 0.7


def test_threshold_env_switch(monkeypatch):
    monkeypatch.setenv("VICA_WAKE_THRESHOLD", "0.65")
    assert wake_threshold_from_env() == 0.65
    for bad in ("", "abc", "0", "1.5"):
        monkeypatch.setenv("VICA_WAKE_THRESHOLD", bad)
        assert wake_threshold_from_env() == 0.7
    monkeypatch.delenv("VICA_WAKE_THRESHOLD")
    assert wake_threshold_from_env() == 0.7


def test_score_between_old_and_new_threshold_no_longer_wakes():
    m, log = _make(Fake([(0.65, 0), (0.65, 0), (0.0, 0)]))
    out = _feed(m, [(LOUD, None)] * 3, 0.0)
    assert "wake" not in [r for _, r in out]
    assert log["wakes"] == []


def test_fire_reports_peak_score():
    m, log = _make(Fake([(0.5, 0), (0.75, 0), (0.82, 0)]))
    _feed(m, [(LOUD, None)] * 3, 0.0)
    assert log["wakes"] == [1]
    assert log["scores"] == [("fire", 0.82)]


def test_near_miss_reports_peak_once_when_sound_ends():
    m, log = _make(Fake([(0.45, 0), (0.65, 0), (0.62, 0), (0.1, 0), (0.0, 0)]))
    _feed(m, [(LOUD, None)] * 5, 0.0)
    assert log["wakes"] == []
    assert log["scores"] == [("near", 0.65)]


def test_single_high_frame_is_a_near_miss():
    m, log = _make(Fake([(0.9, 0), (0.0, 0)]))
    _feed(m, [(LOUD, None)] * 2, 0.0)
    assert log["scores"] == [("near", 0.9)]


def test_low_scores_are_not_reported():
    m, log = _make(Fake([(0.3, 0), (0.35, 0), (0.0, 0)]))
    _feed(m, [(LOUD, None)] * 3, 0.0)
    assert log["scores"] == []


def test_has_wake_word():
    assert has_wake_word("돌아오시면 '비카야'라고 말씀해 주세요.")
    assert not has_wake_word("비카가 대기 중입니다.")
    assert not has_wake_word("")


def test_robot_saying_wake_word_does_not_wake():
    m, log = _make(Fake([(0.9, 0), (0.9, 0), (0.0, 0)]))
    m.note_robot_speech("돌아오시면 '비카야'라고 말씀해 주세요.", now=0.0)
    m.set_speaking(True, now=0.0)
    out = _feed(m, [(LOUD, None)] * 3, 0.1)
    assert "wake" not in [r for _, r in out]
    assert log["wakes"] == []
    assert log["scores"] == [("self", 0.9)]


def test_guard_holds_for_tail_then_releases():
    m, log = _make(Fake([(0.9, 0), (0.9, 0), (0.0, 0), (0.9, 0), (0.9, 0)]))
    m.note_robot_speech("'비카야'라고 불러주세요.", now=0.0)
    m.set_speaking(True, now=0.0)
    m.set_speaking(False, now=1.0)
    _feed(m, [(LOUD, None)] * 3, 1.0 + SELF_WAKE_TAIL_SEC - 0.3)
    assert log["wakes"] == []
    out = _feed(m, [(LOUD, None)] * 2, 1.0 + SELF_WAKE_TAIL_SEC + 0.5)
    assert "wake" in [r for _, r in out]


def test_other_robot_sentences_do_not_guard():
    m, log = _make(Fake([(0.9, 0), (0.9, 0)]))
    m.note_robot_speech("비카가 대기 중입니다.", now=0.0)
    m.set_speaking(True, now=0.0)
    out = _feed(m, [(LOUD, None)] * 2, 0.1)
    assert "wake" in [r for _, r in out]


def test_next_sentence_without_wake_word_ends_guard_with_tail():
    m, log = _make(Fake([(0.9, 0), (0.9, 0)]))
    m.note_robot_speech("'비카야'라고 불러주세요.", now=0.0)
    m.note_robot_speech("어디로 가고 싶으신가요?", now=2.0)
    out = _feed(m, [(LOUD, None)] * 2, 2.0 + SELF_WAKE_TAIL_SEC + 0.1)
    assert "wake" in [r for _, r in out]


def test_in_window_wake_observation_also_ignores_robot_voice():
    m, log = _make(Fake([(0.9, 0), (0.9, 0)] + [(0.0, 0)] * 3 + [(0.9, 0), (0.9, 0)]))
    _feed(m, [(LOUD, None)] * 2, 0.0)
    m.note_robot_speech("'비카야'라고 불러주세요.", now=0.3)
    _feed(m, [(LOUD, True)] * 5, 0.3)
    assert m._listen_heard_wake is False


def test_start_wait_counts_from_end_of_robot_speech():
    m, log = _make(Fake([(0.9, 0), (0.9, 0)], text="화장실로 가자"))
    m.set_speaking(True, now=0.0)
    _feed(m, [(LOUD, None)] * 2, 0.0)
    out1 = _feed(m, [(QUIET, False)] * 62, 0.16)
    m.set_speaking(False, now=5.2)
    out2 = _feed(m, [(QUIET, False)] * 62, 5.2)
    out3 = _feed(m, [(LOUD, True)] * 15 + [(QUIET, False)] * 12, 10.2)
    assert "wake_silent" not in [r for _, r in out1 + out2]
    assert log["texts"] == ["화장실로 가자"]


def test_start_wait_expires_six_seconds_after_robot_speech_ends():
    m, log = _make(Fake([(0.9, 0), (0.9, 0)]))
    m.set_speaking(True, now=0.0)
    _feed(m, [(LOUD, None)] * 2, 0.0)
    _feed(m, [(QUIET, False)] * 30, 0.16)
    m.set_speaking(False, now=2.6)
    out = _feed(m, [(QUIET, False)] * 100, 2.6)
    closed = [t for t, r in out if r == "wake_silent"]
    assert closed and abs(closed[0] - (2.6 + LISTEN_START_SEC)) < 0.1


def test_guard_gives_up_if_speech_end_never_arrives():
    from src.wakeword_monitor import SELF_WAKE_MAX_SEC
    m, log = _make(Fake([(0.9, 0), (0.9, 0)]))
    m.note_robot_speech("'비카야'라고 불러주세요.", now=0.0)
    out = _feed(m, [(LOUD, None)] * 2, SELF_WAKE_MAX_SEC + 0.1)
    assert "wake" in [r for _, r in out]


def test_emergency_still_fires_while_robot_says_wake_word():
    from src.wakeword_monitor import POST_ROLL_FRAMES
    events = []
    fake = Fake([(0, 0.9), (0, 0.9)] + [(0, 0)] * POST_ROLL_FRAMES, text="멈춰!")
    m = WakewordMonitor(on_emergency=events.append, on_user_text=lambda t: None,
                        predict=fake.predict, transcribe=fake.transcribe)
    m.note_robot_speech("'비카야'라고 불러주세요.", now=0.0)
    m.set_speaking(True, now=0.0)
    out = _feed(m, [(LOUD, None)] * (1 + POST_ROLL_FRAMES), 0.1)
    assert out[-1][1] == "emergency"
    assert len(events) == 1


def test_mute_mode_unmute_ends_guard_and_starts_wait_clock():
    m, log = _make(Fake([(0.9, 0), (0.9, 0)]))
    m.note_robot_speech("'비카야'라고 불러주세요.", now=0.0)
    m.set_muted(True, now=0.0)
    m.set_muted(False, now=3.0)
    out = _feed(m, [(LOUD, None)] * 2, 3.0 + SELF_WAKE_TAIL_SEC + 0.1)
    assert "wake" in [r for _, r in out]
    opened = 3.0 + SELF_WAKE_TAIL_SEC + 0.18
    out = _feed(m, [(QUIET, False)] * 90, opened + 0.08)
    closed = [t for t, r in out if r == "wake_silent"]
    assert closed and abs(closed[0] - (opened + LISTEN_START_SEC)) < 0.1


def test_score_callback_error_does_not_block_wake():
    fake = Fake([(0.9, 0), (0.9, 0)])
    wakes = []

    def broken(kind, peak):
        raise RuntimeError("log failed")

    m = WakewordMonitor(on_emergency=lambda e: None, on_user_text=lambda t: None,
                        on_wake=lambda: wakes.append(1), on_wake_score=broken,
                        predict=fake.predict, transcribe=fake.transcribe)
    out = _feed(m, [(LOUD, None)] * 2, 0.0)
    assert out[-1][1] == "wake" and wakes == [1]

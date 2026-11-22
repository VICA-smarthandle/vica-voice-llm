"""Task N2b — "네?" 직후 긴급 관문 건너뛰기: 로봇이 서 있는 상태에서만 (2026-11-22 사용자 결정, 원칙 13 · 8).

로봇의 호출 응답 "네?" 소리가 긴급 모델(gate_b)을 넘겨, 확인 받아쓰기(막힘) 동안 사용자의 이어진 말이 잘렸다
(11-22 실기 세 번 중 두 번 유실, 둘 다 idle). "네?" 재생 시작(note_robot_speech) 뒤 2.5 s 동안, 미션 dialog_state 가
서 있는 여섯 값이고 그 상태를 받은 지 3 s 안일 때만 관문에 점수를 넣지 않고 비운다. 셋 중 하나라도 아니면 지금과 같다.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import wakeword_monitor as wm
from src.replies import WAKE_GREETING
from src.wakeword_monitor import WakewordMonitor

FRAME = np.zeros(1280, dtype=np.int16)
DT = 0.08
SKIP_SEC = 2.5      # 설명서 N2 요구 1
FRESH_SEC = 3.0     # 설명서 N2b C-3
STANDING = ("idle", "confirming", "asking_next", "asking_wait_time", "waiting", "paused")
OTHERS = ("navigating", "returning", "approaching", "turning", "seeking", "moving_to_wait_spot",
          "waiting_release", "paused_handle", "grip_wait", "awaiting_user", "estopped", "", "unknown_state")
HIGH = 0.95


class Scores:
    """predict 가짜 — 넣을 점수를 그때그때 정한다."""

    def __init__(self):
        self.a = 0.0
        self.b = 0.0

    def predict(self, _frame):
        return {"a": self.a, "b": self.b}


def _make(single=None, lines=None, near=None):
    sc = Scores()
    events = []
    m = WakewordMonitor(
        on_emergency=events.append, on_user_text=lambda t: None,
        on_emergency_score=near.append if near is not None else None,
        predict=sc.predict, transcribe=lambda a: "멈춰",
        gate_a=0.7, gate_b=0.5, single_b=single,
        log=lines.append if lines is not None else (lambda s: None))
    return m, sc, events


def _open_listen(m, sc, t):
    """호출 두 조각(t-0.16·t-0.08)으로 자유 창을 연다. 열린 뒤 호출 점수는 0."""
    sc.a = 0.99
    m.process_frame(FRAME, now=t - 0.16)
    assert m.process_frame(FRAME, now=t - 0.08) == "wake"
    sc.a = 0.0
    assert m.state == "listen"


def _feed(m, sc, b, times, vad=None):
    """같은 긴급 점수 b 를 times 에 넣고 (결과, 상태) 목록을 돌려준다. 상태가 postroll 이면 그 조각에서 발동한 것이다."""
    sc.b = b
    return [(m.process_frame(FRAME, now=t, vad=vad), m.state) for t in times]


def _times(start, n):
    return [round(start + i * DT, 4) for i in range(n)]


def _run(state, greet, mode="idle", single=None):
    """상태와 greet 문장 소식을 1.0 에 받은 뒤 1.00~3.40 s 내내 문턱을 넘는 긴급 점수를 넣는다.

    1.0 에서 시작하는 것은 청취 모드의 호출 두 조각(0.84·0.92)이 0 보다 뒤여야 해서다 — 자기 목소리 무시의
    꼬리 시각 초깃값이 0.0 이다."""
    m, sc, events = _make(single=single)
    if mode == "listen":
        _open_listen(m, sc, 1.0)
    if state is not None:
        m.set_dialog_state(state, now=1.0)
    if greet is not None:
        m.note_robot_speech(greet, now=1.0)
    out = _feed(m, sc, HIGH, _times(1.0, 31), vad=True if mode == "listen" else None)
    return out, events


def test_p13_constants():
    assert wm.GREETING_GATE_SKIP_SEC == SKIP_SEC
    assert wm.GREETING_GATE_STATE_FRESH_SEC == FRESH_SEC
    assert wm.GREETING_GATE_SKIP_STATES == frozenset(STANDING)


@pytest.mark.parametrize("single", (None, 0.5))
@pytest.mark.parametrize("state", STANDING)
def test_p13_standing_state_skips_gate_after_greeting(state, single):
    out, events = _run(state, WAKE_GREETING, single=single)
    assert all(r is None and s == "idle" for r, s in out)
    assert events == []


@pytest.mark.parametrize("state", STANDING)
def test_p13_standing_state_skip_keeps_listening(state):
    m, sc, events = _make()
    _open_listen(m, sc, 1.0)
    m.set_dialog_state(state, now=1.0)
    m.note_robot_speech(WAKE_GREETING, now=1.0)
    times = _times(1.0, 31)
    out = _feed(m, sc, HIGH, times, vad=True)
    assert all(r is None and s == "listen" for r, s in out)   # 청취가 끊기지 않는다
    assert len(m._collect) == len(times)                       # 모은 말소리 조각이 그대로
    assert events == []


@pytest.mark.parametrize("mode", ("idle", "listen"))
@pytest.mark.parametrize("state", OTHERS)
def test_p13_other_states_fire_as_before(state, mode):
    got, events = _run(state, WAKE_GREETING, mode)
    base, base_events = _run(state, None, mode)
    assert got == base                                         # "네?" 소식이 없던 때와 같은 조각에서 발동
    assert any(s == "postroll" for _, s in got)
    assert len(events) == len(base_events) >= 1


@pytest.mark.parametrize("mode", ("idle", "listen"))
def test_p13_no_state_received_fires_as_before(mode):
    got, _ = _run(None, WAKE_GREETING, mode)
    base, _ = _run(None, None, mode)
    assert got == base
    assert any(s == "postroll" for _, s in got)


def test_p13_stale_state_fires_as_before():
    """idle 을 받은 지 3 s 가 넘었다(로봇 상태 소식이 끊김) — "네?" 뒤 2.5 s 안이어도 지금처럼 발동한다."""
    def run(greet):
        m, sc, _ = _make()
        m.set_dialog_state("idle", now=0.0)
        if greet:
            m.note_robot_speech(WAKE_GREETING, now=1.0)
        return _feed(m, sc, HIGH, _times(3.04, 5))   # "네?" 뒤 2.04~2.36 s, 상태 받은 지 3.04 s 넘음
    assert run(True) == run(False)
    assert any(s == "postroll" for _, s in run(True))


def test_p13_state_fresh_until_3s_then_gate_counts_from_empty():
    m, sc, _ = _make()
    m.set_dialog_state("idle", now=0.0)
    m.note_robot_speech(WAKE_GREETING, now=1.0)                # 3.5 까지 건너뛸 시간
    out = _feed(m, sc, HIGH, _times(2.6, 5))                   # 2.60~2.92 — 상태가 아직 신선하다
    assert all(s == "idle" for _, s in out)
    out = _feed(m, sc, HIGH, _times(3.04, 2))                  # 상태가 낡았다 — 빈 관문에서 2연속
    assert [s for _, s in out] == ["idle", "postroll"]


@pytest.mark.parametrize("single, fire_at", ((None, 1), (0.5, 0)))
def test_p13_state_turns_moving_mid_window_fires_from_next_frame(single, fire_at):
    m, sc, _ = _make(single=single)
    m.set_dialog_state("idle", now=0.0)
    m.note_robot_speech(WAKE_GREETING, now=0.0)
    out = _feed(m, sc, HIGH, _times(0.0, 10))                  # 0.00~0.72 건너뜀
    assert all(s == "idle" for _, s in out)
    m.set_dialog_state("navigating", now=0.75)
    out = _feed(m, sc, HIGH, _times(0.8, 2))
    assert [s for _, s in out].index("postroll") == fire_at   # 바뀐 뒤 조각부터 지금 규칙대로


def test_p13_state_turns_standing_mid_window_skips_from_next_frame():
    m, sc, events = _make()
    m.set_dialog_state("navigating", now=0.0)
    m.note_robot_speech(WAKE_GREETING, now=0.0)
    assert _feed(m, sc, HIGH, [0.0]) == [(None, "idle")]       # 한 조각 — 2연속 전
    m.set_dialog_state("idle", now=0.05)
    out = _feed(m, sc, HIGH, _times(0.08, 28))                 # 0.08~2.24 — 지금이면 0.08 에 발동
    assert all(r is None and s == "idle" for r, s in out)
    assert events == []


def test_p13_after_window_gate_starts_empty():
    """2.5 s 가 지나면 지금처럼 발동하되, "네?" 직전에 쌓인 지속 조각도 이어지지 않는다(관문을 비운다)."""
    m, sc, _ = _make()
    m.set_dialog_state("idle", now=-0.1)
    assert _feed(m, sc, HIGH, [-0.08]) == [(None, "idle")]     # "네?" 직전 문턱 넘은 한 조각(2연속 전)
    m.note_robot_speech(WAKE_GREETING, now=0.0)
    out = _feed(m, sc, HIGH, _times(0.0, 31))                  # 0.00~2.40 건너뜀
    assert all(s == "idle" for _, s in out)
    out = _feed(m, sc, HIGH, _times(2.56, 2))                  # 시간이 지났다 — 첫 조각은 1연속뿐
    assert [s for _, s in out] == ["idle", "postroll"]


@pytest.mark.parametrize("text", ("화장실로 안내해드릴까요?", "네, 말씀하세요.", "네."))
def test_p13_other_sentences_do_not_start_skip(text):
    got, _ = _run("idle", text)
    base, _ = _run("idle", None)
    assert got == base
    assert any(s == "postroll" for _, s in got)


def test_p13_greeting_with_surrounding_spaces_starts_skip():
    out, events = _run("idle", f"  {WAKE_GREETING} \n")
    assert all(s == "idle" for _, s in out)
    assert events == []


def test_p13_other_sentence_keeps_window():
    """"네?" 뒤의 다른 문장 소식은 건너뛸 시간을 줄이지도 늘리지도 않는다."""
    m, sc, _ = _make()
    m.set_dialog_state("idle", now=0.0)
    m.note_robot_speech(WAKE_GREETING, now=0.0)
    out = _feed(m, sc, HIGH, _times(0.0, 13))                  # 0.00~0.96
    m.note_robot_speech("화장실로 안내해드릴까요?", now=1.0)
    out += _feed(m, sc, HIGH, _times(1.04, 18))                # 1.04~2.40
    assert all(s == "idle" for _, s in out)
    out = _feed(m, sc, HIGH, _times(2.56, 2))
    assert [s for _, s in out] == ["idle", "postroll"]


def test_p13_second_greeting_restarts_window():
    m, sc, _ = _make()
    m.set_dialog_state("idle", now=0.0)
    m.note_robot_speech(WAKE_GREETING, now=0.0)
    _feed(m, sc, 0.0, _times(0.0, 25))                         # 0.00~1.92 조용
    m.set_dialog_state("idle", now=2.0)
    m.note_robot_speech(WAKE_GREETING, now=2.0)                # 4.5 까지 다시
    out = _feed(m, sc, HIGH, _times(2.56, 24))                 # 2.56~4.40 — 첫 "네?" 만이면 2.64 에 발동
    assert all(s == "idle" for _, s in out)
    out = _feed(m, sc, HIGH, _times(4.56, 2))
    assert [s for _, s in out] == ["idle", "postroll"]


def test_p13_one_log_line_per_greeting():
    lines, near = [], []
    m, sc, _ = _make(lines=lines, near=near)
    m.set_dialog_state("idle", now=0.0)
    m.note_robot_speech(WAKE_GREETING, now=0.0)
    _feed(m, sc, 0.6, _times(0.0, 5))
    _feed(m, sc, HIGH, _times(0.4, 5))
    _feed(m, sc, 0.2, _times(0.8, 21))                         # ~2.40
    assert lines == []                                         # 아직 건너뛰는 중 — 끝나야 한 줄
    _feed(m, sc, 0.0, _times(2.56, 3))
    assert lines == ["긴급 관문 건너뜀('네?' 직후 2.5 s · 상태 idle): 최고 점수 0.95"]
    assert near == []                                          # 건너뛴 점수로 '놓침' 기록을 따로 남기지 않는다
    # 문턱을 넘은 점수가 없던 "네?"는 줄을 남기지 않는다
    m.set_dialog_state("waiting", now=3.0)
    m.note_robot_speech(WAKE_GREETING, now=3.0)
    _feed(m, sc, 0.3, _times(3.0, 31))
    _feed(m, sc, 0.0, _times(5.6, 2))
    assert len(lines) == 1


def test_p13_log_line_each_greeting_when_greeting_repeats():
    lines = []
    m, sc, _ = _make(lines=lines)
    m.set_dialog_state("confirming", now=0.0)
    m.note_robot_speech(WAKE_GREETING, now=0.0)
    _feed(m, sc, 0.7, _times(0.0, 10))                         # 0.00~0.72
    m.note_robot_speech(WAKE_GREETING, now=0.8)                # 두 번째 "네?" — 앞 "네?"의 줄이 먼저
    _feed(m, sc, 0.9, _times(0.8, 2))
    assert lines == ["긴급 관문 건너뜀('네?' 직후 2.5 s · 상태 confirming): 최고 점수 0.70"]
    _feed(m, sc, 0.9, _times(0.96, 25))                        # ~2.88
    _feed(m, sc, 0.0, _times(3.36, 2))                         # 3.3 이 지났다
    assert lines[1:] == ["긴급 관문 건너뜀('네?' 직후 2.5 s · 상태 confirming): 최고 점수 0.90"]

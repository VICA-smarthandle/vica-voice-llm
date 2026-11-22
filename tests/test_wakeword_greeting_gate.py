"""Task N2b — "네?" 직후 긴급 관문 건너뛰기: 로봇이 서 있는 상태에서만 (2026-11-22 사용자 결정, 원칙 13 · 8).

로봇의 호출 응답 "네?" 소리가 긴급 모델(gate_b)을 넘겨, 확인 받아쓰기(막힘) 동안 사용자의 이어진 말이 잘렸다
(11-22 실기 세 번 중 두 번 유실, 둘 다 idle). "네?" 재생 시작(note_robot_speech) 때 미션 dialog_state 가 서 있는 다섯 값으로
GREETING_GATE_SETTLED_SEC(3 s) 넘게 끊기지 않고 이어져 왔고 마지막 소식이 3 s 안이면, 그 뒤 2.5 s 동안 조각마다
(상태가 다섯 값 · 3 s 안) 관문에 점수를 넣지 않고 비운다. 하나라도 아니면 지금과 같다.
검토 수정(C1·I1·Minor): confirming 은 빼고(주행 중 바꾸기 질문의 "비카야" = 재출발), "네?" 때 충분히 서 있었어야 하며,
시각이 뒤로 가면(음수) 신선하지 않다.
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
SETTLED_SEC = 3.0   # 검토 I1
STANDING = ("idle", "asking_next", "asking_wait_time", "waiting", "paused")
OTHERS = ("confirming", "navigating", "returning", "approaching", "turning", "seeking", "moving_to_wait_spot",
          "waiting_release", "paused_handle", "grip_wait", "awaiting_user", "estopped", "", "unknown_state",
          "IDLE", " idle", "idle ")   # 글자까지 같아야 한다(검토 Minor 2)
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


def _settle(m, state, t, sec=SETTLED_SEC):
    """t 까지 state 를 sec 동안 1 Hz 로 받아 왔다(t-sec, …, t-1, t)."""
    for i in range(int(round(sec)), -1, -1):
        m.set_dialog_state(state, now=round(t - i, 4))


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


def _postroll_at(out, times):
    """처음 발동한 조각의 시각(없으면 None)."""
    return next((t for t, (_, s) in zip(times, out) if s == "postroll"), None)


def _run(state, greet, mode="idle", single=None):
    """state 를 1.0 까지 3 s 동안 받아 오고 greet 문장 소식을 1.0 에 받은 뒤 1.00~3.40 s 내내 문턱을 넘는 긴급 점수를 넣는다.

    1.0 에서 시작하는 것은 청취 모드의 호출 두 조각(0.84·0.92)이 0 보다 뒤여야 해서다 — 자기 목소리 무시의
    꼬리 시각 초깃값이 0.0 이다."""
    m, sc, events = _make(single=single)
    if state is not None:
        _settle(m, state, 1.0)
    if mode == "listen":
        _open_listen(m, sc, 1.0)
    if greet is not None:
        m.note_robot_speech(greet, now=1.0)
    out = _feed(m, sc, HIGH, _times(1.0, 31), vad=True if mode == "listen" else None)
    return out, events


def _scene(news, greet_at, start, n, single=None):
    """상태 소식 news=[(값, 시각)] 뒤 greet_at 에 "네?"(None 이면 없음), start 부터 n 조각 문턱 넘는 점수. (결과, 상태) 목록."""
    m, sc, _ = _make(single=single)
    for state, t in news:
        m.set_dialog_state(state, now=t)
    if greet_at is not None:
        m.note_robot_speech(WAKE_GREETING, now=greet_at)
    return _feed(m, sc, HIGH, _times(start, n))


def _fires_as_before(news, greet_at, start=0.0, n=31):
    got = _scene(news, greet_at, start, n)
    assert got == _scene(news, None, start, n)       # "네?" 소식이 없던 때와 같은 조각에서 발동
    assert any(s == "postroll" for _, s in got)


def _skipped(news, greet_at, start=0.0, n=31):
    got = _scene(news, greet_at, start, n)
    assert all(r is None and s == "idle" for r, s in got)


def test_p13_constants():
    assert wm.GREETING_GATE_SKIP_SEC == SKIP_SEC
    assert wm.GREETING_GATE_STATE_FRESH_SEC == FRESH_SEC
    assert wm.GREETING_GATE_SETTLED_SEC == SETTLED_SEC
    assert wm.GREETING_GATE_SKIP_STATES == frozenset(STANDING)
    assert "confirming" not in wm.GREETING_GATE_SKIP_STATES


@pytest.mark.parametrize("single", (None, 0.5))
@pytest.mark.parametrize("state", STANDING)
def test_p13_standing_state_skips_gate_after_greeting(state, single):
    out, events = _run(state, WAKE_GREETING, single=single)
    assert all(r is None and s == "idle" for r, s in out)
    assert events == []


@pytest.mark.parametrize("state", STANDING)
def test_p13_standing_state_skip_keeps_listening(state):
    m, sc, events = _make()
    _settle(m, state, 1.0)
    _open_listen(m, sc, 1.0)
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


@pytest.mark.parametrize("since", ((-0.98,), (-3.98, -2.98, -1.98, -0.98)))
def test_p13_c1_confirming_wake_restart_fires_as_before(since):
    """검토 C1 — 주행 중 바꾸기 질문(confirming) 중 "비카야"는 "네?"와 함께 원래 목적지로 재출발한다
    (mission_logic.py 5378–5379 → 5416 _fold_confirming(announce=False)). 로봇 상태는 1 Hz 뒤에야 navigating 이 된다.
    confirming 을 얼마나 오래 받아 왔든 "멈춰"는 지금과 같은 조각에서 긴급이다."""
    def run(greet):
        events = []
        sc = {"b": 0.0}
        m = WakewordMonitor(on_emergency=events.append, on_user_text=lambda t: None,
                            predict=lambda f: {"a": 0.0, "b": sc["b"]}, transcribe=lambda a: "멈춰",
                            gate_a=0.7, gate_b=0.5, single_b=None, log=lambda s: None)
        for t in since:
            m.set_dialog_state("confirming", now=t)
        if greet:
            m.note_robot_speech(WAKE_GREETING, now=0.0)
        out = []
        for t in _times(0.0, 20):                              # 0.00~1.52
            if t == 0.96:
                m.set_dialog_state("navigating", now=0.96)
            sc["b"] = 0.9 if 0.39 < t < 0.57 else 0.0          # 0.40~0.56 진짜 "멈춰" 세 조각
            out.append((t, m.process_frame(FRAME, now=t)))
        return out, len(events)
    got, n_got = run(True)
    base, n_base = run(False)
    assert got == base
    assert n_got == n_base == 1


def test_p13_stale_state_fires_as_before():
    """idle 을 받은 지 3 s 가 넘었다(로봇 상태 소식이 끊김) — "네?" 뒤 2.5 s 안이어도 지금처럼 발동한다."""
    news = [("idle", t) for t in (-3.0, -2.0, -1.0, 0.0)]
    _fires_as_before(news, 1.0, start=3.04, n=5)               # "네?" 뒤 2.04~2.36 s, 상태 받은 지 3.04 s 넘음


def test_p13_state_fresh_until_3s_then_gate_counts_from_empty():
    m, sc, _ = _make()
    _settle(m, "idle", 0.0)
    m.note_robot_speech(WAKE_GREETING, now=1.0)                # 3.5 까지 건너뛸 시간
    out = _feed(m, sc, HIGH, _times(2.6, 5))                   # 2.60~2.92 — 상태가 아직 신선하다
    assert all(s == "idle" for _, s in out)
    out = _feed(m, sc, HIGH, _times(3.04, 2))                  # 상태가 낡았다 — 빈 관문에서 2연속
    assert [s for _, s in out] == ["idle", "postroll"]


@pytest.mark.parametrize("single, fire_at", ((None, 1), (0.5, 0)))
def test_p13_state_turns_moving_mid_window_fires_from_next_frame(single, fire_at):
    m, sc, _ = _make(single=single)
    _settle(m, "idle", 0.0)
    m.note_robot_speech(WAKE_GREETING, now=0.0)
    out = _feed(m, sc, HIGH, _times(0.0, 10))                  # 0.00~0.72 건너뜀
    assert all(s == "idle" for _, s in out)
    m.set_dialog_state("navigating", now=0.75)
    out = _feed(m, sc, HIGH, _times(0.8, 2))
    assert [s for _, s in out].index("postroll") == fire_at   # 바뀐 뒤 조각부터 지금 규칙대로


def test_p13_state_back_to_standing_mid_window_skips_again():
    """조각마다 보는 조건은 그대로다 — 건너뛸 자격이 있던 "네?"의 2.5 s 안에서 목록 밖 값 뒤 다시 목록 값이 오면 그 뒤 조각은 건너뛴다."""
    m, sc, events = _make()
    _settle(m, "idle", 0.0)
    m.note_robot_speech(WAKE_GREETING, now=0.0)
    m.set_dialog_state("navigating", now=0.05)
    assert _feed(m, sc, HIGH, [0.08]) == [(None, "idle")]      # 한 조각 — 2연속 전
    m.set_dialog_state("idle", now=0.1)
    out = _feed(m, sc, HIGH, _times(0.16, 28))                 # 0.16~2.32 — 지금이면 0.16 에 발동
    assert all(r is None and s == "idle" for r, s in out)
    assert events == []


def test_p13_after_window_gate_starts_empty():
    """2.5 s 가 지나면 지금처럼 발동하되, "네?" 직전에 쌓인 지속 조각도 이어지지 않는다(관문을 비운다)."""
    m, sc, _ = _make()
    _settle(m, "idle", -0.1)
    assert _feed(m, sc, HIGH, [-0.08]) == [(None, "idle")]     # "네?" 직전 문턱 넘은 한 조각(2연속 전)
    m.note_robot_speech(WAKE_GREETING, now=0.0)
    out = _feed(m, sc, HIGH, _times(0.0, 31))                  # 0.00~2.40 건너뜀
    assert all(s == "idle" for _, s in out)
    out = _feed(m, sc, HIGH, _times(2.56, 2))                  # 시간이 지났다 — 첫 조각은 1연속뿐
    assert [s for _, s in out] == ["idle", "postroll"]


def test_p13_skip_keeps_cooldown():
    """관문을 비워도 쿨다운은 지우지 않는다(FrameGate.reset 은 연속 조각만 지운다) — 건너뛰기 직전의 발동 뒤 쉬는 시간이 남는다."""
    m, sc, events = _make()
    _settle(m, "idle", 0.0)
    out = _feed(m, sc, HIGH, [-0.6, -0.52])                    # -0.52 발동 → 쿨다운 2.0 s(1.48 까지)
    assert [s for _, s in out] == ["idle", "postroll"]
    _feed(m, sc, 0.0, _times(-0.44, 4))                        # 말끝 보존 네 조각 → 확인 → 긴급
    assert len(events) == 1 and m.state == "idle"
    m.note_robot_speech(WAKE_GREETING, now=0.0)
    out = _feed(m, sc, HIGH, _times(0.0, 6))                   # 0.00~0.40 건너뜀
    assert all(s == "idle" for _, s in out)
    m.set_dialog_state("navigating", now=0.45)
    times = _times(0.48, 14)                                   # 0.48~1.52
    out = _feed(m, sc, HIGH, times)
    assert _postroll_at(out, times) == 1.52                    # 쉬는 시간(1.48)이 지나서야 — 그 전 2연속은 쿨다운에 막힌다


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
    _settle(m, "idle", 0.0)
    m.note_robot_speech(WAKE_GREETING, now=0.0)
    out = _feed(m, sc, HIGH, _times(0.0, 13))                  # 0.00~0.96
    m.note_robot_speech("화장실로 안내해드릴까요?", now=1.0)
    out += _feed(m, sc, HIGH, _times(1.04, 18))                # 1.04~2.40
    assert all(s == "idle" for _, s in out)
    out = _feed(m, sc, HIGH, _times(2.56, 2))
    assert [s for _, s in out] == ["idle", "postroll"]


def test_p13_second_greeting_restarts_window():
    m, sc, _ = _make()
    _settle(m, "idle", 0.0)
    m.note_robot_speech(WAKE_GREETING, now=0.0)
    _feed(m, sc, 0.0, _times(0.0, 25))                         # 0.00~1.92 조용
    m.set_dialog_state("idle", now=2.0)
    m.note_robot_speech(WAKE_GREETING, now=2.0)                # 4.5 까지 다시
    out = _feed(m, sc, HIGH, _times(2.56, 24))                 # 2.56~4.40 — 첫 "네?" 만이면 2.64 에 발동
    assert all(s == "idle" for _, s in out)
    out = _feed(m, sc, HIGH, _times(4.56, 2))
    assert [s for _, s in out] == ["idle", "postroll"]


# --- 검토 I1 — "네?"가 시작될 때 이미 충분히(3 s) 서 있었을 때만 ---------------------------------------------------------

def test_p13_i1_a_returning_then_greeting_then_idle_fires_as_before():
    """복귀 중 "비카야" → 브레이크(CancelNav) → "네?" → idle 소식. "네?" 때 서 있지 않았다 — 건너뛰지 않는다."""
    def run(greet):
        m, sc, _ = _make()
        for t in (-3.0, -2.0, -1.0, -0.05):
            m.set_dialog_state("returning", now=t)
        if greet:
            m.note_robot_speech(WAKE_GREETING, now=0.0)
        m.set_dialog_state("idle", now=0.05)                   # 브레이크로 idle — "네?" 뒤에 왔다
        return _feed(m, sc, HIGH, _times(0.08, 30))
    got = run(True)
    assert got == run(False)
    assert any(s == "postroll" for _, s in got)


def test_p13_i1_b_idle_for_1s_fires_as_before():
    _fires_as_before([("idle", -1.0), ("idle", 0.0)], 0.0)    # 안내 취소 직후 idle 에서 바로 "비카야" 같은 경우


def test_p13_i1_c_idle_over_3s_skips():
    _skipped([("idle", t) for t in (-3.2, -2.2, -1.2, -0.2)], 0.0)


def test_p13_i1_d_idle_navigating_idle_fires_as_before():
    news = ([("idle", t) for t in (-6.0, -5.0, -4.0, -3.0)] + [("navigating", -2.0)]
            + [("idle", -1.0), ("idle", 0.0)])                 # idle 이 다시 이어진 지 1 s
    _fires_as_before(news, 0.0)


def test_p13_i1_e_change_within_list_is_not_a_break():
    news = [("asking_next", t) for t in (-4.0, -3.0, -2.0)] + [("waiting", -1.0), ("waiting", 0.0)]
    _skipped(news, 0.0)


def test_p13_i1_f_news_gap_over_3s_then_greeting_fires_as_before():
    news = [("idle", t) for t in (-9.0, -8.0, -7.0, -6.0, -5.0)] + [("idle", -1.0)]   # 4 s 끊겼다 다시 왔다
    _fires_as_before(news, 0.0)


def test_p13_i1_g_unqualified_greeting_ends_previous_skip():
    m, sc, _ = _make()
    _settle(m, "idle", 0.0)
    m.note_robot_speech(WAKE_GREETING, now=0.0)                # 자격 있는 "네?" — 2.5 까지
    assert all(s == "idle" for _, s in _feed(m, sc, HIGH, _times(0.0, 6)))   # 0.00~0.40 건너뜀
    m.set_dialog_state("navigating", now=0.45)
    _feed(m, sc, 0.0, _times(0.48, 2))                         # 0.48·0.56 조용
    m.set_dialog_state("idle", now=0.6)
    assert _feed(m, sc, HIGH, [0.64]) == [(None, "idle")]      # 조각 조건으로는 다시 건너뛴다
    m.note_robot_speech(WAKE_GREETING, now=0.7)                # idle 이 이어진 지 0.1 s — 자격 없음
    times = _times(0.72, 2)
    out = _feed(m, sc, HIGH, times)
    assert _postroll_at(out, times) == 0.8                     # 앞 "네?"의 건너뛰기도 끝났다(0.72 부터 셈)


# --- 검토 Minor 1 — 시각이 뒤로 가면(음수) 신선하지 않다 ----------------------------------------------------------------

def test_p13_clock_back_state_news_is_not_fresh():
    """자격 있는 "네?" 뒤에 받은 소식의 시각이 조각 시각보다 뒤다(벽시계가 뒤로 감) — 그 조각들은 건너뛰지 않는다."""
    def run(greet):
        m, sc, _ = _make()
        _settle(m, "idle", 0.0)
        if greet:
            m.note_robot_speech(WAKE_GREETING, now=0.0)        # 자격 있음 — 2.5 까지
        m.set_dialog_state("idle", now=1.0)
        return _feed(m, sc, HIGH, _times(0.08, 11))            # 0.08~0.88 — 지금 − 받은 시각 < 0
    got = run(True)
    assert got == run(False)
    assert any(s == "postroll" for _, s in got)


def test_p13_clock_back_greeting_is_not_settled():
    _fires_as_before([("idle", t) for t in (10.0, 11.0, 12.0, 13.0)], 0.0)


# --- 기록 ------------------------------------------------------------------------------------------------------

def test_p13_one_log_line_per_greeting():
    lines, near = [], []
    m, sc, _ = _make(lines=lines, near=near)
    _settle(m, "idle", 0.0)
    m.note_robot_speech(WAKE_GREETING, now=0.0)
    _feed(m, sc, 0.6, _times(0.0, 5))
    _feed(m, sc, HIGH, _times(0.4, 5))
    _feed(m, sc, 0.2, _times(0.8, 21))                         # ~2.40
    assert lines == []                                         # 아직 건너뛰는 중 — 끝나야 한 줄
    _feed(m, sc, 0.0, _times(2.56, 3))
    assert lines == ["긴급 관문 건너뜀('네?' 직후 2.5 s · 상태 idle): 최고 점수 0.95"]
    assert near == []                                          # 건너뛴 점수로 '놓침' 기록을 따로 남기지 않는다
    # 문턱을 넘은 점수가 없던 "네?"는 줄을 남기지 않는다(목록 안에서 바뀐 waiting 은 끊김이 아니다)
    m.set_dialog_state("waiting", now=2.8)
    m.note_robot_speech(WAKE_GREETING, now=3.0)
    _feed(m, sc, 0.3, _times(3.0, 31))
    _feed(m, sc, 0.0, _times(5.6, 2))
    assert len(lines) == 1


def test_p13_log_line_each_greeting_when_greeting_repeats():
    lines = []
    m, sc, _ = _make(lines=lines)
    _settle(m, "asking_next", 0.0)
    m.note_robot_speech(WAKE_GREETING, now=0.0)
    _feed(m, sc, 0.7, _times(0.0, 10))                         # 0.00~0.72
    m.note_robot_speech(WAKE_GREETING, now=0.8)                # 두 번째 "네?" — 앞 "네?"의 줄이 먼저
    _feed(m, sc, 0.9, _times(0.8, 2))
    assert lines == ["긴급 관문 건너뜀('네?' 직후 2.5 s · 상태 asking_next): 최고 점수 0.70"]
    _feed(m, sc, 0.9, _times(0.96, 25))                        # ~2.88
    _feed(m, sc, 0.0, _times(3.36, 2))                         # 3.3 이 지났다
    assert lines[1:] == ["긴급 관문 건너뜀('네?' 직후 2.5 s · 상태 asking_next): 최고 점수 0.90"]

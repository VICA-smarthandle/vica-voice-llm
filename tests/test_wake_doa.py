"""호출 방향 통보 — "비카야"가 온 방향을 밖(ROS)으로 내보낸다."""
from __future__ import annotations

import numpy as np

from src.wakeword_monitor import WakewordMonitor


class Fake:
    def predict(self, _frame):
        return {"a": 0.0, "b": 0.0}

    def transcribe(self, _audio):
        return ""


def make(**kwargs):
    fake = Fake()
    return WakewordMonitor(
        on_emergency=lambda e: None,
        on_user_text=lambda t: None,
        predict=fake.predict,
        transcribe=fake.transcribe,
        **kwargs,
    )


def test_wake_direction_is_announced():
    seen: list = []
    m = make(on_wake_doa=seen.append)
    m.note_wake_direction(123.0, now=0.0)
    assert seen == [123.0]


def test_wake_direction_is_float():
    """칩은 정수(0~359)를 준다. 밖으로는 float 로 낸다 — Float32 토픽이다."""
    seen: list = []
    m = make(on_wake_doa=seen.append)
    m.note_wake_direction(200, now=0.0)
    assert seen == [200.0]
    assert isinstance(seen[0], float)


def test_no_direction_announces_nothing():
    """DOA 를 못 읽으면 조용히 아무 일도 없다 — 무음 실패."""
    seen: list = []
    m = make(on_wake_doa=seen.append)
    m.note_wake_direction(None, now=0.0)
    assert seen == []


def test_direction_is_still_locked_for_barge_in():
    """통보를 붙였다고 기존 barge-in 잠금이 사라지면 안 된다."""
    m = make()
    m.note_wake_direction(236.0, now=0.0)
    assert m._locked_doa == 236.0
    assert m._locked_doa_at == 0.0


def test_callback_is_optional():
    """콜백을 안 주면 통보 없이 잠금만 한다 (기존 도구·시험 호환)."""
    m = make()
    m.note_wake_direction(90.0, now=0.0)
    assert m._locked_doa == 90.0


def test_consecutive_calls_each_announce_their_own_direction():
    """지키는 불변식: 발행되는 방향은 항상 "이번" 호출의 것이다 — 깨지면"""
    seen: list = []
    m = make(on_wake_doa=seen.append)
    m.note_wake_direction(10, now=0.0)
    m.note_wake_direction(200, now=1.0)
    assert seen == [10.0, 200.0]

"""reSpeaker 소리 끊김 기록(11-11 사용자: "usb 가 끊기면 끊겼다고 음성 노드 터미널에 남겨야") — 시각 규칙만 시험한다."""
from __future__ import annotations

from src.frame_watch import FrameWatch


def test_gap_reported_once_then_every_10s():
    w = FrameWatch()
    assert w.on_frame(0.0) is None
    assert w.on_idle(0.5) is None
    assert w.on_idle(1.2) == "[USB] reSpeaker 소리가 1 s 째 안 들어옴 — 끊김 의심"
    assert w.on_idle(5.0) is None
    assert w.on_idle(11.3).startswith("[USB] reSpeaker 소리가 11 s 째")


def test_recovery_reported_with_gap_length():
    w = FrameWatch()
    w.on_frame(0.0)
    w.on_idle(3.0)
    assert w.on_frame(4.5) == "[USB] 소리 다시 들어옴(끊김 4.5 s)"
    assert w.on_frame(4.6) is None


def test_no_recovery_line_without_reported_gap():
    w = FrameWatch()
    w.on_frame(0.0)
    assert w.on_frame(0.08) is None


def test_no_gap_before_first_frame():
    assert FrameWatch().on_idle(5.0) is None


def test_status_same_text_rate_limited():
    w = FrameWatch()
    assert w.on_status("input overflow", 0.0) == "[USB] 입력 상태: input overflow"
    assert w.on_status("input overflow", 5.0) is None
    assert w.on_status("input overflow", 10.1) is not None


def test_status_different_text_not_limited():
    w = FrameWatch()
    assert w.on_status("input overflow", 0.0) is not None
    assert w.on_status("input underflow", 1.0) is not None

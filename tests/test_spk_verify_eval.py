"""화자 검증 신뢰도 평가기의 순수한 부분 — 말소리 자르기·평균 벡터·시도 묶기·통과율·EER."""
from __future__ import annotations

import numpy as np
import pytest

from tools.spk_verify_eval import (attempts_for, duration_bin, eer, latest_by_file, mean_embedding, summarize,
                                   trim_speech)

SR = 16000


def _tone(sec: float, amp: float) -> np.ndarray:
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * 300 * t)).astype(np.float32)


def test_trim_keeps_speech_plus_padding_only():
    x = np.concatenate([np.zeros(SR), _tone(0.5, 0.3), np.zeros(SR)]).astype(np.float32)
    y = trim_speech(x, SR)
    assert abs(len(y) / SR - 0.7) <= 0.02


def test_trim_ignores_quiet_hiss_relative_to_speech():
    rng = np.random.default_rng(0)
    hiss = lambda n: (0.005 * rng.standard_normal(n)).astype(np.float32)  # noqa: E731
    x = np.concatenate([hiss(SR), _tone(0.5, 0.3) + hiss(SR // 2), hiss(SR)])
    y = trim_speech(x, SR)
    assert abs(len(y) / SR - 0.7) <= 0.02


def test_trim_drops_small_click_far_after_speech():
    x = np.concatenate([np.zeros(SR), _tone(0.5, 0.3), np.zeros(2 * SR), _tone(0.04, 0.02), np.zeros(SR)])
    y = trim_speech(x.astype(np.float32), SR)
    assert abs(len(y) / SR - 0.7) <= 0.02


def test_trim_keeps_two_loud_words_with_a_pause_between():
    x = np.concatenate([np.zeros(SR), _tone(0.4, 0.3), np.zeros(int(0.6 * SR)), _tone(0.4, 0.2), np.zeros(SR)])
    y = trim_speech(x.astype(np.float32), SR)
    assert abs(len(y) / SR - 1.6) <= 0.02


def test_trim_returns_empty_for_silence():
    assert len(trim_speech(np.zeros(2 * SR, dtype=np.float32), SR)) == 0


def test_mean_embedding_normalizes_each_before_averaging():
    m = mean_embedding([np.array([2.0, 0.0]), np.array([0.0, 1.0])])
    assert m == pytest.approx([0.70711, 0.70711], abs=1e-4)


def test_attempts_pair_each_command_with_next_as_retry_and_wake_in_front():
    got = attempts_for(["a", "b", "c"], "w")
    assert got == [
        {"first": ("a",), "second": ("a", "b"), "wake": ("w", "a")},
        {"first": ("b",), "second": ("b", "c"), "wake": ("w", "b")},
        {"first": ("c",), "second": ("c", "a"), "wake": ("w", "c")},
    ]


def test_attempts_without_wake_leave_wake_empty():
    got = attempts_for(["a", "b"], None)
    assert [g["wake"] for g in got] == [None, None]


def test_attempts_with_single_command_have_no_retry():
    assert attempts_for(["a"], None) == [{"first": ("a",), "second": None, "wake": None}]


@pytest.mark.parametrize("sec,label", [
    (0.59, "<0.6"), (0.6, "0.6~1.0"), (0.99, "0.6~1.0"), (1.0, "1.0~1.5"),
    (1.5, "1.5~2.5"), (2.49, "1.5~2.5"), (2.5, "≥2.5"),
])
def test_duration_bin_edges(sec, label):
    assert duration_bin(sec) == label


def test_eer_perfect_separation_is_zero():
    rate, _ = eer([0.9, 0.8], [0.2, 0.1])
    assert rate == 0.0


def test_eer_crossing_point():
    rate, th = eer([0.9, 0.8, 0.4], [0.5, 0.3, 0.1])
    assert rate == pytest.approx(1 / 3)
    assert th == pytest.approx(0.5)


def test_latest_line_per_file_wins():
    lines = [
        {"file": "p1/cmd_01.wav", "text": "취소해", "time": "1"},
        {"file": "p1/cmd_02.wav", "text": "안내 끝내줘", "time": "2"},
        {"file": "p1/cmd_01.wav", "text": "취소해", "time": "3"},
    ]
    got = latest_by_file(lines)
    assert [(g["file"], g["time"]) for g in got] == [("p1/cmd_01.wav", "3"), ("p1/cmd_02.wav", "2")]


def _rec(genuine, s1, s12=None, sw=None):
    return {"genuine": genuine, "s1": s1, "s12": s12, "sw": sw, "dur1": 1.2}


def test_summarize_counts_single_and_retry_pass_rates():
    recs = [
        _rec(True, 0.7, 0.8), _rec(True, 0.3, 0.6), _rec(True, 0.2, 0.3), _rec(True, 0.9, 0.9),
        _rec(False, 0.1, 0.2), _rec(False, 0.55, 0.4), _rec(False, 0.3, 0.6), _rec(False, 0.0, 0.1),
    ]
    row = {r["threshold"]: r for r in summarize(recs, thresholds=[0.5])["by_threshold"]}[0.5]
    assert row["genuine_single"] == pytest.approx(2 / 4)
    assert row["genuine_retry"] == pytest.approx(3 / 4)
    assert row["impostor_single"] == pytest.approx(1 / 4)
    assert row["impostor_retry"] == pytest.approx(2 / 4)


def test_summarize_retry_ignores_missing_second_try():
    recs = [_rec(True, 0.3, None), _rec(False, 0.3, None)]
    row = summarize(recs, thresholds=[0.5])["by_threshold"][0]
    assert row["genuine_retry"] == 0.0 and row["impostor_retry"] == 0.0


def test_summarize_operating_point_is_highest_threshold_meeting_target():
    recs = [_rec(True, s, s) for s in (0.9, 0.7, 0.5, 0.2)] + [_rec(False, s, s) for s in (0.6, 0.1)]
    op = summarize(recs, thresholds=[0.3, 0.5, 0.6, 0.8], target=0.75)["op_retry"]
    assert op == {"threshold": 0.5, "genuine": 0.75, "impostor": 0.5}


def test_summarize_operating_point_none_when_target_unreachable():
    recs = [_rec(True, 0.1, 0.1), _rec(False, 0.0, 0.0)]
    assert summarize(recs, thresholds=[0.5], target=0.9)["op_retry"] is None

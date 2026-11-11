"""화자 검증 신뢰도 평가기의 순수한 부분 — 말소리 자르기·평균 벡터·시도 묶기·통과율·EER.

모델(sherpa-onnx)·wav 읽기는 실기로 본다. 기대값은 전부 손으로 계산한 숫자다.
"""
from __future__ import annotations

import numpy as np
import pytest

from tools.spk_verify_eval import (attempts_for, duration_bin, eer, latest_by_file, mean_embedding, summarize,
                                   trim_speech)

SR = 16000


def _tone(sec: float, amp: float) -> np.ndarray:
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * 300 * t)).astype(np.float32)


# --- 말소리 자르기 -----------------------------------------------------------------------------------------

def test_trim_keeps_speech_plus_padding_only():
    # 1 s 조용 + 0.5 s 말 + 1 s 조용 → 말 0.5 s 앞뒤로 0.1 s 씩 = 0.7 s
    x = np.concatenate([np.zeros(SR), _tone(0.5, 0.3), np.zeros(SR)]).astype(np.float32)
    y = trim_speech(x, SR)
    assert abs(len(y) / SR - 0.7) <= 0.02


def test_trim_ignores_quiet_hiss_relative_to_speech():
    # 바닥 잡음 0.005(말 0.3 보다 −35 dB) 는 말로 치지 않는다 → 말 0.5 s + 덧댐 0.2 s
    rng = np.random.default_rng(0)
    hiss = lambda n: (0.005 * rng.standard_normal(n)).astype(np.float32)  # noqa: E731
    x = np.concatenate([hiss(SR), _tone(0.5, 0.3) + hiss(SR // 2), hiss(SR)])
    y = trim_speech(x, SR)
    assert abs(len(y) / SR - 0.7) <= 0.02


def test_trim_drops_small_click_far_after_speech():
    # 말 0.5 s(0.3) 뒤 2 s 떨어진 0.04 s 짧은 잡음(0.02, 말보다 −23 dB) — 녹음 64개 중 30여 개에 있던 모양.
    # 잡음까지 이으면 2.24 s 가 되어 무음이 판정에 섞인다 → 말 0.5 s + 덧댐 0.2 s 만
    x = np.concatenate([np.zeros(SR), _tone(0.5, 0.3), np.zeros(2 * SR), _tone(0.04, 0.02), np.zeros(SR)])
    y = trim_speech(x.astype(np.float32), SR)
    assert abs(len(y) / SR - 0.7) <= 0.02


def test_trim_keeps_two_loud_words_with_a_pause_between():
    # 말 0.4 s · 쉼 0.6 s · 말 0.4 s — 둘 다 크면 둘 다 말이다 → 1.4 s + 덧댐 0.2 s
    x = np.concatenate([np.zeros(SR), _tone(0.4, 0.3), np.zeros(int(0.6 * SR)), _tone(0.4, 0.2), np.zeros(SR)])
    y = trim_speech(x.astype(np.float32), SR)
    assert abs(len(y) / SR - 1.6) <= 0.02


def test_trim_returns_empty_for_silence():
    assert len(trim_speech(np.zeros(2 * SR, dtype=np.float32), SR)) == 0


# --- 평균 벡터 -------------------------------------------------------------------------------------------

def test_mean_embedding_normalizes_each_before_averaging():
    # 길이가 다른 두 직교 벡터 — 각자 단위 길이로 맞춘 뒤 평균해야 45° 방향이 나온다
    m = mean_embedding([np.array([2.0, 0.0]), np.array([0.0, 1.0])])
    assert m == pytest.approx([0.70711, 0.70711], abs=1e-4)


# --- 시도 묶기(되묻기·호출어 붙이기) ------------------------------------------------------------------------

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


# --- 길이 구간 ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sec,label", [
    (0.59, "<0.6"), (0.6, "0.6~1.0"), (0.99, "0.6~1.0"), (1.0, "1.0~1.5"),
    (1.5, "1.5~2.5"), (2.49, "1.5~2.5"), (2.5, "≥2.5"),
])
def test_duration_bin_edges(sec, label):
    assert duration_bin(sec) == label


# --- EER ---------------------------------------------------------------------------------------------

def test_eer_perfect_separation_is_zero():
    rate, _ = eer([0.9, 0.8], [0.2, 0.1])
    assert rate == 0.0


def test_eer_crossing_point():
    # 문턱 0.5: 본인 0.4 거절(1/3) · 남 0.5 통과(1/3) → 같아지는 점
    rate, th = eer([0.9, 0.8, 0.4], [0.5, 0.3, 0.1])
    assert rate == pytest.approx(1 / 3)
    assert th == pytest.approx(0.5)


# --- 재녹음 -------------------------------------------------------------------------------------------

def test_latest_line_per_file_wins():
    lines = [
        {"file": "p1/cmd_01.wav", "text": "취소해", "time": "1"},
        {"file": "p1/cmd_02.wav", "text": "안내 끝내줘", "time": "2"},
        {"file": "p1/cmd_01.wav", "text": "취소해", "time": "3"},   # 다시 녹음
    ]
    got = latest_by_file(lines)
    assert [(g["file"], g["time"]) for g in got] == [("p1/cmd_01.wav", "3"), ("p1/cmd_02.wav", "2")]


# --- 통과율 요약 ----------------------------------------------------------------------------------------

def _rec(genuine, s1, s12=None, sw=None):
    return {"genuine": genuine, "s1": s1, "s12": s12, "sw": sw, "dur1": 1.2}


def test_summarize_counts_single_and_retry_pass_rates():
    recs = [
        _rec(True, 0.7, 0.8), _rec(True, 0.3, 0.6), _rec(True, 0.2, 0.3), _rec(True, 0.9, 0.9),
        _rec(False, 0.1, 0.2), _rec(False, 0.55, 0.4), _rec(False, 0.3, 0.6), _rec(False, 0.0, 0.1),
    ]
    row = {r["threshold"]: r for r in summarize(recs, thresholds=[0.5])["by_threshold"]}[0.5]
    assert row["genuine_single"] == pytest.approx(2 / 4)    # 0.7·0.9
    assert row["genuine_retry"] == pytest.approx(3 / 4)     # + 0.3→0.6
    assert row["impostor_single"] == pytest.approx(1 / 4)   # 0.55
    assert row["impostor_retry"] == pytest.approx(2 / 4)    # + 0.3→0.6 (남도 두 번 기회를 얻는다)


def test_summarize_retry_ignores_missing_second_try():
    recs = [_rec(True, 0.3, None), _rec(False, 0.3, None)]
    row = summarize(recs, thresholds=[0.5])["by_threshold"][0]
    assert row["genuine_retry"] == 0.0 and row["impostor_retry"] == 0.0


def test_summarize_operating_point_is_highest_threshold_meeting_target():
    # 본인 되묻기 포함 통과율 ≥ 0.75 를 지키는 가장 높은 문턱
    recs = [_rec(True, s, s) for s in (0.9, 0.7, 0.5, 0.2)] + [_rec(False, s, s) for s in (0.6, 0.1)]
    op = summarize(recs, thresholds=[0.3, 0.5, 0.6, 0.8], target=0.75)["op_retry"]
    assert op == {"threshold": 0.5, "genuine": 0.75, "impostor": 0.5}


def test_summarize_operating_point_none_when_target_unreachable():
    recs = [_rec(True, 0.1, 0.1), _rec(False, 0.0, 0.0)]
    assert summarize(recs, thresholds=[0.5], target=0.9)["op_retry"] is None

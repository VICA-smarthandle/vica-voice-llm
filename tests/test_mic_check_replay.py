"""녹음 재채점 — 같은 녹음·같은 값이면 같은 결과, 문턱을 바꾸면 결과가 따라 바뀐다(설계 5절)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.mic_check_replay import Models, Recording, replay_answer, replay_call, replay_near, replay_stop
from src.mic_check_rules import MicValues

WAKE = np.full(1280, 1111, dtype=np.int16)
STOP = np.full(1280, 2222, dtype=np.int16)
VOICE = np.full(1280, 3000, dtype=np.int16)
QUIET = np.zeros(1280, dtype=np.int16)


def predict(frame):
    v = int(frame[0])
    return {"a": 0.9 if v == 1111 else 0.0, "b": 0.9 if v == 2222 else 0.0}


def models(verify="멈춰", listen="화장실로 가자", confirm="네"):
    return Models(predict=predict, transcribe=lambda a: verify,
                  transcribe_listen=lambda a: listen, transcribe_confirm=lambda a: confirm)


V = MicValues(0.008, 0.02, 0.70, 0.50, 0.007, 1.5, 1)


def rec(*parts):
    frames, vads = [], []
    for frame, n, vad in parts:
        frames += [frame] * n
        vads += [vad] * n
    return Recording(tuple(frames), tuple(vads))


QUIET_REC = rec((QUIET, 10, False))
CALL = rec((QUIET, 3, False), (WAKE, 2, False), (VOICE, 10, True), (QUIET, 5, False))
ANSWER = rec((QUIET, 3, False), (VOICE, 10, True), (QUIET, 5, False))
STOP_REC = rec((QUIET, 3, False), (STOP, 1, False), (QUIET, 8, False))


def test_call_woke_and_captured():
    r = replay_call(models(), CALL, QUIET_REC, V)
    assert r.woke and r.captured and r.text == "화장실로 가자"
    assert r.peak == pytest.approx(0.9) and not r.emergency


def test_call_missed_when_wake_threshold_above_score():
    r = replay_call(models(), CALL, QUIET_REC, replace(V, wake_threshold=0.95))
    assert not r.woke and not r.captured


def test_answer_yes_no():
    yes = replay_answer(models(confirm="네."), ANSWER, QUIET_REC, V, "yes")
    assert yes.accepted and yes.correct and yes.clip_rms is not None
    no = replay_answer(models(confirm="네."), ANSWER, QUIET_REC, V, "no")
    assert no.accepted and not no.correct


def test_answer_rejected_when_listen_min_above_clip():
    r = replay_answer(models(), ANSWER, QUIET_REC, replace(V, listen_min_rms=0.2), "yes")
    assert not r.accepted and r.clip_rms is not None and r.clip_rms < 0.2


def test_stop_confirmed():
    r = replay_stop(models(), STOP_REC, QUIET_REC, V)
    assert r.confirmed and r.gate and r.text == "멈춰" and r.peak == pytest.approx(0.9) and not r.woke


def test_stop_rejected_by_verify():
    r = replay_stop(models(verify="멈추라"), STOP_REC, QUIET_REC, V)
    assert not r.confirmed and r.gate and r.text == "멈추라"


def test_stop_missed_below_single_threshold():
    r = replay_stop(models(), STOP_REC, QUIET_REC, replace(V, emergency_single=0.95))
    assert not r.confirmed and not r.gate and r.peak == pytest.approx(0.9)


def test_same_recording_same_result():
    assert replay_stop(models(), STOP_REC, QUIET_REC, V) == replay_stop(models(), STOP_REC, QUIET_REC, V)
    assert replay_call(models(), CALL, QUIET_REC, V) == replay_call(models(), CALL, QUIET_REC, V)


def test_cross_trigger_seen_in_answer():
    mixed = rec((QUIET, 3, False), (STOP, 1, False), (QUIET, 8, False))
    r = replay_answer(models(verify="멈춰"), mixed, QUIET_REC, V, "yes")
    assert r.emergency


def test_without_quiet_recording():
    assert replay_stop(models(), STOP_REC, None, V).confirmed


def tone(hz, amp=3000):
    t = np.arange(1280) / 16000.0
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.int16)


HUM = tone(150)
SPEECH = tone(1000)


def test_band_gate_follows_values():
    """⑫ 처럼 대역 최소를 귀에 넣는다 — 칩이 말이라 해도 대역 밖 소리는 청취 창을 열지 않는다."""
    hum = rec((QUIET, 3, False), (HUM, 10, True), (QUIET, 5, False))
    assert replay_answer(models(), hum, QUIET_REC, V, "yes").accepted
    assert not replay_answer(models(), hum, QUIET_REC, replace(V, band_min=0.004), "yes").accepted
    speech = rec((QUIET, 3, False), (SPEECH, 10, True), (QUIET, 5, False))
    assert replay_answer(models(), speech, QUIET_REC, replace(V, band_min=0.004), "yes").accepted


FAR = rec((QUIET, 3, False), (np.full(1280, 600, dtype=np.int16), 10, True), (QUIET, 5, False))


def test_approach_answer_uses_approach_min():
    """접근 질문 대답 창(⑫: 미션 awaiting_user → 호출 끔)은 접근 대답 최소로 받는다. 질문 창은 수음 최소 그대로."""
    size = replay_answer(models(), FAR, QUIET_REC, V, "yes").clip_rms
    v = replace(V, listen_min_rms=size * 1.2, approach_min_rms=size * 0.8)
    assert not replay_answer(models(), FAR, QUIET_REC, v, "yes").accepted
    far = replay_answer(models(), FAR, QUIET_REC, v, "yes", approach=True)
    assert far.accepted and far.correct
    assert not replay_answer(models(), FAR, QUIET_REC, replace(v, approach_min_rms=None), "yes", approach=True).accepted


def test_near_checks_question_and_approach_windows():
    size = replay_answer(models(), FAR, QUIET_REC, V, "yes").clip_rms
    v = replace(V, listen_min_rms=size * 1.2, approach_min_rms=size * 0.8)
    r = replay_near(models(confirm="어제 그거"), FAR, QUIET_REC, v)
    assert not r.leaked and r.leaked_approach and r.text == "어제 그거"
    assert r.clip_rms == pytest.approx(size) and not r.emergency
    quiet = replay_near(models(), FAR, QUIET_REC, replace(v, approach_min_rms=size * 1.2))
    assert not quiet.leaked and not quiet.leaked_approach and quiet.text == ""


def test_approach_none_ignores_env_value(monkeypatch):
    """점검기는 .env 를 환경에 올린다 — 값이 None(줄 없음)이면 귀가 환경의 옛 값을 쓰면 안 된다(검토 10-21)."""
    size = replay_answer(models(), FAR, QUIET_REC, V, "yes").clip_rms
    monkeypatch.setenv("VICA_APPROACH_MIN_RMS", f"{size * 0.8:.6f}")
    v = replace(V, listen_min_rms=size * 1.2, approach_min_rms=None)
    assert not replay_answer(models(), FAR, QUIET_REC, v, "yes", approach=True).accepted
    assert not replay_near(models(), FAR, QUIET_REC, v).leaked_approach

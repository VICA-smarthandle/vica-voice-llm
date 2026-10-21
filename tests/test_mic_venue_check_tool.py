"""점검기 진입점의 순수한 부분 — ⑫·⑬ 찾기, 멘트 읽기(빠진 파일 건너뜀). 마이크·칩·모델은 실기로 본다."""
from __future__ import annotations

import json
import queue
import wave
from datetime import datetime

import numpy as np

from src.mic_check_session import previous_robot_level
from src.mic_check_rules import MicValues
from src.wakeword_monitor import LISTEN_BAND_MIN_DEFAULT, speech_band_rms
from tools.mic_venue_check import (DATA, ROBOT_CLIPS, _Demo, _drain, _may_drain, lan_ip, load_clips, record_dir_for,
                                   voice_stack_running, write_chip)


def test_voice_stack_running():
    lines = [
        "/usr/bin/python3 /opt/ros/humble/bin/ros2 launch launch/vica_voice.launch.py map_id:=x",
        "/home/ji_w/VICA-smarthandle/vica-voice-llm/.venv/bin/python -m src.ros_wakeword_node",
        "/home/ji_w/VICA-smarthandle/vica-voice-llm/.venv/bin/python -m tools.mic_venue_check --name 연구실",
        "bash",
    ]
    assert voice_stack_running(lines) == lines[:2]


def test_robot_clips_are_the_eight_in_spec():
    assert len(ROBOT_CLIPS) == 8 and ROBOT_CLIPS[0] == "mission_msg_approach_onboarding.wav"


def test_load_clips_skips_missing(tmp_path):
    with wave.open(str(tmp_path / "a.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(1600, dtype="<i2").tobytes())
    (tmp_path / "manifest.json").write_text(json.dumps({"a.wav": "안녕하세요"}), encoding="utf-8")
    clips, missing = load_clips(tmp_path, ("a.wav", "b.wav"))
    assert [c.name for c in clips] == ["a.wav"] and clips[0].text == "안녕하세요" and clips[0].rate == 16000
    assert clips[0].pcm.dtype == np.float32 and missing == ["b.wav"]


def test_lan_ip_is_text():
    assert isinstance(lan_ip(), str) and lan_ip().count(".") == 3


def test_record_dir_for_real_and_demo():
    now = datetime(2026, 10, 27, 14, 32)
    assert record_dir_for("시연장", False, now) == DATA / "20261027-1432_시연장"
    assert record_dir_for("시연장", True, now) == DATA / "demo" / "20261027-1432_시연장"
    assert record_dir_for("a/b c", False, now).name == "20261027-1432_a_b_c"
    assert record_dir_for("///", False, now).name.endswith("_점검")


def test_previous_level_skips_demo_folder(tmp_path):
    real = tmp_path / "20261026-0900_연구실"
    real.mkdir()
    (real / "result.json").write_text(json.dumps({"measurements": {"robot": {"level": 0.029}}}), encoding="utf-8")
    demo = tmp_path / "demo" / "20261027-1000_데모"
    demo.mkdir(parents=True)
    (demo / "result.json").write_text(json.dumps({"measurements": {"robot": {"level": 0.5}}}), encoding="utf-8")
    assert previous_robot_level(tmp_path) == 0.029


def test_drain_empties_queue_and_counts():
    q = queue.Queue()
    for i in range(5):
        q.put(i)
    assert _drain(q) == 5 and q.empty()
    assert _drain(q) == 0


class _Busy:
    def __init__(self, kind=None, robot=False):
        self.recording_kind = kind
        self.robot_running = robot


def test_may_drain_only_when_not_recording():
    """녹음·6단계 도중에 비우면 녹음에 빈틈이 생긴다 — 그때는 밀린 조각을 차례로 처리한다."""
    assert _may_drain(_Busy()) is True
    assert _may_drain(_Busy(kind="stop")) is False
    assert _may_drain(_Busy(robot=True)) is False


def test_write_chip_takes_max_gain_from_values(monkeypatch):
    """최대 증폭은 화면 값(10-21)을 쓴다. 줄 없음이면 공장값 ×31.6 을 써서, 저장 안 한 ×2 시험이 칩에 남지 않게."""
    import src.audio_out as audio_out
    import src.dsp_state as dsp

    gains = []
    monkeypatch.setattr(audio_out, "reset_device_cache", lambda: None)
    monkeypatch.setattr(dsp, "apply_agc_desired_level", lambda v: True)
    monkeypatch.setattr(dsp, "apply_agc_max_gain", lambda v: gains.append(v) or True)
    monkeypatch.setattr(dsp, "apply_echo_tuning", lambda e, n: {"GAMMA_ETAIL": e, "NLATTENONOFF": n})
    monkeypatch.setenv("VICA_MIC_AGC_MAX_GAIN", "8")          # 환경 변수가 아니라 값을 따른다
    v = MicValues(0.008, 0.02, 0.70, 0.50, 0.007, 1.5, 1, agc_max_gain=2.0)
    assert write_chip(v) == [] and gains == [2.0]
    assert write_chip(MicValues(0.008, 0.02, 0.70, 0.50, 0.007, 1.5, 1)) == [] and gains == [2.0, 31.6]
    monkeypatch.setattr(dsp, "apply_agc_max_gain", lambda v: False)
    assert write_chip(v) == ["AGC 최대 증폭 ×2 쓰기 실패"]


class _Rec:
    def __init__(self, kind, pos, expected=None):
        self.recording_kind, self.recording_pos, self.recording_expected, self.robot_running = kind, pos, expected, False


def test_demo_voice_frames_pass_band_gate():
    """--demo 의 가짜 말소리도 말소리 대역에 힘이 있어야 한다 — 재채점이 대역 관문을 켠 뒤(10-21)."""
    demo = _Demo()
    for kind, pos, expected in (("call", 6, None), ("answer", 5, "yes"), ("answer", 5, "no")):
        f = demo.frame(_Rec(kind, pos, expected))
        assert demo.vad and speech_band_rms(f) >= LISTEN_BAND_MIN_DEFAULT
    no = demo.frame(_Rec("answer", 5, "no"))
    assert demo.models().transcribe_confirm(no) == "아니"
    assert demo.predict(demo.frame(_Rec("call", 3)))["a"] > 0.9

"""긴급 관문 기록(2026-10-12 사용자 결정) — 동작은 그대로, 남기는 것만 늘린다."""
from __future__ import annotations

from src.wakeword_monitor import POST_ROLL_FRAMES, WakewordMonitor, describe_emergency_check
from tests.test_wakeword_monitor import LOUD, QUIET, Fake, run_frames


def make(fake: Fake, events: list, texts: list, rejects: list) -> WakewordMonitor:
    return WakewordMonitor(
        on_emergency=events.append,
        on_user_text=texts.append,
        on_wake=lambda: None,
        on_reject=rejects.append,
        predict=fake.predict,
        transcribe=fake.transcribe,
    )


def test_confirmed_records_gate_score_and_times():
    fake = Fake(scores=[(0, 0.9), (0, 0.95)] + [(0, 0)] * 10, text="멈춰!")
    events, texts, rejects = [], [], []
    m = make(fake, events, texts, rejects)
    results = run_frames(m, 1 + POST_ROLL_FRAMES, LOUD, t0=10.0)
    assert results[-1] == "emergency" and len(events) == 1
    c = m.last_emergency_check
    assert c["result"] == "emergency" and c["where"] == "idle" and c["text"] == "멈춰!"
    assert c["score"] == 0.9
    assert abs(c["postroll"] - POST_ROLL_FRAMES * 0.08) < 1e-6
    assert c["stt"] >= 0.0


def test_idle_reject_records_too():
    fake = Fake(scores=[(0, 0.9), (0, 0.9)] + [(0, 0)] * 10, text="감사합니다.")
    events, texts, rejects = [], [], []
    m = make(fake, events, texts, rejects)
    results = run_frames(m, 1 + POST_ROLL_FRAMES, LOUD)
    assert results[-1] == "reject" and rejects == ["감사합니다."] and events == []
    assert m.last_emergency_check["result"] == "reject"
    assert m.last_emergency_check["where"] == "idle"


def test_reject_inside_listen_window_is_now_reported():
    fake = Fake(
        scores=[(0.9, 0), (0.9, 0)] + [(0, 0)] * 5 + [(0, 0.9), (0, 0.9)] + [(0, 0)] * 10,
        text="화장실로 가줘",
    )
    events, texts, rejects = [], [], []
    m = make(fake, events, texts, rejects)
    run_frames(m, 2, LOUD)
    run_frames(m, 5, LOUD, t0=1.0, vad=True)
    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD, t0=1.5, vad=True)
    assert "listen_resumed" in results
    assert rejects == ["화장실로 가줘"]
    c = m.last_emergency_check
    assert c["where"] == "listen" and c["result"] == "listen_resumed"
    results = run_frames(m, 14, QUIET, t0=3.0)
    assert "user_text" in results and texts == ["화장실로 가줘"] and events == []


def test_followup_answer_path_is_reported():
    fake = Fake(scores=[(0, 0.9), (0, 0.9)] + [(0, 0)] * 10, text="필요없다구")
    events, texts, rejects = [], [], []
    m = make(fake, events, texts, rejects)
    m.arm_followup(now=0.0)
    results = run_frames(m, 1 + POST_ROLL_FRAMES, LOUD, t0=1.0)
    assert results[-1] == "user_text" and texts == ["필요없다구"] and events == []
    assert rejects == ["필요없다구"]
    assert m.last_emergency_check["result"] == "followup_answer"


def test_describe_emergency_check():
    base = {"score": 0.874, "postroll": 0.32, "stt": 1.612, "text": "멈춰!"}
    s = describe_emergency_check({**base, "where": "idle", "result": "emergency"})
    assert "관문 0.87" in s and "관문→판정 1.93 s" in s and "STT 1.61 s" in s
    assert "청취 창" not in s
    s = describe_emergency_check({**base, "where": "listen", "result": "listen_resumed"})
    assert "청취 창 안" in s and "듣기 이어감" in s
    s = describe_emergency_check({**base, "where": "idle", "result": "followup_answer"})
    assert "질문 답으로 넘김" in s
    assert describe_emergency_check(None) == ""


import json
import wave

import numpy as np

from src.wakeword_gate import FrameGate
from src.wakeword_monitor import (
    EMERGENCY_NEAR_MISS_FLOOR, describe_emergency_near, save_emergency_clip)


def make_near(fake: Fake, nears: list, rejects=None) -> WakewordMonitor:
    return WakewordMonitor(
        on_emergency=lambda e: None,
        on_user_text=lambda t: None,
        on_reject=(rejects.append if rejects is not None else None),
        on_emergency_score=nears.append,
        predict=fake.predict,
        transcribe=fake.transcribe,
    )


def test_gate_reports_cooldown_block():
    g = FrameGate(0.5, persist=2, cooldown_sec=2.0)
    assert not g.feed(0.9, 0.0) and g.feed(0.9, 0.08) and not g.cooldown_blocked
    g.feed(0.9, 0.5)
    assert not g.feed(0.9, 0.58) and g.cooldown_blocked
    g.feed(0.1, 0.66)
    assert not g.cooldown_blocked


def test_below_threshold_episode_is_reported_with_peak():
    fake = Fake(scores=[(0, 0.35), (0, 0.45), (0, 0.4), (0, 0.1)])
    nears = []
    m = make_near(fake, nears)
    run_frames(m, 4, LOUD)
    assert len(nears) == 1
    n = nears[0]
    assert n["peak"] == 0.45 and n["frames_over"] == 0 and not n["cooldown"]
    assert n["audio"].dtype == np.int16 and len(n["audio"]) > 0


def test_single_frame_over_speaking_threshold_is_reported():
    fake = Fake(scores=[(0, 0.45), (0, 0.1)])
    nears = []
    m = make_near(fake, nears)
    m.set_speaking(True, now=0.0)
    run_frames(m, 2, LOUD)
    assert [(n["peak"], n["frames_over"]) for n in nears] == [(0.45, 1)]


def test_below_floor_is_not_reported():
    fake = Fake(scores=[(0, EMERGENCY_NEAR_MISS_FLOOR - 0.05), (0, 0.0)])
    nears = []
    m = make_near(fake, nears)
    run_frames(m, 2, LOUD)
    assert nears == []


def test_fired_episode_is_not_a_near_miss():
    fake = Fake(scores=[(0, 0.9), (0, 0.9)] + [(0, 0.6)] * 4 + [(0, 0)] * 3, text="멈춰")
    nears = []
    m = make_near(fake, nears)
    run_frames(m, 9, LOUD)
    assert nears == []


def test_second_shout_inside_cooldown_is_reported():
    fake = Fake(scores=[(0, 0.9), (0, 0.9)] + [(0, 0)] * 4 + [(0, 0)] * 3
                + [(0, 0.9), (0, 0.9), (0, 0)], text="멈춰가")
    nears, rejects = [], []
    m = make_near(fake, nears, rejects)
    run_frames(m, 12, LOUD)
    assert rejects == ["멈춰가"]
    assert len(nears) == 1 and nears[0]["cooldown"] and nears[0]["frames_over"] == 2
    assert abs(nears[0]["after_fire"] - 0.72) < 1e-6


def test_describe_emergency_near():
    s = describe_emergency_near({"peak": 0.434, "frames_over": 0, "cooldown": False}, 0.5)
    assert s == "🎯 긴급 점수 놓침: 0.43 (문턱 0.50)"
    s = describe_emergency_near({"peak": 0.71, "frames_over": 1, "cooldown": False}, 0.5)
    assert "문턱 넘은 조각 1개(2개 연속 필요)" in s
    s = describe_emergency_near({"peak": 0.9, "frames_over": 2, "cooldown": True, "after_fire": 0.64}, 0.5)
    assert "쉬는 시간 중 무시(앞 반응 0.64 s 뒤)" in s


def test_save_emergency_clip_writes_wav_and_index(tmp_path):
    audio = (np.arange(16000 * 2) % 100).astype(np.int16)
    info = {"result": "reject", "where": "idle", "score": 0.94, "postroll": 0.32, "stt": 1.56,
            "text": "멈춰가"}
    path = save_emergency_clip(tmp_path, audio, info, wall_time=1791771218.085)
    assert path.exists() and path.suffix == ".wav" and "reject" in path.name
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 16000, 32000)
    row = json.loads((tmp_path / "clips.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert row["file"] == path.name and row["text"] == "멈춰가" and row["score"] == 0.94
    assert "audio" not in row


import sys
import types

import pytest

from src.wakeword_monitor import ear_stt_model_names, emergency_single_frame_from_env


def test_gate_single_frame_rule():
    g = FrameGate(0.5, persist=2, cooldown_sec=2.0, single=0.5)
    assert g.feed(0.6, 0.0)
    assert not g.feed(0.9, 0.08) and g.cooldown_blocked
    talk = FrameGate(0.35, persist=2, cooldown_sec=2.0, single=0.5)
    assert not talk.feed(0.4, 0.0) and talk.feed(0.4, 0.08)
    assert FrameGate(0.35, persist=2, single=0.5).feed(0.55, 0.0)
    assert not FrameGate(0.5, persist=2).feed(0.9, 0.0)


def test_monitor_uses_single_rule_only_for_emergency():
    fake = Fake(scores=[(0.9, 0), (0, 0)] + [(0, 0)] * 6)
    wakes = []
    m = WakewordMonitor(on_emergency=lambda e: None, on_user_text=lambda t: None,
                        on_wake=lambda: wakes.append(1), predict=fake.predict, transcribe=fake.transcribe)
    run_frames(m, 8, LOUD)
    assert wakes == [] and m.gate_b.single == 0.5 and m.gate_a.single is None


def test_single_frame_env(monkeypatch):
    monkeypatch.delenv("VICA_EMERGENCY_SINGLE_FRAME", raising=False)
    assert emergency_single_frame_from_env() == 0.5
    monkeypatch.setenv("VICA_EMERGENCY_SINGLE_FRAME", "off")
    assert emergency_single_frame_from_env() is None
    monkeypatch.setenv("VICA_EMERGENCY_SINGLE_FRAME", "0.6")
    assert emergency_single_frame_from_env() == 0.6
    monkeypatch.setenv("VICA_EMERGENCY_SINGLE_FRAME", "abc")
    assert emergency_single_frame_from_env() == 0.5
    monkeypatch.setenv("VICA_EMERGENCY_SINGLE_FRAME", "off")
    m = WakewordMonitor(on_emergency=lambda e: None, on_user_text=lambda t: None,
                        predict=lambda f: {"a": 0, "b": 0}, transcribe=lambda a: "")
    assert m.gate_b.single is None


def test_ear_stt_model_names(monkeypatch):
    monkeypatch.setenv("VICA_STT_MODEL", "turbo-x")
    monkeypatch.setenv("VICA_VERIFY_STT_MODEL", "small")
    assert ear_stt_model_names() == ("turbo-x", "small")
    monkeypatch.delenv("VICA_STT_MODEL")
    monkeypatch.delenv("VICA_VERIFY_STT_MODEL")
    assert ear_stt_model_names() == ("medium", "medium")


def _load_with_stub(monkeypatch, listen, verify, engines=("whisper", "whisper")):
    """faster_whisper·stt 를 가짜로 바꿔 _load_real 이 어떤 모델을 몇 개 띄우는지 본다."""
    loaded = []

    class Stub:
        def __init__(self, name, device=None, compute_type=None):
            self.name = name
            loaded.append(name)

        def transcribe(self, audio, **kw):
            return [types.SimpleNamespace(text=self.name)], None

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=Stub))
    import src
    stub_stt = types.ModuleType("src.stt")
    monkeypatch.setitem(sys.modules, "src.stt", stub_stt)
    monkeypatch.setattr(src, "stt", stub_stt, raising=False)
    monkeypatch.setenv("VICA_STT_MODEL", listen)
    monkeypatch.setenv("VICA_VERIFY_STT_MODEL", verify)
    monkeypatch.setenv("VICA_STT_ENGINE", engines[0])
    monkeypatch.setenv("VICA_VERIFY_STT_ENGINE", engines[1])
    m = WakewordMonitor(on_emergency=lambda e: None, on_user_text=lambda t: None,
                        predict=lambda f: {"a": 0, "b": 0})
    m._load_real()
    return m, loaded


def test_verify_uses_its_own_model(monkeypatch):
    m, loaded = _load_with_stub(monkeypatch, "turbo-x", "small")
    audio = np.zeros(1600, dtype=np.int16)
    assert sorted(loaded) == ["small", "turbo-x"]
    assert m._transcribe(audio) == "small"
    assert m._transcribe_listen(audio) == "turbo-x"
    assert m._transcribe_confirm(audio) == "turbo-x"


def test_same_name_loads_one_model(monkeypatch):
    m, loaded = _load_with_stub(monkeypatch, "turbo-x", "turbo-x")
    assert loaded == ["turbo-x"]


class _FakeQwen:
    name = "Qwen3-ASR-1.7B-Q8_0"

    def listen(self, audio):
        return "qwen-listen"

    def verify(self, audio, is_emergency):
        return "멈춰" if is_emergency("멈춰") else "?"


def _with_qwen(monkeypatch, engines, fail=False):
    import src.wakeword_monitor as wm_mod
    from src.stt_qwen import QwenUnavailable

    def fake_from_env(log=print):
        if fail:
            raise QwenUnavailable("모델 없음")
        return _FakeQwen()
    monkeypatch.setattr(wm_mod, "qwen_from_env", fake_from_env)
    return _load_with_stub(monkeypatch, "turbo-x", "small", engines)


def test_both_roles_qwen_load_no_whisper(monkeypatch):
    m, loaded = _with_qwen(monkeypatch, ("qwen", "qwen"))
    audio = np.zeros(1600, dtype=np.int16)
    assert loaded == []
    assert m._transcribe_listen(audio) == "qwen-listen"
    assert m._transcribe_confirm(audio) == "qwen-listen"
    assert m._transcribe(audio) == "멈춰"
    assert m.qwen is not None


def test_only_conversation_on_qwen_keeps_whisper_small_for_stop(monkeypatch):
    m, loaded = _with_qwen(monkeypatch, ("qwen", "whisper"))
    audio = np.zeros(1600, dtype=np.int16)
    assert loaded == ["small"]
    assert m._transcribe(audio) == "small"
    assert m._transcribe_listen(audio) == "qwen-listen"


def test_only_stop_check_on_qwen_keeps_whisper_turbo_for_conversation(monkeypatch):
    m, loaded = _with_qwen(monkeypatch, ("whisper", "qwen"))
    audio = np.zeros(1600, dtype=np.int16)
    assert loaded == ["turbo-x"]
    assert m._transcribe_listen(audio) == "turbo-x"
    assert m._transcribe(audio) == "멈춰"


def test_qwen_failure_stops_the_ear_without_whisper_fallback(monkeypatch):
    from src.stt_qwen import QwenUnavailable
    with pytest.raises(QwenUnavailable):
        _with_qwen(monkeypatch, ("qwen", "whisper"), fail=True)

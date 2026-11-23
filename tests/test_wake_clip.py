"""호출 소리 저장(2026-11-17 사용자) — 호출 점수가 문턱을 넘을 때마다 직전 ~2.5 s 를 남긴다. 판정에는 쓰지 않는다."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.wakeword_monitor import WakewordMonitor

FRAME = np.full(1280, 1000, dtype=np.int16)
NODE = (Path(__file__).resolve().parents[1] / "src" / "ros_wakeword_node.py").read_text(encoding="utf-8")


def monitor(clips: list, **kw) -> WakewordMonitor:
    m = WakewordMonitor(on_emergency=lambda e: None, on_user_text=lambda t: None, on_wake=lambda: None,
                        predict=lambda f: {"a": 0.0, "b": 0.0}, transcribe=lambda a: "",
                        on_wake_clip=(lambda audio, info: clips.append((audio, info))), **kw)
    for _ in range(5):
        m._ring.append(FRAME)
    return m


def test_confirmed_call_saves_the_last_audio():
    clips: list = []
    m = monitor(clips)
    m._report_wake_score("fire", 0.92)
    [(audio, info)] = clips
    assert len(audio) == 5 * 1280 and audio.dtype == np.int16
    assert info["result"] == "wake" and info["peak"] == 0.92 and info["threshold"] == m.gate_a.threshold
    assert info["listening"] is False and info["robot_speaking"] is False


def test_over_threshold_but_not_confirmed_is_saved_as_near():
    clips: list = []
    m = monitor(clips)
    m._report_wake_score("near", m.gate_a.threshold + 0.07)
    assert [info["result"] for _a, info in clips] == ["near"]


def test_below_threshold_near_miss_is_not_saved():
    clips: list = []
    m = monitor(clips)
    m._report_wake_score("near", m.gate_a.threshold - 0.05)
    assert clips == []


def test_own_voice_over_threshold_is_saved_as_self():
    clips: list = []
    m = monitor(clips)
    m._report_wake_score("self", 0.95)
    assert [info["result"] for _a, info in clips] == ["self"]


def test_score_callback_still_runs_and_clip_failure_is_swallowed():
    scores: list = []

    def boom(audio, info):
        raise OSError("디스크 가득")

    m = WakewordMonitor(on_emergency=lambda e: None, on_user_text=lambda t: None, on_wake=lambda: None,
                        predict=lambda f: {"a": 0.0, "b": 0.0}, transcribe=lambda a: "",
                        on_wake_score=lambda kind, peak: scores.append((kind, peak)), on_wake_clip=boom)
    m._ring.append(FRAME)
    m._report_wake_score("fire", 0.9)
    assert scores == [("fire", 0.9)]


def test_no_clip_callback_or_empty_ring_is_fine():
    m = WakewordMonitor(on_emergency=lambda e: None, on_user_text=lambda t: None, on_wake=lambda: None,
                        predict=lambda f: {"a": 0.0, "b": 0.0}, transcribe=lambda a: "")
    m._report_wake_score("fire", 0.9)
    clips: list = []
    m2 = monitor(clips)
    m2._ring.clear()
    m2._report_wake_score("fire", 0.9)
    assert clips == []


def test_node_wires_the_wake_clip_folder():
    assert 'os.environ.get("VICA_WAKE_CLIP_DIR", "~/vica_data/wake_clips")' in NODE
    assert "on_wake_clip=self._save_wake_clip if self._wake_clip_dir else None" in NODE
    at = NODE.index("def _save_wake_clip")
    assert "save_emergency_clip(self._wake_clip_dir" in NODE[at: at + 700]
    assert "호출 소리 저장" in NODE

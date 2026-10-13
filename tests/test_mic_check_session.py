"""점검 진행 — 가짜 마이크·칩·스피커로 단계 진행, 두 번 누르기, 값 바꾸기, 다시 채점, 저장·되돌리기(설계 3·5·7절)."""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime

import numpy as np
import pytest

from src.env_file import read_env, values_from
from src.mic_check_replay import Models
from src.mic_check_rules import PASS, REDO
from src.mic_check_session import STEPS, Clip, Hardware, Session, previous_robot_level

NOISE = np.full(1280, 100, dtype=np.int16)    # rms ≈ 0.003
WAKE = np.full(1280, 1111, dtype=np.int16)
STOP = np.full(1280, 2222, dtype=np.int16)
STOP44 = np.full(1280, 2244, dtype=np.int16)  # 멈춰 점수 0.44 — 0.5 아래
YES = np.full(1280, 3000, dtype=np.int16)
NO = np.full(1280, 3001, dtype=np.int16)

SCORES = {1111: (0.9, 0.0), 2222: (0.0, 0.9), 2244: (0.0, 0.44)}


def predict(frame):
    a, b = SCORES.get(int(frame[0]), (0.0, 0.0))
    return {"a": a, "b": b}


MODELS = Models(predict=predict, transcribe=lambda a: "멈춰", transcribe_listen=lambda a: "화장실로 가자",
                transcribe_confirm=lambda a: "아니" if np.any(a == 3001) else "그래")


class FakeHw:
    def __init__(self):
        self.applied, self.vad, self.session = [], False, None

    def chip_apply(self, v):
        self.applied.append(v)
        return []

    def read_vad(self):
        return self.vad

    def play(self, pcm, rate):
        for _ in range(5):                    # 재생하는 동안 마이크 조각이 들어온다
            self.session.on_frame(NOISE)


def make(tmp_path, prev=None):
    msgs = []
    hw = FakeHw()
    env = tmp_path / ".env"
    env.write_text("VICA_MIC_AGC_DESIRED=0.007\nVICA_MIC_GAMMA_ETAIL=1.5\nVICA_MIC_NLATTEN=1\n", encoding="utf-8")
    s = Session(models=MODELS,
                hw=Hardware(hw.chip_apply, hw.read_vad, hw.play, sleep=lambda sec: None, spawn=lambda fn: fn()),
                publish=lambda m, keep=None: msgs.append(m),
                env_values=values_from(read_env(env)), env_path=env, place="연구실",
                record_dir=tmp_path / "rec", backup_dir=tmp_path / "bk",
                clips=[Clip("a.wav", "안녕하세요", np.zeros(1600, np.float32), 16000)],
                previous_robot_level=prev, clock=lambda: datetime(2026, 10, 27, 14, 32, 5))
    hw.session = s
    return s, hw, msgs


def feed(s, hw, frame, n, vad=False):
    hw.vad = vad
    for _ in range(n):
        s.on_frame(frame)


def record(s, hw, parts, total=None):
    if total is None:
        total = int(round(s.step.seconds / 0.08))   # 단계 녹음 길이만큼(호출 6 s, 나머지 4 s)
    s.handle({"cmd": "record"})
    used = 0
    for frame, n, vad in parts:
        feed(s, hw, frame, n, vad)
        used += n
    feed(s, hw, NOISE, total - used)


def last(msgs, kind):
    return [m for m in msgs if m["type"] == kind][-1]


def run_all(s, hw, stop_frames=(STOP,) * 5):
    record(s, hw, [], total=188)                                              # 2단계 조용히 15 s
    for _ in range(5):                                                        # 쉬고 3 + 이어서 2
        record(s, hw, [(NOISE, 3, False), (WAKE, 2, False), (YES, 10, True)])
    for frame in (YES, YES, NO, NO):
        record(s, hw, [(NOISE, 3, False), (frame, 10, True)])
    for frame in stop_frames:
        record(s, hw, [(NOISE, 3, False), (frame, 1, False)])
    s.handle({"cmd": "robot"})
    feed(s, hw, NOISE, 1)                                                     # 재생 끝을 받아 마무리


def test_full_run_passes_and_records(tmp_path):
    s, hw, msgs = make(tmp_path)
    run_all(s, hw)
    assert s.step.key == "result"
    res = last(msgs, "results")
    assert res["overall"] == PASS, res["rows"]
    assert len(s.meas.noise) == 3 and s.meas.robot.self_reactions == 0
    data = json.loads((tmp_path / "rec" / "result.json").read_text(encoding="utf-8"))
    assert data["verdict"]["overall"] == PASS and data["place"] == "연구실"
    assert (tmp_path / "rec" / "quiet.wav").exists() and (tmp_path / "rec" / "stop_5.wav").exists()


def test_call_step_is_three_paused_then_two_run_on(tmp_path):
    """3단계 = 쉬고 부르기 3 + 이어서 부르기 2(2026-10-13). 이어서 부르기는 참고 줄로 따로 판정한다."""
    s, hw, msgs = make(tmp_path)
    record(s, hw, [], total=188)
    says = []
    for _ in range(5):
        says.append(last(msgs, "status")["step"]["say"])
        record(s, hw, [(NOISE, 3, False), (WAKE, 2, False), (YES, 10, True)])
    assert len(s.meas.calls) == 3 and len(s.meas.runs) == 2 and s.step.key == "answer"
    assert "쉬고" in says[0] and "이어서" in says[3]
    assert last(msgs, "status")["progress"]["call"] == ["pass"] * 5
    rows = {r["key"]: r["grade"] for r in last(msgs, "results")["rows"]}
    assert rows["call"] == PASS and rows["run"] == PASS
    assert (tmp_path / "rec" / "call_1_pause.wav").exists() and (tmp_path / "rec" / "call_4_run.wav").exists()
    s.handle({"cmd": "set", "key": "wake_threshold", "value": 0.75})
    s.handle({"cmd": "rejudge"})
    assert len(s.meas.calls) == 3 and len(s.meas.runs) == 2


def test_answer_step_asks_the_words_people_use(tmp_path):
    s, hw, msgs = make(tmp_path)
    assert [p for st in STEPS if st.key == "answer" for p in st.prompts] == ["“그래”", "“그래”", "“아니”", "“아니”"]


def test_double_record_is_ignored(tmp_path):
    s, hw, msgs = make(tmp_path)
    s.handle({"cmd": "record"})
    s.handle({"cmd": "record"})
    assert last(msgs, "toast")["text"].startswith("지금은")
    feed(s, hw, NOISE, 188)
    assert len(s.meas.noise) == 3 and s.step.key == "call"


def test_set_value_out_of_range_ignored(tmp_path):
    s, hw, msgs = make(tmp_path)
    before = s.values
    for cmd in ({"cmd": "set", "key": "wake_threshold", "value": 0.95},
                {"cmd": "set", "key": "listen_min_rms", "value": "abc"},
                {"cmd": "set", "key": "모르는값", "value": 1},
                {"cmd": "set", "key": "agc", "value": 0.02},
                {"cmd": "set", "key": "echo", "value": "엄청 셈"}):
        s.handle(cmd)
        assert s.values == before
    assert len([m for m in msgs if m["type"] == "toast"]) == 5
    assert hw.applied == []


def test_set_value_updates_live_and_chip(tmp_path):
    s, hw, msgs = make(tmp_path)
    s.handle({"cmd": "set", "key": "wake_threshold", "value": 0.653})
    assert s.values.wake_threshold == 0.65 and s.live.gate_a.threshold == 0.65    # 눈금(0.01)에 맞춤
    assert hw.applied == []
    s.handle({"cmd": "set", "key": "agc", "value": 0.010})
    assert hw.applied[-1].agc == 0.010
    s.handle({"cmd": "set", "key": "echo", "value": "mid"})
    assert (hw.applied[-1].echo_etail, hw.applied[-1].echo_nlatten) == (1.0, 1)
    assert last(msgs, "status")["values"]["echo"] == "mid"


def test_rejudge_with_lower_stop_threshold(tmp_path):
    s, hw, msgs = make(tmp_path)
    run_all(s, hw, stop_frames=(STOP,) * 4 + (STOP44,))
    stop_row = next(r for r in last(msgs, "results")["rows"] if r["key"] == "stop")
    assert stop_row["grade"] == "warn" and "점수 0.44" in stop_row["detail"]
    s.handle({"cmd": "set", "key": "emergency_single", "value": 0.43})
    assert last(msgs, "results")["stale"] is True
    s.handle({"cmd": "rejudge"})
    res = last(msgs, "results")
    rows = {r["key"]: r for r in res["rows"]}
    assert rows["stop"]["grade"] == PASS
    assert rows["robot"]["grade"] == REDO                     # 문턱이 바뀌어 6단계는 다시
    assert res["stale"] is False
    first = {r["key"]: r for r in res["first"]}
    assert first["stop"]["grade"] == "warn"


def test_agc_change_then_record_restarts_that_step(tmp_path):
    s, hw, msgs = make(tmp_path)
    record(s, hw, [], total=188)
    record(s, hw, [(NOISE, 3, False), (WAKE, 2, False), (YES, 10, True)])
    assert len(s.meas.calls) == 1
    s.handle({"cmd": "set", "key": "agc", "value": 0.010})
    rows = {r["key"]: r["grade"] for r in last(msgs, "results")["rows"]}
    assert rows["noise"] == REDO and rows["call"] == REDO
    record(s, hw, [(NOISE, 3, False), (WAKE, 2, False), (YES, 10, True)])
    assert len(s.meas.calls) == 1 and "처음부터" in [m for m in msgs if m["type"] == "toast"][-1]["text"]


def test_redo_clears_step(tmp_path):
    s, hw, msgs = make(tmp_path)
    record(s, hw, [], total=188)
    record(s, hw, [(NOISE, 3, False), (WAKE, 2, False), (YES, 10, True)])
    s.handle({"cmd": "redo", "step": "quiet"})
    assert s.meas.noise == [] and s.step.key == "quiet" and len(s.meas.calls) == 1


def test_rejudge_without_recordings(tmp_path):
    s, hw, msgs = make(tmp_path)
    s.handle({"cmd": "rejudge"})
    assert "녹음이 없습니다" in last(msgs, "toast")["text"]


def test_save_and_restore(tmp_path):
    s, hw, msgs = make(tmp_path)
    original_values = values_from(read_env(tmp_path / ".env"))
    s.handle({"cmd": "set", "key": "wake_threshold", "value": 0.65})
    s.handle({"cmd": "save_preview"})
    assert last(msgs, "save_preview")["lines"] == [
        "VICA_WAKE_THRESHOLD=0.65  # 10-27 연구실 점검에서 바꿈 (전 0.70)"]
    s.handle({"cmd": "save"})
    assert "VICA_WAKE_THRESHOLD=0.65" in (tmp_path / ".env").read_text(encoding="utf-8")
    assert s.env_values.wake_threshold == 0.65 and (tmp_path / "bk").is_dir()
    s.handle({"cmd": "restore"})
    assert "VICA_WAKE_THRESHOLD" not in read_env(tmp_path / ".env")      # 저장이 넣은 줄은 사라진다(기본값)
    assert values_from(read_env(tmp_path / ".env")) == original_values
    assert s.values.wake_threshold == 0.70 and s.env_values.wake_threshold == 0.70
    assert s.live.gate_a.threshold == 0.70


def test_save_blocked_by_rule(tmp_path):
    s, hw, msgs = make(tmp_path)
    s.handle({"cmd": "set", "key": "listen_min_rms", "value": 0.014})     # 0.02 < 0.014 × 1.5
    s.handle({"cmd": "save_preview"})
    assert "1.5배" in last(msgs, "toast")["text"]
    assert not [m for m in msgs if m["type"] == "save_preview"]


def go_to_robot(s):
    s.step_index = [x.key for x in STEPS].index("robot")


def test_play_failure_makes_robot_row_fail_and_is_not_a_baseline(tmp_path):
    s, hw, msgs = make(tmp_path)
    go_to_robot(s)

    def boom(pcm, rate):
        raise RuntimeError("스피커 없음")

    s.hw.play = boom
    s.handle({"cmd": "robot"})
    feed(s, hw, NOISE, 1)
    robot = next(r for r in last(msgs, "results")["rows"] if r["key"] == "robot")
    assert robot["grade"] == "fail" and "재생 실패" in robot["detail"]
    data = json.loads((tmp_path / "rec" / "result.json").read_text(encoding="utf-8"))
    assert data["measurements"]["robot"]["played_ok"] is False
    assert previous_robot_level(tmp_path) is None            # 조용한 level 이 다음 점검 기준이 되지 않는다


def test_previous_robot_level_skips_failed_playback(tmp_path):
    root = tmp_path / "mic_checks"
    for name, robot in (("20261020-1000_a", {"level": 0.020}),
                        ("20261026-0900_b", {"level": 0.004, "played_ok": False})):
        d = root / name
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({"measurements": {"robot": robot}}), encoding="utf-8")
    assert previous_robot_level(root) == 0.020


def test_finish_stops_running_robot_playback(tmp_path):
    s, hw, msgs = make(tmp_path)
    go_to_robot(s)
    started, stopped = threading.Event(), threading.Event()
    threads = []

    def play(pcm, rate):
        started.set()
        stopped.wait(10)

    def spawn(fn):
        t = threading.Thread(target=fn, daemon=True)
        threads.append(t)
        t.start()

    s.hw.play, s.hw.stop_play, s.hw.spawn = play, stopped.set, spawn
    s.handle({"cmd": "robot"})
    assert started.wait(3)
    t0 = time.monotonic()
    s.handle({"cmd": "finish"})
    assert time.monotonic() - t0 < 3.5 and s.done
    threads[0].join(1)
    assert not threads[0].is_alive() and s._robot_error == "중단됨"


def test_replay_failure_in_recording_toasts_and_drops_it(tmp_path):
    s, hw, msgs = make(tmp_path)
    record(s, hw, [], total=188)
    record(s, hw, [(NOISE, 3, False), (WAKE, 2, False), (YES, 10, True)])
    n = len(s.meas.calls)

    def boom(audio):
        raise RuntimeError("모델 죽음")

    s.models = s.models.__class__(predict, boom, boom, boom)
    record(s, hw, [(NOISE, 3, False), (WAKE, 2, False), (YES, 10, True)])   # on_frame 이 예외를 던지지 않는다
    toast = [m for m in msgs if m["type"] == "toast"][-1]["text"]
    assert "채점하지 못했습니다" in toast and "RuntimeError" in toast
    assert len(s.meas.calls) == n and len(s._calls) == n and s.recording_kind is None
    assert last(msgs, "busy")["text"] == ""


def test_rejudge_failure_keeps_previous_results(tmp_path):
    s, hw, msgs = make(tmp_path)
    run_all(s, hw)
    before_calls = list(s.meas.calls)
    before_judged = s.judged_values
    s.handle({"cmd": "set", "key": "emergency_single", "value": 0.45})

    def boom(audio):
        raise RuntimeError("모델 죽음")

    s.models = s.models.__class__(predict, boom, boom, boom)
    s.handle({"cmd": "rejudge"})
    assert "다시 채점하지 못했습니다" in [m for m in msgs if m["type"] == "toast"][-1]["text"]
    assert s.meas.calls == before_calls and s.judged_values == before_judged
    assert last(msgs, "busy")["text"] == ""


def test_save_unexpected_error_is_toasted(tmp_path, monkeypatch):
    import src.mic_check_session as mod

    s, hw, msgs = make(tmp_path)
    s.handle({"cmd": "set", "key": "wake_threshold", "value": 0.65})

    def boom(*a, **k):
        raise OSError("디스크 가득 참")

    monkeypatch.setattr(mod, "write_changes", boom)
    s.handle({"cmd": "save"})
    assert "디스크 가득 참" in last(msgs, "toast")["text"] and s.env_values.wake_threshold == 0.70


def test_chip_problems_block_recording_and_robot(tmp_path):
    s, hw, msgs = make(tmp_path)
    s.chip_problems = ["AGC 쓰기 실패"]
    s.handle({"cmd": "record"})
    assert "칩에 .env 값을 쓰지 못해" in last(msgs, "toast")["text"] and s.recording_kind is None
    go_to_robot(s)
    s.handle({"cmd": "robot"})
    assert "AGC 쓰기 실패" in last(msgs, "toast")["text"] and not s.robot_running


def test_robot_blocked_when_thresholds_not_rejudged(tmp_path):
    s, hw, msgs = make(tmp_path)
    go_to_robot(s)
    s.handle({"cmd": "set", "key": "wake_threshold", "value": 0.65})
    s.handle({"cmd": "robot"})
    assert "다시 채점을 누른 뒤" in last(msgs, "toast")["text"] and not s.robot_running


def test_previous_robot_level(tmp_path):
    root = tmp_path / "mic_checks"
    for name, level in (("20261020-1000_연구실", 0.020), ("20261026-0900_연구실", 0.029)):
        d = root / name
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({"measurements": {"robot": {"level": level}}}), encoding="utf-8")
    now_dir = root / "20261027-1430_시연장"
    now_dir.mkdir()
    (root / "env_backups").mkdir()
    assert previous_robot_level(root, exclude=now_dir) == 0.029
    assert previous_robot_level(tmp_path / "없음") is None


def test_finish(tmp_path):
    s, hw, msgs = make(tmp_path)
    s.handle({"cmd": "finish"})
    assert s.done

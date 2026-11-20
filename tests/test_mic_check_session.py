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
from src.mic_check_rules import FAIL, PASS, REDO
from src.mic_check_session import STEPS, Clip, Hardware, Session, previous_robot_level

NOISE = np.full(1280, 100, dtype=np.int16)    # rms ≈ 0.003
WAKE = np.full(1280, 1111, dtype=np.int16)
STOP = np.full(1280, 2222, dtype=np.int16)
STOP44 = np.full(1280, 2244, dtype=np.int16)  # 멈춰 점수 0.44 — 0.5 아래
YES = np.full(1280, 3000, dtype=np.int16)
NO = np.full(1280, 3001, dtype=np.int16)
CHAT = np.full(1280, 150, dtype=np.int16)     # 옆사람 잡담 rms ≈ 0.0046 — 수음 최소 0.008 아래

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
    # 가짜 소리는 직류(한 값)라 말소리 대역이 0 이다 — 대역 관문은 끄고 본다(관문 자체는 test_mic_check_replay)
    env.write_text("VICA_MIC_AGC_DESIRED=0.007\nVICA_MIC_GAMMA_ETAIL=1.5\nVICA_MIC_NLATTEN=1\nVICA_LISTEN_BAND_MIN=off\n",
                   encoding="utf-8")
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
    for frame in (YES, NO):                                                   # 7단계 먼 대답
        record(s, hw, [(NOISE, 3, False), (frame, 10, True)])
    for _ in range(3):                                                        # 8단계 옆사람 잡담
        record(s, hw, [(NOISE, 3, False), (CHAT, 40, True)])


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
    assert (tmp_path / "rec" / "far_2_no.wav").exists() and (tmp_path / "rec" / "near_3.wav").exists()
    status = last(msgs, "status")
    assert status["progress"]["far"] == ["pass"] * 2 and status["progress"]["near"] == ["pass"] * 3
    assert status["step"]["of"] == 9


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
    assert last(msgs, "toast")["text"] == "지금은 녹음 중입니다 · 끝난 뒤 다시 누르세요"
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
    assert last(msgs, "toast")["text"] == "저장했습니다 · 음성 노드를 다시 켜야 적용됩니다"
    s.handle({"cmd": "restore"})
    assert last(msgs, "toast")["text"].endswith(" 으로 되돌렸습니다 · 음성 노드를 다시 켜야 적용됩니다")
    assert "VICA_WAKE_THRESHOLD" not in read_env(tmp_path / ".env")      # 저장이 넣은 줄은 사라진다(기본값)
    assert values_from(read_env(tmp_path / ".env")) == original_values
    assert s.values.wake_threshold == 0.70 and s.env_values.wake_threshold == 0.70
    assert s.live.gate_a.threshold == 0.70


def test_save_blocked_by_rule(tmp_path):
    s, hw, msgs = make(tmp_path)
    s.handle({"cmd": "set", "key": "listen_min_rms", "value": 0.025})     # 짧은 답 최소 0.02 < 수음 최소 0.025
    s.handle({"cmd": "save_preview"})
    assert "짧은 답 최소는 수음 최소 이상" in last(msgs, "toast")["text"]
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
    assert last(msgs, "toast")["text"].endswith(" · 끝내기 뒤 다시 시작하세요")
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


# ---------------------------------------------------------------- 10-21: 최대 증폭·접근 대답 최소·대역 최소, 먼 대답·옆사람
def go_to(s, key):
    s.step_index = [x.key for x in STEPS].index(key)


def test_steps_order_keeps_old_numbers():
    """새 단계는 로봇 혼자 말하기(6단계) 뒤 — 결과표·안내문의 2~6단계 번호가 그대로다."""
    assert [x.key for x in STEPS] == ["quiet", "call", "answer", "stop", "robot", "far", "near", "result"]


def test_far_step_turns_wake_off_only_while_recording(tmp_path):
    """접근 질문 대답처럼 듣는다 — 녹음하는 동안만 호출을 끄고(⑫: awaiting_user), 끝나면 되돌린다."""
    s, hw, msgs = make(tmp_path)
    go_to(s, "far")
    s.handle({"cmd": "record"})
    assert s.live._wake_suppressed
    feed(s, hw, NOISE, 50)
    assert len(s.meas.far) == 1 and s.meas.far[0].expected == "yes"
    # 말을 기다리는 질문 창이 아직 열려 있다 — 질문 창은 말이 없으면 CONFIRM_WINDOW_SEC(30 s)에 닫힌다
    assert s.live.state == "listen" and s.live._wake_suppressed
    feed(s, hw, NOISE, 340)
    assert s.live.state != "listen" and not s.live._wake_suppressed


def test_far_keeps_wake_off_until_live_window_closes(tmp_path):
    """녹음이 끝날 때 실시간 청취 창이 아직 열려 있으면 그 창이 닫힐 때까지 접근 대답 창으로 듣는다(⑫ 와 같게)."""
    s, hw, msgs = make(tmp_path)
    go_to(s, "far")
    record(s, hw, [(NOISE, 3, False), (YES, 47, True)])                      # 녹음 끝까지 말이 이어진다
    assert len(s.meas.far) == 1 and s.live.state == "listen" and s.live._wake_suppressed
    feed(s, hw, NOISE, 30)
    assert s.live.state != "listen" and not s.live._wake_suppressed


def test_live_ignores_env_approach_when_values_say_none(tmp_path, monkeypatch):
    monkeypatch.setenv("VICA_APPROACH_MIN_RMS", "0.006")
    s, hw, msgs = make(tmp_path)
    assert s.values.approach_min_rms is None and s.live.approach_min_rms is None


def test_factory_pick_without_line_is_no_change(tmp_path):
    """줄이 없을 때 ×31.6 공장값을 누르면 바뀐 것이 없다 — 칩 쓰기·다시 녹음·저장 줄 없음(검토 10-21)."""
    s, hw, msgs = make(tmp_path)
    record(s, hw, [], total=188)
    s.handle({"cmd": "set", "key": "agc_max_gain", "value": 31.6})
    assert s.values.agc_max_gain is None and hw.applied == []
    assert next(r for r in last(msgs, "results")["rows"] if r["key"] == "noise")["grade"] == PASS
    s.handle({"cmd": "save_preview"})
    assert "다른 값이 없습니다" in last(msgs, "toast")["text"]
    s.handle({"cmd": "set", "key": "agc_max_gain", "value": 2})
    s.handle({"cmd": "set", "key": "agc_max_gain", "value": 31.6})
    assert s.values.agc_max_gain is None and hw.applied[-1].agc_max_gain is None
    assert last(msgs, "status")["gain_factory"] == 31.6


def test_near_leak_makes_row_fail(tmp_path):
    s, hw, msgs = make(tmp_path)
    go_to(s, "near")
    for _ in range(3):
        record(s, hw, [(NOISE, 3, False), (YES, 10, True)])                  # 옆사람이 손잡이 사용자만큼 크다
    near = next(r for r in last(msgs, "results")["rows"] if r["key"] == "near")
    assert near["grade"] == FAIL and "질문 창 통과 3/3" in near["detail"]
    assert last(msgs, "status")["progress"]["near"] == ["fail"] * 3


def test_set_max_gain_writes_chip_and_needs_rerecord(tmp_path):
    s, hw, msgs = make(tmp_path)
    record(s, hw, [], total=188)
    s.handle({"cmd": "set", "key": "agc_max_gain", "value": 2})
    assert hw.applied[-1].agc_max_gain == 2.0 and s.values.agc_max_gain == 2.0
    assert last(msgs, "status")["values"]["agc_max_gain"] == 2.0
    noise = next(r for r in last(msgs, "results")["rows"] if r["key"] == "noise")
    assert noise["grade"] == REDO
    n = len(hw.applied)
    s.handle({"cmd": "set", "key": "agc_max_gain", "value": 3})                # 잰 적 없는 값
    assert len(hw.applied) == n and s.values.agc_max_gain == 2.0
    assert "×2" in last(msgs, "toast")["text"]


def test_status_offers_gain_choices_and_new_names(tmp_path):
    s, hw, msgs = make(tmp_path)
    st = last(msgs, "status")
    assert st["gain_choices"] == [2.0, 4.0, 8.0, 31.6]
    assert st["values"]["agc_max_gain"] is None and st["values"]["band_min"] is None
    assert st["env_keys"]["agc_max_gain"] == "VICA_MIC_AGC_MAX_GAIN"
    assert st["env_keys"]["approach_min_rms"] == "VICA_APPROACH_MIN_RMS"
    assert st["env_keys"]["band_min"] == "VICA_LISTEN_BAND_MIN"
    assert st["ranges"]["band_min"] == [0.001, 0.008, 0.0005]
    assert st["ranges"]["listen_min_rms"] == [0.005, 0.04, 0.0005]
    assert st["short_over_listen"] == 1.0


def test_set_approach_and_band_update_live_without_chip(tmp_path):
    s, hw, msgs = make(tmp_path)
    record(s, hw, [], total=188)
    record(s, hw, [(NOISE, 3, False), (WAKE, 2, False), (YES, 10, True)])
    s.handle({"cmd": "set", "key": "approach_min_rms", "value": 0.0061})
    s.handle({"cmd": "set", "key": "band_min", "value": 0.003})
    assert s.values.approach_min_rms == 0.006 and s.values.band_min == 0.003   # 눈금(0.0005)에 맞춤
    assert s.live.approach_min_rms == 0.006 and s.live.listen_band_min == 0.003
    assert hw.applied == [] and last(msgs, "results")["stale"] is True
    s.handle({"cmd": "rejudge"})
    assert s.judged_values.band_min == 0.003 and last(msgs, "results")["stale"] is False


def test_save_and_restore_new_keys(tmp_path):
    s, hw, msgs = make(tmp_path)
    original = values_from(read_env(tmp_path / ".env"))
    s.handle({"cmd": "set", "key": "agc_max_gain", "value": 2})
    s.handle({"cmd": "set", "key": "band_min", "value": 0.003})
    s.handle({"cmd": "save_preview"})
    assert last(msgs, "save_preview")["lines"] == [
        "VICA_MIC_AGC_MAX_GAIN=2  # 10-27 연구실 점검에서 바꿈 (전 off)",
        "VICA_LISTEN_BAND_MIN=0.003  # 10-27 연구실 점검에서 바꿈 (전 off)"]
    s.handle({"cmd": "save"})
    saved = read_env(tmp_path / ".env")
    assert saved["VICA_MIC_AGC_MAX_GAIN"] == "2" and saved["VICA_LISTEN_BAND_MIN"] == "0.003"
    s.handle({"cmd": "restore"})
    assert values_from(read_env(tmp_path / ".env")) == original
    assert "VICA_MIC_AGC_MAX_GAIN" not in read_env(tmp_path / ".env")
    assert s.live.listen_band_min is None and hw.applied[-1].agc_max_gain is None

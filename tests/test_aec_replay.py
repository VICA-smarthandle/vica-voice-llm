import json
import numpy as np
import pytest
from src.aec_replay import (Item, ItemResult, Ment, Session, format_report, frames_for, judge, load_session,
                            ment_reductions, run_item, split_blocks)
from src.echo_cancel import BLOCK, EchoConfig
from src.env_file import values_from
from src.mic_check_replay import Models

DT = 0.08
def res(id, wakes=0, emer=0, fw=None, fe=None, texts=(), rms=0.1):
    return ItemResult(id, wakes, emer, fw, fe, 0.0, 0.0, tuple(texts), "", rms)

def session(items, ments=()):
    return Session(wav=None, ments=tuple(ments), items=tuple(items))

def test_load_session_and_blocks(tmp_path):
    d = {"version": 1, "rate": 16000, "channels": 6, "wav": "set.wav",
         "ments": [{"name": "a.wav", "text": "가", "start": 2.0, "end": 5.0}],
         "items": [{"id": "over_stop_1", "kind": "overlap", "say": "멈춰", "expect": "emergency", "start": 2.0, "end": 7.0}]}
    p = tmp_path / "set.json"; p.write_text(json.dumps(d, ensure_ascii=False))
    s = load_session(p)
    assert s.wav == tmp_path / "set.wav" and s.ments[0] == Ment("a.wav", "가", 2.0, 5.0) and s.items[0].expect == "emergency"
    blocks = split_blocks(np.zeros((BLOCK * 3 + 100, 6), np.int16))
    assert len(blocks) == 3 and blocks[0].shape == (BLOCK, 6)

def test_frames_for_keeps_one_learner_for_the_whole_file():
    made = []
    class Ec:
        def __init__(self, **_): made.append(1)
        def process(self, mic, ref): return (mic // 2).astype(np.int16)
        def close(self): pass
    blocks = [np.full((BLOCK, 6), 1000, np.int16) for _ in range(5)]
    assert all(np.array_equal(f, b[:, 0]) for f, b in zip(frames_for(blocks, "off", EchoConfig("on")), blocks))
    on = frames_for(blocks, "on", EchoConfig("on"), factory=Ec)
    assert len(made) == 1 and all(int(f[0]) == 500 for f in on)

def _models():
    def predict(frame):
        v = int(frame[0])
        return {"a": 0.95 if v == 5000 else 0.0, "b": 0.9 if v == 7000 else 0.0}
    return Models(predict=predict, transcribe=lambda audio: "멈춰")

def _frames(marks):                       # marks: {블록 번호: 값}
    return [np.full(BLOCK, marks.get(i, 0), np.int16) for i in range(200)]

def test_run_item_counts_emergency_inside_the_window():
    item = Item("over_stop_1", "overlap", "멈춰", "emergency", 4.0, 9.0)
    r = run_item(_models(), values_from({}), _frames({70: 7000}), item, [Ment("a.wav", "잠시 멈추겠습니다.", 4.0, 7.0)])
    assert r.emergencies == 1 and 70 * DT <= r.first_emergency <= 76 * DT and r.peak_b == 0.9

def test_run_item_counts_wake_and_ignores_marks_outside_the_window():
    item = Item("over_wake_1", "overlap", "비카야", "wake", 4.0, 9.0)
    r = run_item(_models(), values_from({}), _frames({70: 5000, 71: 5000, 150: 7000}), item, [])
    assert r.wakes == 1 and abs(r.first_wake - 71 * DT) < DT and r.emergencies == 0

ITEMS = [Item("over_stop_1", "overlap", "멈춰", "emergency", 0, 5), Item("over_wake_1", "overlap", "비카야", "wake", 5, 10),
         Item("user_stop_1", "user_only", "멈춰", "emergency", 10, 14), Item("robot_only", "robot_only", None, None, 14, 74)]
GOOD = {"over_stop_1": res("over_stop_1", emer=1, fe=3.0), "over_wake_1": res("over_wake_1", wakes=1, fw=7.0),
        "user_stop_1": res("user_stop_1", emer=1, fe=11.0), "robot_only": res("robot_only")}

def test_judge_passes_when_on_is_no_worse():
    rep = judge(session(ITEMS), GOOD, GOOD, [-8.0, -9.0, -7.0])
    assert rep["pass"] and rep["overlap_stop"] == {"n": 1, "off": 1, "on": 1, "lost": [], "late": [], "pass": True}
    assert rep["robot_only"]["reduce_db_median"] == -8.0

def test_judge_fails_on_a_lost_stop_a_late_stop_or_a_changed_user_item():
    lost = {**GOOD, "over_stop_1": res("over_stop_1")}
    assert not judge(session(ITEMS), GOOD, lost, [-8.0])["pass"]
    late = {**GOOD, "over_stop_1": res("over_stop_1", emer=1, fe=3.16)}
    assert judge(session(ITEMS), GOOD, late, [-8.0])["overlap_stop"]["late"] == ["over_stop_1"]
    quiet = {**GOOD, "user_stop_1": res("user_stop_1", emer=1, fe=11.0, rms=0.08)}     # −1.9 dB
    assert not judge(session(ITEMS), GOOD, quiet, [-8.0])["user_only"]["pass"]
    text = {**GOOD, "user_stop_1": res("user_stop_1", emer=1, fe=11.0, texts=("머",))}
    assert judge(session(ITEMS), GOOD, text, [-8.0])["user_only"]["differs"] == ["user_stop_1"]

def test_judge_fails_on_weak_reduction_or_new_robot_events():
    assert not judge(session(ITEMS), GOOD, GOOD, [-5.0, -5.5])["robot_only"]["pass"]
    noisy = {**GOOD, "robot_only": res("robot_only", wakes=1, fw=20.0)}
    assert judge(session(ITEMS), GOOD, noisy, [-8.0])["robot_only"]["new_events"] == 1

def test_ment_reductions_skip_the_first_4_seconds():
    ments = [Ment("a.wav", "가", 14.0, 16.0), Ment("b.wav", "나", 20.0, 22.0)]
    off = [np.full(BLOCK, 4000, np.int16)] * 1000; on = [np.full(BLOCK, 1000, np.int16)] * 1000
    out = ment_reductions(session(ITEMS, ments), off, on)
    assert len(out) == 1 and abs(out[0] - (-12.04)) < 0.05


ITEMS2 = ITEMS + [Item("over_stop_2", "overlap", "멈춰", "emergency", 20, 25)]
GOOD2 = {**GOOD, "over_stop_2": res("over_stop_2", emer=1, fe=21.0)}

def test_judge_lost_item_is_not_hidden_by_a_gain_elsewhere():
    off = {**GOOD2, "over_stop_2": res("over_stop_2")}
    on = {**GOOD2, "over_stop_1": res("over_stop_1")}
    rep = judge(session(ITEMS2), off, on, [-8.0])
    assert rep["overlap_stop"]["lost"] == ["over_stop_1"] and not rep["overlap_stop"]["pass"]
    assert rep["overlap_stop"]["off"] == rep["overlap_stop"]["on"] == 1

def test_judge_late_is_counted_in_blocks_and_none_time_is_late():
    def late(dt_blocks):
        on = {**GOOD, "over_stop_1": res("over_stop_1", emer=1, fe=3.0 + 40 * DT + dt_blocks * DT)}
        off = {**GOOD, "over_stop_1": res("over_stop_1", emer=1, fe=3.0 + 40 * DT)}
        return judge(session(ITEMS), off, on, [-8.0])["overlap_stop"]["late"]
    assert late(1) == [] and late(2) == ["over_stop_1"]
    none = {**GOOD, "over_stop_1": res("over_stop_1", emer=1, fe=None)}
    assert judge(session(ITEMS), GOOD, none, [-8.0])["overlap_stop"]["late"] == ["over_stop_1"]

def test_run_item_counts_events_confirmed_in_the_trailing_quiet_frames():
    item = Item("over_stop_1", "overlap", "멈춰", "emergency", 4.0, 9.0)       # 창 끝 = 블록 113
    r = run_item(_models(), values_from({}), _frames({110: 7000, 111: 7000, 112: 7000}), item, [])
    assert r.emergencies == 1 and r.first_emergency is not None
    w = Item("over_wake_1", "overlap", "비카야", "wake", 4.0, 9.0)
    r = run_item(_models(), values_from({}), _frames({111: 5000, 112: 5000}), w, [])
    assert r.wakes == 1 and r.first_wake is not None

def test_run_item_speaking_is_a_union_of_ments(monkeypatch):
    from src.wakeword_monitor import WakewordMonitor
    calls = []
    orig_s, orig_n = WakewordMonitor.set_speaking, WakewordMonitor.note_robot_speech
    def spy_s(self, value, now=None, **kw):
        calls.append(("speak", bool(value), now)); return orig_s(self, value, now=now, **kw)
    def spy_n(self, text, now=None, **kw):
        calls.append(("note", text)); return orig_n(self, text, now=now, **kw)
    monkeypatch.setattr(WakewordMonitor, "set_speaking", spy_s)
    monkeypatch.setattr(WakewordMonitor, "note_robot_speech", spy_n)
    item = Item("robot_only", "robot_only", None, None, 4.0, 9.0)
    run_item(_models(), values_from({}), _frames({}), item,
             [Ment("a.wav", "가", 4.0, 4.5), Ment("b.wav", "나", 4.8, 5.3)])
    assert [c[:2] for c in calls if c[0] == "note"] == [("note", "가"), ("note", "나")]
    speaks = [c for c in calls if c[0] == "speak" and c[2] >= 0]
    assert [c[1] for c in speaks] == [True, False] and speaks[1][2] >= 5.3 + 0.4 - 1e-9
    assert calls.index(("note", "가")) < calls.index(speaks[0])

def test_complete_flag(tmp_path):
    rep = judge(session(ITEMS), GOOD, GOOD, [-8.0])
    assert rep["complete"] and rep["pass"]
    cut = Session(wav=None, ments=(), items=tuple(ITEMS), complete=False)
    rep = judge(cut, GOOD, GOOD, [-8.0])
    assert not rep["complete"] and not rep["pass"]
    no_over = [i for i in ITEMS if i.kind != "overlap"]
    assert not judge(session(no_over), GOOD, GOOD, [-8.0])["complete"]
    d = {"wav": "s.wav", "ments": [], "items": [], "complete": False}
    p = tmp_path / "s.json"; p.write_text(json.dumps(d))
    assert load_session(p).complete is False

def test_frames_for_on_fails_closed_when_the_canceller_dies():
    class Bad:
        def __init__(self, **_): pass
        def process(self, mic, ref): raise ValueError("boom")
        def close(self): pass
    blocks = [np.full((BLOCK, 6), 1000, np.int16) for _ in range(3)]
    with pytest.raises(RuntimeError, match="도중에 꺼졌다"):
        frames_for(blocks, "on", EchoConfig("on"), factory=Bad)


def test_judge_user_only_stop_two_blocks_later_is_late():
    two = {**GOOD, "user_stop_1": res("user_stop_1", emer=1, fe=11.0 + 2 * DT)}
    rep = judge(session(ITEMS), GOOD, two, [-8.0])
    assert rep["user_only"]["late"] == ["user_stop_1"] and not rep["user_only"]["pass"] and not rep["pass"]
    one = {**GOOD, "user_stop_1": res("user_stop_1", emer=1, fe=11.0 + DT)}
    assert judge(session(ITEMS), GOOD, one, [-8.0])["user_only"]["late"] == []
    none = {**GOOD, "user_stop_1": res("user_stop_1", emer=1, fe=None)}
    assert judge(session(ITEMS), GOOD, none, [-8.0])["user_only"]["late"] == ["user_stop_1"]


def test_judge_user_only_zero_baseline_is_incomplete_with_a_note():
    assert judge(session(ITEMS), GOOD, GOOD, [-8.0])["notes"] == []
    zero = {**GOOD, "user_stop_1": res("user_stop_1")}
    rep = judge(session(ITEMS), zero, zero, [-8.0])
    assert not rep["complete"] and not rep["pass"] and len(rep["notes"]) == 1 and "멈춰" in rep["notes"][0]
    wake = ITEMS + [Item("user_wake_1", "user_only", "비카야 화장실 가자", "wake", 74, 78)]
    good = {**GOOD, "user_wake_1": res("user_wake_1", wakes=1, fw=75.0)}
    assert judge(session(wake), good, good, [-8.0])["complete"]
    deaf = {**GOOD, "user_wake_1": res("user_wake_1")}
    rep = judge(session(wake), deaf, deaf, [-8.0])
    assert not rep["complete"] and len(rep["notes"]) == 1 and "비카야" in rep["notes"][0]


def test_judge_without_user_only_stop_items_is_incomplete():     # "멈춰" 기준선이 아예 없으면 판정하지 않는다
    wake_only = [i for i in ITEMS if i.id != "user_stop_1"] + [Item("user_wake_1", "user_only", "비카야 화장실 가자", "wake", 74, 78)]
    good = {**GOOD, "user_wake_1": res("user_wake_1", wakes=1, fw=75.0)}
    rep = judge(session(wake_only), good, good, [-8.0])
    assert not rep["complete"] and not rep["pass"] and len(rep["notes"]) == 1 and "멈춰" in rep["notes"][0]


def test_judge_overlap_stop_zero_baseline_keeps_complete_and_adds_a_note():
    zero = {**GOOD, "over_stop_1": res("over_stop_1")}
    rep = judge(session(ITEMS), zero, zero, [-8.0])
    assert rep["overlap_stop"]["off"] == 0 and rep["complete"] and rep["pass"]
    assert len(rep["notes"]) == 1 and "겹침" in rep["notes"][0]


def test_judge_user_only_silenced_item_fails():
    gone = {**GOOD, "user_stop_1": res("user_stop_1", emer=1, fe=11.0, rms=0.0)}
    rep = judge(session(ITEMS), GOOD, gone, [-8.0])
    assert rep["user_only"]["silenced"] == ["user_stop_1"] and not rep["user_only"]["pass"] and not rep["pass"]
    assert rep["user_only"]["level_db_max"] == 0.0
    json.dumps(rep, allow_nan=False)
    assert judge(session(ITEMS), GOOD, GOOD, [-8.0])["user_only"]["silenced"] == []


def test_format_report_prints_notes_late_and_silenced():
    off = {**GOOD, "over_stop_1": res("over_stop_1")}
    on = {**off, "user_stop_1": res("user_stop_1", emer=1, fe=11.0 + 2 * DT, rms=0.0)}
    rep = judge(session(ITEMS), off, on, [-8.0])
    lines = format_report(rep).splitlines()
    assert rep["notes"] and all(any(n in line for line in lines) for n in rep["notes"])
    assert any("늦은" in line and "user_stop_1" in line for line in lines)
    assert any("지워진" in line and "user_stop_1" in line for line in lines)


def test_replay_tool_refuses_a_missing_env_file(tmp_path, monkeypatch, capsys):
    import dotenv
    import tools.mic_venue_check as mvc
    from tools.aec_replay import main
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)       # 시험 환경에 .env 를 올리지 않는다
    monkeypatch.setattr(mvc, "_load_models", lambda hint: pytest.fail("모델을 올리면 안 된다"))
    rc = main([str(tmp_path / "set.json"), "--env", str(tmp_path / "없음.env"), "--force"])
    assert rc != 0 and "없음.env" in capsys.readouterr().out


def test_replay_settings_record_what_the_run_used(tmp_path):
    from tools.aec_replay import run_settings
    env = tmp_path / "x.env"
    s = run_settings(env, EchoConfig("off", 120, 128), values_from({}))
    assert s["env"] == str(env.resolve()) and (s["delay_ms"], s["tail_ms"]) == (120, 128)
    assert s["values"]["wake_threshold"] == values_from({}).wake_threshold
    json.dumps(s, ensure_ascii=False, allow_nan=False)

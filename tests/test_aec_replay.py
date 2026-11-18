import json
import numpy as np
from src.aec_replay import (Item, ItemResult, Ment, Session, frames_for, judge, load_session, ment_reductions,
                            run_item, split_blocks)
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
    assert rep["pass"] and rep["overlap_stop"] == {"n": 1, "off": 1, "on": 1, "late": [], "pass": True}
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

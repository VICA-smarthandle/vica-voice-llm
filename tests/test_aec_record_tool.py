import json
from pathlib import Path
from tools.aec_record import OVERLAP_CLIPS, build_script, session_dict

MANIFEST = json.loads((Path(__file__).resolve().parents[1] / "assets/baked/manifest.json").read_text())

def test_script_matches_the_design_counts():
    steps = build_script()
    by = lambda kind, expect: [s for s in steps if s.kind == kind and s.expect == expect]
    assert len([s for s in steps if s.kind == "robot_only"]) == 16
    assert {s.item_id for s in steps if s.kind == "robot_only"} == {"robot_only"}
    assert [s.item_id for s in by("user_only", "wake")] == [f"user_wake_{i}" for i in range(1, 6)]
    assert [s.item_id for s in by("user_only", "emergency")] == [f"user_stop_{i}" for i in range(1, 6)]
    assert [s.item_id for s in by("overlap", "emergency")] == [f"over_stop_{i}" for i in range(1, 11)]
    assert [s.item_id for s in by("overlap", "wake")] == [f"over_wake_{i}" for i in range(1, 11)]
    assert {s.say for s in by("overlap", "wake")} == {"비카야"} and {s.say for s in by("overlap", "emergency")} == {"멈춰"}

def test_overlap_ments_never_block_the_listener():
    assert "mission_msg_obstacle_stop.wav" not in OVERLAP_CLIPS
    for s in build_script():
        if s.kind == "overlap":
            assert s.clip in OVERLAP_CLIPS and "비카야" not in MANIFEST[s.clip]

def test_session_dict_shape():
    d = session_dict("2026-11-19T10:00:00", "set.wav", {"agc": 0.007}, [{"name": "a.wav", "text": "가", "start": 2.0, "end": 5.0}],
                     [{"id": "over_stop_1", "kind": "overlap", "say": "멈춰", "expect": "emergency", "start": 2.0, "end": 7.0}])
    assert (d["version"], d["rate"], d["channels"], d["wav"]) == (1, 16000, 6, "set.wav")
    assert d["ments"][0]["text"] == "가" and d["items"][0]["expect"] == "emergency"
    json.dumps(d, ensure_ascii=False)


def test_session_dict_defaults_and_partial_flag():
    d = session_dict("t", "set.wav", {}, [], [])
    assert d["cues"] == [] and d["complete"] is True
    p = session_dict("t", "set.wav", {}, [], [], cues=[{"start": 1.0, "end": 1.15}], complete=False)
    assert p["cues"] == [{"start": 1.0, "end": 1.15}] and p["complete"] is False
    json.dumps(p, ensure_ascii=False)


def test_out_dir_problem_refuses_existing_set_wav(tmp_path):
    from tools.aec_record import out_dir_problem
    assert out_dir_problem(tmp_path) is None
    (tmp_path / "set.wav").write_bytes(b"")
    assert "set.wav" in out_dir_problem(tmp_path)


def test_close_robot_item_is_idempotent_and_needs_a_ment():
    from tools.aec_record import close_robot_item
    items = []
    close_robot_item(items, None, 0.0)
    assert items == []
    close_robot_item(items, 2.0, 40.0)
    close_robot_item(items, 2.0, 99.0)
    assert items == [{"id": "robot_only", "kind": "robot_only", "say": None, "expect": None,
                      "start": 2.0, "end": 40.5}]

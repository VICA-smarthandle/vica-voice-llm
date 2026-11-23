import ast
import json
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import numpy as np
import pytest
from tools.aec_record import OVERLAP_CLIPS, build_script, session_dict

MANIFEST = json.loads((Path(__file__).resolve().parents[1] / "assets/baked/manifest.json").read_text())
ROOT = Path(__file__).resolve().parents[1]

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


@pytest.mark.parametrize("name", ["set.wav", "set.json", "set.npy"])
def test_out_dir_problem_refuses_any_leftover_of_a_previous_set(tmp_path, name):
    from tools.aec_record import out_dir_problem
    (tmp_path / name).write_bytes(b"")
    assert name in out_dir_problem(tmp_path)


def test_out_dir_problem_allows_empty_or_missing_folder(tmp_path):
    from tools.aec_record import out_dir_problem
    assert out_dir_problem(tmp_path) is None and out_dir_problem(tmp_path / "없는폴더") is None


def test_close_robot_item_is_idempotent_and_needs_a_ment():
    from tools.aec_record import close_robot_item
    items = []
    close_robot_item(items, None, 0.0)
    assert items == []
    close_robot_item(items, 2.0, 40.0)
    close_robot_item(items, 2.0, 99.0)
    assert items == [{"id": "robot_only", "kind": "robot_only", "say": None, "expect": None,
                      "start": 2.0, "end": 40.5}]


def test_session_dict_carries_overflows():
    assert session_dict("t", "set.wav", {}, [], [])["overflows"] == 0
    assert session_dict("t", "set.wav", {}, [], [], overflows=3)["overflows"] == 3


def test_failed_chip_writes_names_each_failure():
    from tools.aec_record import failed_chip_writes
    chip = {"agc_desired": {"value": 0.01, "ok": True}, "agc_max_gain": {"value": 8.0, "ok": False},
            "echo_tuning": {"gamma_etail": 2.0, "nlatten": 1, "wrote": {}, "ok": False}}
    assert failed_chip_writes(chip) == ["agc_max_gain", "echo_tuning"]
    assert failed_chip_writes({"agc_desired": {"value": 0.01, "ok": True}}) == []


def test_echo_tuning_ok_needs_every_requested_value(monkeypatch):
    import src.dsp_state as dsp
    from tools.aec_record import apply_chip_from_env
    monkeypatch.setattr(dsp, "apply_agc_desired_level", lambda v: True)
    monkeypatch.setattr(dsp, "apply_agc_max_gain", lambda v: True)
    monkeypatch.setattr(dsp, "apply_echo_tuning", lambda e, n: {"NLATTENONOFF": 1.0})
    for key, value in (("VICA_MIC_AGC_DESIRED", "0.01"), ("VICA_MIC_AGC_MAX_GAIN", ""),
                       ("VICA_MIC_GAMMA_ETAIL", "2.0"), ("VICA_MIC_NLATTEN", "1")):
        monkeypatch.setenv(key, value)
    chip = apply_chip_from_env()
    assert chip["agc_desired"]["ok"] is True and chip["echo_tuning"]["ok"] is False
    monkeypatch.setattr(dsp, "apply_echo_tuning", lambda e, n: {"NLATTENONOFF": 1.0, "GAMMA_ETAIL": 2.0})
    assert apply_chip_from_env()["echo_tuning"]["ok"] is True


def _save(out_dir, chunks, complete=True, overflows=0):
    from tools.aec_record import _save_session
    return _save_session(out_dir, chunks, threading.Lock(), "t", {}, [], [], [], complete, None, None, 0.0,
                         overflows=overflows)


def test_save_writes_json_first_and_keeps_audio_as_npy_when_wav_fails(tmp_path, monkeypatch, capsys):
    import soundfile
    json_first = []

    def bad_write(path, data, *a, **k):
        json_first.append((tmp_path / "set.json").exists())
        raise OSError("디스크 가득")
    monkeypatch.setattr(soundfile, "write", bad_write)
    data = np.arange(12, dtype=np.int16).reshape(2, 6)
    assert _save(tmp_path, [data]) is False
    assert json_first == [True]
    assert np.array_equal(np.load(tmp_path / "set.npy"), data)
    assert json.loads((tmp_path / "set.json").read_text(encoding="utf-8"))["complete"] is True
    assert "set.npy" in capsys.readouterr().out


def test_failed_wav_write_leaves_no_truncated_wav_but_keeps_npy(tmp_path, monkeypatch):
    import soundfile

    def partial_write(path, data, *a, **k):
        Path(path).write_bytes(b"RIFF-partial")
        raise OSError("쓰기 중 끊김")
    monkeypatch.setattr(soundfile, "write", partial_write)
    data = np.arange(12, dtype=np.int16).reshape(2, 6)
    assert _save(tmp_path, [data]) is False
    assert not (tmp_path / "set.wav").exists()
    assert np.array_equal(np.load(tmp_path / "set.npy"), data)


def test_finally_ignores_exit_signals_before_saving():
    tree = ast.parse((ROOT / "tools" / "aec_record.py").read_text(encoding="utf-8"))
    rec = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "record")
    fin = next(n for n in ast.walk(rec) if isinstance(n, ast.Try) and n.finalbody).finalbody
    assert ast.unparse(fin[0]) == "_ignore_exit_signals()"
    assert any("_save_session" in ast.unparse(st) for st in fin[1:])
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_ignore_exit_signals")
    text = ast.unparse(helper)
    assert all(x in text for x in ("SIGINT", "SIGTERM", "SIGHUP", "signal.SIG_IGN"))


def test_ignore_exit_signals_survives_a_second_hangup():
    code = textwrap.dedent('''
        import os, signal
        import tools.aec_record as t
        t._ignore_exit_signals()
        for s in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
            os.kill(os.getpid(), s)
        print("survived", flush=True)
    ''')
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert "survived" in p.stdout, p.stdout + p.stderr


def test_save_json_failure_still_writes_wav_and_reports_failure(tmp_path):
    (tmp_path / "set.json").mkdir()
    assert _save(tmp_path, [np.zeros((160, 6), np.int16)]) is False
    assert (tmp_path / "set.wav").exists()


def test_save_success_records_overflows_and_survives_a_dead_terminal(tmp_path, monkeypatch, capsys):
    assert _save(tmp_path, [np.zeros((1600, 6), np.int16)], overflows=3) is True
    assert json.loads((tmp_path / "set.json").read_text(encoding="utf-8"))["overflows"] == 3
    assert (tmp_path / "set.wav").exists() and not (tmp_path / "set.npy").exists()
    assert "3번" in capsys.readouterr().out
    import tools.aec_record as rec

    def dead(*a, **k):
        raise OSError("터미널 끊김")
    monkeypatch.setattr(rec, "say", dead)
    other = tmp_path / "b"; other.mkdir()
    assert _save(other, [np.zeros((160, 6), np.int16)]) is True and (other / "set.wav").exists()


def test_save_without_audio_is_a_failure(tmp_path):
    assert _save(tmp_path, []) is False and not (tmp_path / "set.json").exists()


_SIGNAL_RUN = textwrap.dedent('''
    import os, signal, sys, time
    sys.modules["sounddevice"] = None          # 진짜 녹음이 불려도 장치를 열기 전에 실패한다
    sys.modules["usb"] = None
    import tools.aec_record as t
    t.voice_stack_running = lambda lines: []
    def fake_record(out_dir, countdown):
        try:
            os.kill(os.getpid(), getattr(signal, sys.argv[2]))
            time.sleep(5)
        except BaseException as exc:
            print("save-path", type(exc).__name__, flush=True)
            raise
        return 0
    t.record = fake_record
    sys.exit(t.main(["--out", sys.argv[1]]))
''')


@pytest.mark.parametrize("sig", ["SIGTERM", "SIGHUP"])
def test_hangup_or_kill_goes_through_the_save_path(sig, tmp_path):
    p = subprocess.run([sys.executable, "-c", _SIGNAL_RUN, str(tmp_path / "x"), sig], cwd=ROOT,
                       capture_output=True, text=True, timeout=60)
    assert "save-path KeyboardInterrupt" in p.stdout, p.stdout + p.stderr
    assert p.returncode != 0


@pytest.mark.parametrize("tool", ["tools.aec_record", "tools.aec_replay"])
def test_help_exits_0_without_device_or_model_imports(tool):
    code = textwrap.dedent(f'''
        import sys
        from {tool} import main
        try:
            main(["--help"])
        except SystemExit as e:
            print("RESULT", e.code, [m for m in ("sounddevice", "usb", "torch", "onnxruntime", "openwakeword",
                                                 "faster_whisper") if m in sys.modules])
    ''')
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert "RESULT 0 []" in p.stdout, p.stdout + p.stderr


def test_every_print_flushes():
    tree = ast.parse((ROOT / "tools" / "aec_record.py").read_text(encoding="utf-8"))
    bare = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == "print"
            and not any(k.arg == "flush" for k in n.keywords)]
    assert bare == []

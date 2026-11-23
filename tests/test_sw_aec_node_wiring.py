import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / "src" / "ros_wakeword_node.py").read_text())

def _method(name):
    return next(n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef) and n.name == name)

def _calls(node, dotted):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == dotted]

def test_stage_is_built_from_env_and_attached_before_the_mic_thread():
    init = _method("__init__")
    (build,) = _calls(init, "build_echo_stage")
    assert ast.unparse(build.args[0]) == "os.environ"
    attach = [n for n in ast.walk(init) if isinstance(n, ast.Assign)
              and ast.unparse(n.targets[0]) == "self._monitor.echo_stage"]
    assert len(attach) == 1
    (thread,) = _calls(init, "threading.Thread")
    assert build.lineno < attach[0].lineno < thread.lineno

def test_every_clip_row_goes_through_clip_info_with_aec():
    for name in ("_save_estop_clip", "_save_listen_clip", "_save_wake_clip"):
        assert len(_calls(_method(name), "clip_info_with_aec")) == 1, name

def test_env_example_and_test_defaults_name_the_switch():
    example = (ROOT / ".env.example").read_text()
    for key in ("VICA_SW_AEC=", "VICA_SW_AEC_DELAY_MS=", "VICA_SW_AEC_TAIL_MS="):
        assert key in example, key
    assert '"VICA_SW_AEC": "off"' in (ROOT / "tests" / "conftest.py").read_text()

def test_env_example_names_the_reduction_log():
    assert "~/vica_data/aec_log/YYYYMMDD.jsonl" in (ROOT / ".env.example").read_text()

def test_echo_cancel_imports_are_relative_and_in_order():
    assert "from src." not in (ROOT / "src" / "echo_cancel.py").read_text()
    node = (ROOT / "src" / "ros_wakeword_node.py").read_text()
    at = [node.index(f"from .{m} import") for m in ("dsp_state", "echo_cancel", "patrol_sound")]
    assert at == sorted(at)

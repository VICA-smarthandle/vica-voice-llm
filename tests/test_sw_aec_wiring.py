import ast
from pathlib import Path
import numpy as np
from src.wakeword_monitor import FRAME, WakewordMonitor, _queue_item

SRC = Path(__file__).resolve().parents[1] / "src" / "wakeword_monitor.py"

class Stage:
    def __init__(self): self.seen = []
    def feed(self, ch0, ch5):
        self.seen.append((int(ch0[0]), int(ch5[0]))); return np.full(len(ch0), 7, np.int16)

def monitor():
    return WakewordMonitor(on_emergency=lambda e: None, on_user_text=lambda t: None,
                           predict=lambda f: {"a": 0.0, "b": 0.0}, transcribe=lambda a: "")

def block6(v0, v5):
    b = np.zeros((FRAME, 6), np.int16); b[:, 0] = v0; b[:, 5] = v5; return b

def test_queue_item_off_is_plain_ch0():
    b = block6(11, 55)
    item = _queue_item(b, False)
    assert item.ndim == 1 and np.array_equal(item, b[:, 0]) and item.base is None

def test_queue_item_with_ref_carries_ch0_and_ch5():
    item = _queue_item(block6(11, 55), True)
    assert item.shape == (FRAME, 2) and item[0, 0] == 11 and item[0, 1] == 55

def test_plain_frame_passes_untouched():
    m = monitor(); f = np.zeros(FRAME, np.int16)
    assert m.echo_stage is None and m._frame_from_queue(f) is f

def test_pair_goes_through_stage_even_while_muted_and_in_order():   # 설계 4.1 · Review Focus 1
    m = monitor(); m.echo_stage = Stage(); m.set_muted(True, now=0.0)
    for i in range(40):                                              # 밀린 블록 40개가 한꺼번에 풀린다
        f = m._frame_from_queue(_queue_item(block6(i, 100 + i), True))
        assert f[0] == 7
        m.process_frame(f, now=i * 0.08)
    assert m.echo_stage.seen == [(i, 100 + i) for i in range(40)]

def test_pair_without_stage_falls_back_to_ch0():
    m = monitor()
    assert np.array_equal(m._frame_from_queue(_queue_item(block6(11, 55), True)), np.full(FRAME, 11, np.int16))

def test_run_uses_both_helpers_before_process_frame():
    tree = ast.parse(SRC.read_text())
    run = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run")
    calls = lambda name: [n for n in ast.walk(run) if isinstance(n, ast.Call) and ast.unparse(n.func) == name]
    assert len(calls("_queue_item")) == 1 and len(calls("self._frame_from_queue")) == 1
    assert calls("self._frame_from_queue")[0].lineno < calls("self.process_frame")[0].lineno

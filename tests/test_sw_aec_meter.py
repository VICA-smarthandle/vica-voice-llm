import json
import numpy as np
from src.echo_meter import EchoMeter, WEAK_WARNING

B = 1280
def blk(v): return np.full(B, v, np.int16)

def feed(m, n, mic, out, ref):
    return [r for _ in range(n) if (r := m.update(blk(mic), blk(out), blk(ref))) is not None]

def test_one_row_when_robot_chunk_ends(tmp_path):
    m = EchoMeter("log", tmp_path, lambda s: None, wall=lambda: 1795000000.0)
    assert feed(m, 25, 4000, 1000, 5000) == []
    (row,) = feed(m, 8, 10, 10, 0)
    assert row["mode"] == "log" and abs(row["sec"] - 2.0) < 0.01
    assert abs(row["reduce_db"] - (-12.04)) < 0.05
    assert abs(row["rms_in"] - 4000 / 32768) < 1e-4
    (f,) = tmp_path.glob("*.jsonl")
    assert len(f.stem) == 8 and json.loads(f.read_text().splitlines()[0]) == row

def test_short_chunk_is_not_logged(tmp_path):
    m = EchoMeter("on", tmp_path, lambda s: None)
    assert feed(m, 10, 4000, 1000, 5000) + feed(m, 8, 10, 10, 0) == []
    assert list(tmp_path.glob("*.jsonl")) == []

def test_gap_shorter_than_hang_joins_chunks(tmp_path):
    m = EchoMeter("on", None, lambda s: None)
    rows = feed(m, 15, 4000, 1000, 5000) + feed(m, 5, 10, 10, 0) + feed(m, 15, 4000, 1000, 5000) + feed(m, 8, 10, 10, 0)
    assert len(rows) == 1 and abs(rows[0]["sec"] - 2.4) < 0.01

def test_weak_warning_once_then_rearms_after_recovery():
    warns = []
    m = EchoMeter("on", None, warns.append)
    def chunk(out): feed(m, 40, 4000, out, 5000); feed(m, 8, 10, 10, 0)
    for _ in range(4): chunk(3900)
    assert warns == []
    chunk(3900); chunk(3900)
    assert warns == [WEAK_WARNING]
    chunk(1000)
    for _ in range(5): chunk(3900)
    assert warns == [WEAK_WARNING, WEAK_WARNING]

def test_chunks_under_3s_do_not_count_as_weak():
    warns = []
    m = EchoMeter("on", None, warns.append)
    for _ in range(8): feed(m, 25, 4000, 3900, 5000); feed(m, 8, 10, 10, 0)
    assert warns == []

def test_unwritable_log_dir_does_not_raise(tmp_path):
    notdir = tmp_path / "file"; notdir.write_text("x")
    warns = []
    m = EchoMeter("on", notdir, warns.append)
    for _ in range(2): feed(m, 25, 4000, 1000, 5000); rows = feed(m, 8, 10, 10, 0)
    assert len(rows) == 1 and len(warns) == 1

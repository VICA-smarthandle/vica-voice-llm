"""시운전용 장애물 안내 점검 도구(scripts/avoid_cue.py)의 ROS 없이 시험할 수 있는 부분.

판정 자체는 tests/test_obstacle_judge.py 가 본다. 여기서는 라이다 점 바꾸기와 옛 녹화본의
RobotState 앞 칸 풀기만 본다.
"""
import importlib.util
import math
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "avoid_cue.py"
_spec = importlib.util.spec_from_file_location("avoid_cue", _PATH)
ac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ac)


def scan(ranges, angle_min=-math.pi / 2, inc=math.pi / 2, rmin=0.15, rmax=12.0):
    return SimpleNamespace(ranges=ranges, angle_min=angle_min, angle_increment=inc, range_min=rmin, range_max=rmax)


class TestScanPoints:
    def test_points_in_robot_frame_with_laser_offset(self):
        # 오른쪽(-90°) 1 m, 앞(0°) 2 m, 왼쪽(+90°) 1 m — 라이다는 차체 기준 x +0.031
        x, y = ac.scan_points(scan([1.0, 2.0, 1.0]), offset=(0.031, 0.0))
        assert x == pytest.approx([0.031, 2.031, 0.031], abs=1e-9)
        assert y == pytest.approx([-1.0, 0.0, 1.0], abs=1e-9)

    def test_drops_invalid_and_far_points(self):
        x, y = ac.scan_points(scan([float("inf"), 0.1, 5.0], angle_min=0.0, inc=0.1))
        assert len(x) == 0   # inf·최소 거리 미만·앞 4 m 밖

    def test_keeps_only_box_around_robot(self):
        x, y = ac.scan_points(scan([2.0, 2.0], angle_min=math.pi / 2, inc=math.pi / 2))
        assert len(x) == 0   # 옆 2 m(상자 ±1.5 m 밖)와 뒤 2 m(상자 -0.3 m 밖)


def _cdr_robot_state(dialog: str, building: str = "B1", extra: bytes = b"") -> bytes:
    """vica_interfaces/RobotState 앞 칸(little endian CDR): int32, string, bool, bool, string + 뒤 칸."""
    def cdr_str(s, off):
        b = s.encode() + b"\0"
        pad = (-off) % 4
        return b"\0" * pad + struct.pack("<I", len(b)) + b, off + pad + 4 + len(b)

    body, off = struct.pack("<i", 2), 4
    s, off = cdr_str(building, off)
    body += s
    body += b"\x01\x00"
    off += 2
    s, off = cdr_str(dialog, off)
    body += s + extra
    return b"\x00\x01\x00\x00" + body


class TestRobotState:
    @pytest.mark.parametrize("building", ["B1", "별빛관", "x" * 7])
    def test_dialog_from_old_and_new_layouts(self, building):
        assert ac._robot_state_dialog(_cdr_robot_state("navigating", building)) == "navigating"
        assert ac._robot_state_dialog(_cdr_robot_state("returning", building, extra=b"\x07" * 9)) == "returning"

"""SimRobot 순수 로직 검증 — 이동·도착·E-stop 래치의 안전 의미론."""
from __future__ import annotations

from src.robot_sim import DEFAULT_TRAVEL_SEC, FLOOR_CHANGE_SEC, SimRobot
from src.schema import VicaIntent


class Pose:
    def __init__(self, x=0.0, y=0.0):
        self.x, self.y, self.yaw = x, y, 0.0
        self.frame_id = "map"


class Dest:
    def __init__(self, id, floor=1, building="starlight_building",
                 pose=None, arrival_message="", name="목적지"):
        self.id, self.floor, self.building = id, floor, building
        self.pose = pose or Pose()
        self.arrival_message = arrival_message
        self.name = name


def navigate(dest_id, need_confirm=False, safety="normal"):
    return VicaIntent(intent="navigate", matched_destination_id=dest_id,
                      need_confirm=need_confirm, reply="", safety_flag=safety,
                      confidence=0.9)


def make():
    return SimRobot([
        Dest("restroom", floor=1, arrival_message="화장실 앞에 도착했습니다."),
        Dest("office_4f", floor=4),
        Dest("far_room", floor=1, pose=Pose(x=8.0, y=0.0)),
    ])


def test_confirmed_navigate_starts_move():
    sim = make()
    assert sim.handle_intent(navigate("restroom"), now=0.0) == "move_started"
    assert sim.is_moving


def test_unconfirmed_or_unknown_does_not_move():
    sim = make()
    assert sim.handle_intent(navigate("restroom", need_confirm=True), now=0.0) is None
    assert sim.handle_intent(navigate("no_such_place"), now=0.0) is None
    assert not sim.is_moving


def test_arrival_after_travel_time():
    sim = make()
    sim.handle_intent(navigate("restroom"), now=0.0)
    assert sim.tick(now=DEFAULT_TRAVEL_SEC - 0.5) is None       # 아직 이동 중
    event = sim.tick(now=DEFAULT_TRAVEL_SEC + 0.1)
    assert event and event["kind"] == "arrived"
    assert event["message"] == "화장실 앞에 도착했습니다."
    assert not sim.is_moving


def test_floor_change_adds_time():
    sim = make()
    sim.handle_intent(navigate("office_4f"), now=0.0)
    assert sim.tick(now=DEFAULT_TRAVEL_SEC + 1.0) is None       # 승강기 시간만큼 더 걸림
    assert sim.tick(now=DEFAULT_TRAVEL_SEC + FLOOR_CHANGE_SEC + 0.1) is not None
    assert sim.floor == 4


def test_real_pose_uses_distance():
    sim = make()
    sim.handle_intent(navigate("far_room"), now=0.0)             # 8m / 0.8m/s = 10초
    assert sim.tick(now=9.0) is None
    assert sim.tick(now=10.1) is not None


def test_emergency_latches_and_blocks():
    sim = make()
    sim.handle_intent(navigate("restroom"), now=0.0)
    sim.handle_emergency(now=1.0)
    assert sim.state == "estopped" and not sim.is_moving
    # 래치 중 새 이동 거부 + 시간이 지나도 자동 재개 없음
    assert sim.handle_intent(navigate("restroom"), now=2.0) == "blocked_estop"
    assert sim.tick(now=100.0) is None
    assert sim.state == "estopped"


def test_reset_clears_latch_but_discards_goal():
    sim = make()
    sim.handle_intent(navigate("restroom"), now=0.0)
    sim.handle_emergency(now=1.0)
    sim.reset(now=2.0)
    assert sim.state == "idle"
    assert sim.tick(now=100.0) is None      # 이전 목적지가 되살아나지 않는다

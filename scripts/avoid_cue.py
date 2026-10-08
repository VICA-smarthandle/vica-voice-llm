#!/usr/bin/env python3
"""시운전용 장애물 안내 점검 도구 — 앞에 진짜 물체가 있을 때만 한 번 말한다.

설계서: 루트 docs/superpowers/specs/2026-10-08-obstacle-narration-design.md (10-08 사용자 승인).
판정은 scripts/obstacle_judge.py(순수 코드)에 있고, 이 파일은 ROS 입력·소리·기록만 맡는다.
로봇에는 아무것도 보내지 않는다(구독만).

로봇이 크게 비키거나(차선 옮김·경로 우회) 급히 줄이거나 설 때(급감속·장애물 정지·충돌감시),
그 순간 0.5초 안에 라이다·깊이 카메라가 가던 길 위에서 지도에 없는 물체를 보면 말한다.
  비켜 갈 때      "앞에 장애물이 있어 피해 갈게요."
  줄이거나 설 때  "앞에 장애물이 있어 천천히 갈게요."
안내 주행(미션 navigating) 중에만, 목적지 1 m 밖에서만, 한 장애물에 한 번(6초 + 평상 주행 2초 뒤 다시).

LLM·음성 스택(⑫)을 내린 뒤, 같은 칸(음성 저장소·ROS 환경)에서 실행한다:

    .venv/bin/python scripts/avoid_cue.py run71

    --force   음성 스택이 떠 있어도 실행(스피커를 두고 다툰다)
    --map     정지 지도 yaml (기본: 환경변수 VICA_MAP_YAML, 없으면 /map 토픽)
    -v        말하지 않은 판정(안내 주행 아님·목적지 앞 등)도 화면에 보인다

지난 녹화본으로 소리 없이 판정만 돌려 볼 수 있다(설계서 7.2 재현):

    .venv/bin/python scripts/avoid_cue.py --bag ~/vica_data/bags/1008_run69_map_1002_150946_waitspot

모든 행동과 판정은 ~/vica_data/marks/<이름>_obstacle.csv 에 한 줄씩 남는다(녹화본 모드는 --csv 일 때).
scripts/vica_mark.py 로 찍은 표시와 같은 시계(epoch)라 그대로 맞춰 볼 수 있다.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import math
import os
import queue
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import obstacle_judge as oj  # noqa: E402

SOUND_DIR = ROOT / "assets" / "obstacle_narration"
SPEED = 1.2                  # 사용자가 고른 말 속도(Supertonic-3 F2)
LASER_DEFAULT = (0.031, 0.0)  # base_footprint → laser_frame (tf_static 실측)
MAPS_DIR = Path.home() / "VICA-smarthandle" / "vica_ros2_ws" / "maps"
KIND_KO = {"LAT": "차선 옮김", "DET": "경로 우회", "DEC": "급감속", "HOLD": "장애물 정지", "CM": "충돌감시"}
DECISION_KO = {"announce": "말함", "merged": "묶음(같은 장애물)", "excluded": "뺌", "no_cause": "원인 없음",
               "no_map": "지도 없음"}
WHY_KO = {"not_guided": "안내 주행 아님", "resync": "유턴 뒤 자리 맞추기", "after_turn": "유턴 직후",
          "near_goal": "목적지 1 m 안", "cm_not_driving": "회전·정지 중 충돌감시", "dec_from_lane_shift": "차선 옮김 감속"}


# ------------------------------------------------------------------ 소리

def _load_sounds() -> dict:
    """고른 두 녹음. 없으면 같은 목소리·속도로 그 자리에서 만든다."""
    import soundfile as sf
    sounds = {}
    for key in oj.PHRASES:
        path = SOUND_DIR / f"{key}.wav"
        if path.exists():
            wave, rate = sf.read(str(path), dtype="float32")
            sounds[key] = (wave, rate)
    missing = [k for k in oj.PHRASES if k not in sounds]
    if missing:
        import supertonic
        import supertonic.loader
        supertonic.loader.DEFAULT_ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print(f"녹음이 없어 만드는 중: {missing} (Supertonic-3 F2, {SPEED}배)", flush=True)
        tts = supertonic.TTS(model="supertonic-3", auto_download=True)
        style = tts.get_voice_style("F2")
        for k in missing:
            wave, _ = tts.synthesize(oj.PHRASES[k], style, lang="ko", speed=SPEED)
            sounds[k] = (np.asarray(wave, dtype=np.float32).squeeze(), int(tts.sample_rate))
    return sounds


class Speaker:
    """말을 차례로 튼다. 밀린 말은 2.5초 지나면 버린다(늦은 말은 헷갈리게 한다)."""

    STALE_SEC = 2.5

    def __init__(self):
        sys.path.insert(0, str(ROOT))
        self.sounds = _load_sounds()
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, key: str) -> None:
        self.q.put((time.time(), key))

    def _run(self) -> None:
        from src import audio_out
        while True:
            t, key = self.q.get()
            if time.time() - t > self.STALE_SEC:
                continue
            wave, rate = self.sounds[key]
            try:
                audio_out.play(wave, rate, blocking=True)
            except Exception as e:  # noqa: BLE001 — 점검 도구는 멈추지 않고 알리기만 한다
                print(f"\033[0;31m재생 실패: {e}\033[0m", flush=True)


def voice_stack_running() -> list:
    """음성 스택 파이썬 프로세스(pid, 모듈). 셸 명령줄에 글자만 들어간 것은 빼려고 argv 를 본다."""
    found = []
    for d in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            argv = open(d, "rb").read().split(b"\0")
        except OSError:
            continue
        if not argv or b"python" not in os.path.basename(argv[0]):
            continue
        for mod in (b"src.ros_tts_node", b"src.ros_node", b"src.ros_wakeword_node"):
            if mod in argv:
                found.append((d.split("/")[2], mod.decode()))
    return found


# ------------------------------------------------------------------ 기록·화면

def _clock(t: float) -> str:
    return datetime.datetime.fromtimestamp(t).strftime("%H:%M:%S.%f")[:-4]


class DecisionLog:
    HEAD = ["epoch", "시각", "행동", "판정", "문장", "뺀 이유", "점", "라이다", "깊이", "벽 점", "앞(m)", "옆(m)",
            "기준", "목적지(m)", "상세"]

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        new = not path.exists()
        self.f = open(path, "a", newline="", buffering=1)
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(self.HEAD)

    def write(self, d: dict) -> None:
        self.w.writerow([f"{d['t']:.3f}", _clock(d["t"]), KIND_KO[d["kind"]],
                         DECISION_KO.get(d["decision"], d["decision"]),
                         oj.PHRASES[d["phrase"]] if d["phrase"] else "", "·".join(WHY_KO.get(w, w) for w in d["why"]),
                         d["n"], d["n_scan"], d["n_depth"], d["n_wall"], "" if d["near"] is None else d["near"],
                         "" if d["lat"] is None else d["lat"], d["mode"] or "", d["goal_dist"],
                         json.dumps(d["detail"], ensure_ascii=False)])


def show(d: dict, verbose: bool) -> None:
    kind, dec = KIND_KO[d["kind"]], d["decision"]
    where = "" if d["near"] is None else f" · 앞 {d['near']:.2f} m 옆 {d['lat']:+.2f} m"
    pts = f"점 {d['n']}(라이다 {d['n_scan']}·깊이 {d['n_depth']})"
    if dec == "announce":
        print(f"\033[1;33m{_clock(d['t'])}  🔊 {oj.PHRASES[d['phrase']]}\033[0m  {kind} · {pts}{where}", flush=True)
    elif dec in ("merged", "no_cause"):
        print(f"\033[2m{_clock(d['t'])}  · {kind} → {DECISION_KO[dec]} · {pts}{where}\033[0m", flush=True)
    elif verbose:
        why = "·".join(WHY_KO.get(w, w) for w in d["why"])
        print(f"\033[2m{_clock(d['t'])}  · {kind} → {DECISION_KO.get(dec, dec)} ({why})\033[0m", flush=True)


# ------------------------------------------------------------------ 점·자세

def scan_points(msg, offset=(0.0, 0.0)):
    """LaserScan → base_footprint 기준 (x, y). 앞 -0.3~4.0 m, 옆 ±1.5 m 만."""
    r = np.asarray(msg.ranges, dtype=float)
    a = msg.angle_min + np.arange(len(r)) * msg.angle_increment
    ok = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)
    x = offset[0] + r[ok] * np.cos(a[ok])
    y = offset[1] + r[ok] * np.sin(a[ok])
    keep = (x >= -0.3) & (x <= 4.0) & (np.abs(y) <= 1.5)
    return x[keep], y[keep]


def _yaw(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def load_map(arg: Optional[str]) -> Optional[oj.MapGrid]:
    path = arg or os.environ.get("VICA_MAP_YAML")
    if not path:
        return None
    path = os.path.expanduser(path)
    grid = oj.MapGrid.from_yaml(path)
    print(f"지도: {path}", flush=True)
    return grid


# ------------------------------------------------------------------ 실시간

def run_live(args) -> None:
    import rclpy
    import tf2_ros
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
    from std_msgs.msg import String
    from nav_msgs.msg import Path as PathMsg, OccupancyGrid
    from sensor_msgs.msg import LaserScan
    from rcl_interfaces.msg import Log
    from nav2_msgs.msg import BehaviorTreeLog
    from vica_interfaces.msg import RobotState

    busy = voice_stack_running()
    if busy and not args.force:
        print("\033[0;31m음성 스택이 떠 있습니다 — 스피커를 두고 다툽니다. ⑫ 칸을 Ctrl+C 로 내린 뒤 다시 실행하세요.\033[0m")
        for pid, mod in busy:
            print(f"  pid {pid}  {mod}")
        print("그래도 실행하려면 --force")
        sys.exit(1)

    speaker = Speaker()
    log = DecisionLog(Path.home() / "vica_data" / "marks" / f"{args.name}_obstacle.csv")
    judge = oj.ObstacleJudge(load_map(args.map))

    rclpy.init()
    node = Node("vica_obstacle_cue")
    tf_buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buf, node)
    laser = {"off": None}
    counts = {"announce": 0, "actions": 0}
    seen = {k: 0 for k in ("scan", "depth", "vcc", "rail", "state")}
    status = {"next": time.time() + 5.0, "since": time.time(), "pose": False}

    def now() -> float:
        return time.time()

    def show_status(t: float) -> None:
        """입력이 제대로 들어오는지 한 줄로 — 깊이 카메라·라이다가 끊기면 원인을 못 본다."""
        dt = max(t - status["since"], 1e-6)
        hz = {k: v / dt for k, v in seen.items()}
        ok = lambda f, good: f"\033[0;32m{f}\033[0m" if good else f"\033[0;31m{f}\033[0m"  # noqa: E731
        print("입력: " + " · ".join([
            ok(f"라이다 {hz['scan']:.0f}Hz", hz["scan"] >= 5),
            ok(f"깊이 {hz['depth']:.0f}Hz", hz["depth"] >= 3),
            f"VCC {hz['vcc']:.0f}Hz" + ("" if hz["vcc"] >= 1 else "(주행 중 아님)"),
            f"레일 {seen['rail']}번",
            ok("위치 OK" if status["pose"] else "위치 없음", status["pose"]),
            ok("지도 OK" if judge.grid is not None else "지도 없음", judge.grid is not None),
            f"미션 {judge.dialog or '?'}"]), flush=True)
        for k in seen:
            seen[k] = 0
        status["since"], status["next"] = t, t + 60.0

    def update_pose(t: float) -> None:
        try:
            tr = tf_buf.lookup_transform("map", args.base_frame, rclpy.time.Time())
        except Exception:  # noqa: BLE001 — 위치를 아직 모르면 원인 판정만 쉰다
            return
        p, q = tr.transform.translation, tr.transform.rotation
        judge.on_pose(t, p.x, p.y, _yaw(q))
        status["pose"] = True

    def laser_offset():
        if laser["off"] is None:
            try:
                tr = tf_buf.lookup_transform(args.base_frame, "laser_frame", rclpy.time.Time())
                laser["off"] = (tr.transform.translation.x, tr.transform.translation.y)
            except Exception:  # noqa: BLE001
                return LASER_DEFAULT
        return laser["off"]

    def on_scan(msg):
        t = now()
        seen["scan"] += 1
        update_pose(t)
        judge.on_points("scan", t, *scan_points(msg, laser_offset()))

    def on_depth(msg):
        if msg.header.frame_id not in ("", args.base_frame):
            return   # depth_band_to_scan 은 base_footprint 로 낸다(10-08 확인). 다른 틀이면 쓰지 않는다
        seen["depth"] += 1
        judge.on_points("depth", now(), *scan_points(msg))

    def on_state(msg):
        st = oj.parse_state(msg.data)
        if st is not None:
            seen["vcc"] += 1
            judge.on_vcc(now(), st)

    def on_bt(msg):
        for e in msg.event_log:
            judge.on_bt(e.timestamp.sec + e.timestamp.nanosec * 1e-9, e.node_name, e.current_status)

    def on_rosout(msg):
        if msg.name == "collision_monitor":
            what = oj.cm_what(msg.msg)
            if what:
                judge.on_cm(now(), what)

    def on_rail(msg):
        seen["rail"] += 1
        judge.on_rail(now(), [(p.pose.position.x, p.pose.position.y) for p in msg.poses])

    def on_goal(msg):
        ev = oj.goal_event(msg.data)
        if ev:
            judge.on_goal(now(), ev["event"], ev["x"], ev["y"], ev["loc"])

    def on_robot_state(msg):
        seen["state"] += 1
        judge.on_dialog(now(), msg.dialog_state)

    def on_map(msg):
        if judge.grid is None:
            i = msg.info
            judge.grid = oj.MapGrid.from_occupancy(msg.data, i.width, i.height, i.resolution,
                                                   i.origin.position.x, i.origin.position.y)
            print(f"지도: /map 토픽 ({i.width}×{i.height}, {i.resolution} m)", flush=True)

    def on_timer():
        t = now()
        update_pose(t)
        if t >= status["next"]:
            show_status(t)
        for d in judge.tick(t):
            counts["actions"] += 1
            log.write(d)
            show(d, args.verbose)
            if d["decision"] == "announce":
                counts["announce"] += 1
                speaker.say(d["phrase"])

    node.create_subscription(String, "/vcc/state", on_state, 10)
    node.create_subscription(BehaviorTreeLog, "/behavior_tree_log", on_bt, 50)
    node.create_subscription(Log, "/rosout", on_rosout, 100)
    node.create_subscription(LaserScan, "/scan", on_scan, qos_profile_sensor_data)
    node.create_subscription(LaserScan, "/camera/depth_scan", on_depth, qos_profile_sensor_data)
    node.create_subscription(PathMsg, "/rail_plan", on_rail, 10)
    node.create_subscription(String, "/vica_goal_event", on_goal, 10)
    node.create_subscription(RobotState, "/vica/robot_state", on_robot_state, 10)
    if judge.grid is None:
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        node.create_subscription(OccupancyGrid, "/map", on_map, latched)
        print("지도: /map 토픽을 기다립니다", flush=True)
    node.create_timer(0.1, on_timer)

    print(f"장애물 안내 점검 시작 — 기록 {log.path}", flush=True)
    print("말하는 때: 안내 주행 중 크게 비키거나 급히 줄일 때, 앞에 지도에 없는 물체가 보이면 한 번. Ctrl+C 로 끝.",
          flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n행동 {counts['actions']}번 판정, 그중 말함 {counts['announce']}번 — 기록 {log.path}", flush=True)
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


# ------------------------------------------------------------------ 녹화본

def _robot_state_dialog(data: bytes) -> str:
    """vica_interfaces/RobotState CDR 의 앞 칸만 푼다(int32 floor, string building, bool, bool, string dialog).

    10-07 에 칸이 늘어 옛 녹화본은 지금 메시지 정의로 못 푼다 — 앞 칸 순서는 그대로다.
    """
    e = "<" if data[1] == 1 else ">"
    buf, off = data[4:], 4          # int32 floor
    n = struct.unpack_from(e + "I", buf, off)[0]
    off += 4 + n                    # building
    off += 2                        # is_moving, is_paused
    off = (off + 3) // 4 * 4
    n = struct.unpack_from(e + "I", buf, off)[0]
    off += 4
    return buf[off:off + max(n - 1, 0)].decode("utf-8", "replace")


def _bag_map_id(reader) -> Optional[str]:
    from rclpy.serialization import deserialize_message
    from std_msgs.msg import String
    r, _ = reader(["/vica_goal_event"])
    while r.has_next():
        _, data, _ = r.read_next()
        ev = oj.goal_event(deserialize_message(data, String).data)
        if ev and ev.get("map_id"):
            return ev["map_id"]
    return None


def run_bag(args) -> None:
    import rosbag2_py
    import yaml
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    bag = os.path.expanduser(args.bag.rstrip("/"))
    meta = yaml.safe_load(open(f"{bag}/metadata.yaml"))["rosbag2_bagfile_information"]

    def reader(topics):
        r = rosbag2_py.SequentialReader()
        r.open(rosbag2_py.StorageOptions(uri=bag, storage_id=meta["storage_identifier"]),
               rosbag2_py.ConverterOptions("cdr", "cdr"))
        types = {t.name: t.type for t in r.get_all_topics_and_types()}
        r.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics if t in types]))
        return r, types

    grid = None
    if args.map:
        grid = load_map(args.map)
    else:
        map_id = _bag_map_id(reader)
        if map_id and (MAPS_DIR / f"{map_id}.yaml").exists():
            grid = load_map(str(MAPS_DIR / f"{map_id}.yaml"))
    if grid is None:
        sys.exit("지도를 정할 수 없습니다 — --map 으로 yaml 을 주세요.")

    want = ["/odom", "/tf", "/tf_static", "/vcc/state", "/behavior_tree_log", "/rosout", "/scan",
            "/camera/depth_scan", "/rail_plan", "/vica_goal_event", "/vica/robot_state"]
    r, types = reader(want)
    if "/vcc/state" not in types:
        sys.exit("이 녹화본에는 /vcc/state 가 없습니다.")
    cls = {t: get_message(types[t]) for t in want if t in types and t != "/vica/robot_state"}

    judge = oj.ObstacleJudge(grid)
    log = DecisionLog(Path(args.csv).expanduser()) if args.csv else None
    laser = list(LASER_DEFAULT)
    map_odom = None
    said, decided = [], []
    first = last = None

    def handle(ds):
        for d in ds:
            decided.append(d)
            if log:
                log.write(d)
            if d["decision"] == "announce":
                said.append(d)
            if not args.quiet:
                show(d, args.verbose)

    while r.has_next():
        topic, data, ns = r.read_next()
        t = ns / 1e9
        first = t if first is None else first
        last = t
        if topic == "/vica/robot_state":
            try:
                judge.on_dialog(t, _robot_state_dialog(data))
            except (struct.error, IndexError):
                pass
            handle(judge.tick(t))
            continue
        msg = deserialize_message(data, cls[topic])
        if topic == "/tf":
            for tr in msg.transforms:
                if tr.header.frame_id == "map" and tr.child_frame_id == "odom":
                    p = tr.transform.translation
                    map_odom = (p.x, p.y, _yaw(tr.transform.rotation))
        elif topic == "/tf_static":
            for tr in msg.transforms:
                if tr.child_frame_id == "laser_frame":
                    laser = [tr.transform.translation.x, tr.transform.translation.y]
        elif topic == "/odom":
            if map_odom is not None:
                p, q = msg.pose.pose.position, msg.pose.pose.orientation
                mx, my, myaw = map_odom
                c, s = math.cos(myaw), math.sin(myaw)
                judge.on_pose(t, mx + c * p.x - s * p.y, my + s * p.x + c * p.y, myaw + _yaw(q))
        elif topic == "/vcc/state":
            st = oj.parse_state(msg.data)
            if st is not None:
                judge.on_vcc(t, st)
        elif topic == "/behavior_tree_log":
            for e in msg.event_log:
                judge.on_bt(e.timestamp.sec + e.timestamp.nanosec * 1e-9, e.node_name, e.current_status)
        elif topic == "/rosout":
            if msg.name == "collision_monitor":
                what = oj.cm_what(msg.msg)
                if what:
                    judge.on_cm(t, what)
        elif topic == "/scan":
            judge.on_points("scan", t, *scan_points(msg, laser))
        elif topic == "/camera/depth_scan":
            judge.on_points("depth", t, *scan_points(msg))
        elif topic == "/rail_plan":
            judge.on_rail(t, [(p.pose.position.x, p.pose.position.y) for p in msg.poses])
        elif topic == "/vica_goal_event":
            ev = oj.goal_event(msg.data)
            if ev:
                judge.on_goal(t, ev["event"], ev["x"], ev["y"], ev["loc"])
        handle(judge.tick(t))
    handle(judge.tick(math.inf))

    mins = (last - first) / 60 if first is not None else 0
    kinds = {}
    for d in decided:
        kinds[d["decision"]] = kinds.get(d["decision"], 0) + 1
    print(f"\n녹화 {mins:.1f}분 · 행동 {len(decided)}번 · 판정 {kinds}")
    print(f"말함 {len(said)}번:")
    for d in said:
        print(f"  {_clock(d['t'])}  {KIND_KO[d['kind']]:<6}  {oj.PHRASES[d['phrase']]}  (앞 {d['near']} m)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name", nargs="?", default=datetime.datetime.now().strftime("obstacle_%m%d_%H%M"),
                    help="기록 파일 이름 (예: run71)")
    ap.add_argument("--bag", help="녹화본 폴더 — 소리 없이 판정만 출력")
    ap.add_argument("--map", help="정지 지도 yaml (기본: VICA_MAP_YAML → /map 토픽, 녹화본은 goal 의 map_id)")
    ap.add_argument("--csv", help="녹화본 모드에서 판정을 이 파일에 쓴다")
    ap.add_argument("--force", action="store_true", help="음성 스택이 떠 있어도 실행")
    ap.add_argument("--quiet", action="store_true", help="녹화본 모드에서 마지막 요약만 보인다")
    ap.add_argument("-v", "--verbose", action="store_true", help="말하지 않은 판정도 보인다")
    ap.add_argument("--base-frame", default="base_footprint")
    args = ap.parse_args()
    if args.bag:
        run_bag(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()

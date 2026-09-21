# P1 로봇 대장(화이트보드) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 미션 매니저가 "몇 층·지금 어디·가는 중·직전 도착·하려다 만 곳·대화 단계·대기·시각·배터리"를 적어 `/vica/robot_state` 로 방송하고, LLM 노드가 그것을 읽어 지시문 맨 뒤 `[지금 상황]` 블록으로 모델에게 준다. 다른 층 장소는 건물 디렉터리 파일로 안다.

**Architecture:** ROS 쪽은 (1) `RobotState.msg` 칸 추가, (2) 순수 파이썬 모듈 `map_meta.py`(지도 한 장 = 건물·층)와 `ledger.py`(대장 자료형·좌표→"OO 앞/사이" 계산·JSON 보존·메시지 칸 파생), (3) 미션 로직의 대기 접근자 2개, (4) 노드가 goal 사건·tick 전이·AMCL 좌표로 대장을 채워 방송. 음성 쪽은 (5) pydantic `RobotState` 칸 추가와 변환, (6) `ledger_view.py` 가 방송을 `[지금 상황]` 블록으로 렌더(옛 미션이면 기존 `SituationBoard` 로 폴백), (7) `building_directory.py` 가 `directory.yaml` 을 읽어 "[다른 층 장소]" 블록, (8) 지시문 규칙. 판단은 어디에도 없다 — 사실을 적고 옮길 뿐이다.

**Tech Stack:** ROS 2 Humble(rclpy, vica_interfaces msg, colcon), Python 3.10, pytest, PyYAML, pydantic(음성).

**Spec:** `vica-voice-llm/docs/superpowers/specs/2026-09-21-single-mouth-ledger-journal-design.md` 3절(개정 3). 이 계획은 파일 위치 세 곳을 스펙보다 구체화한다 — `ledger.json`·`map.yaml` 은 **목적지 폴더**(`~/vica_data/destinations/<map_id>/`, `home.yaml` 과 같은 자리), 건물 디렉터리는 `~/vica_data/destinations/directory.yaml` 하나. Task 13 에서 스펙에 반영한다.

## Global Constraints

- 미션·Safety 우회 금지. LLM 은 제안만(루트 CLAUDE.md). 이 계획은 판단 로직을 추가하지 않는다.
- 어떤 로그·파일 오류도 노드를 죽이지 않는다(대장 파일 쓰기 실패 = 경고 로그).
- ROS 저장소 브랜치: `vica_ros2_ws` `feat/robot-ledger`(현재 체크아웃 `feat/arrival-reconfirm` 에서 분기). 음성: `vica-voice-llm` `feat/realtime-intent`(현재).
- ROS 시험은 ROS 를 source 하지 않고 `cd vica_ros2_ws/src/vica_mission_manager && PYTHONPATH=. python3 -m pytest -q test/`. `test_progress_narration` 3건 실패는 기존 표류(무시).
- 음성 시험은 `cd vica-voice-llm && .venv/bin/python -m pytest -q`(현재 551 통과). 노드 임포트 확인은 ROS 를 source 한 뒤 `.venv/bin/python -c "import src.ros_node"`.
- 메시지 변경 뒤에는 `colcon build --packages-select vica_interfaces vica_mission_manager` 와 두 저장소 노드 재기동이 필요하다(음성 노드는 ROS install 의 `vica_interfaces` 를 임포트한다).
- 커밋 메시지는 한글, 끝에 `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- 배터리는 지금 측정기가 없어 항상 -1(모름)이다. 칸만 만든다.

---

### Task 1: RobotState 메시지 칸 추가

**Files:**
- Modify: `vica_ros2_ws/src/vica_interfaces/msg/RobotState.msg`

**Interfaces:**
- Produces: `RobotState` 에 필드 `dialog_state`(string), `place_here`(string), `place_here_dist_m`(float32), `active_destination`(string), `last_destination`(string), `last_arrived_age_sec`(int32), `aborted_destination`(string), `wait_minutes`(int32), `wait_left_sec`(int32), `battery_pct`(int32). 기존 4칸 유지.

- [ ] **Step 1: 브랜치 만들기**

```bash
cd /home/ji_w/VICA-smarthandle/vica_ros2_ws
git checkout -b feat/robot-ledger    # feat/arrival-reconfirm 에서 분기, 파일 변화 없음
```

- [ ] **Step 2: 메시지 파일 끝에 칸 추가**

`vica_ros2_ws/src/vica_interfaces/msg/RobotState.msg` 의 `bool is_paused` 뒤에 붙인다:

```
# ---- 로봇 대장 (2026-09-21 P1, 스펙 3절). 미션이 적고 LLM 노드가 읽는 사실. 없으면 ""/-1 ----
string dialog_state           # idle / awaiting_user / confirming / seeking / navigating / paused / asking_next / asking_wait_time / waiting / returning / estopped …
string place_here             # "407호 앞" / "407호와 화장실 사이" / "" = 위치 미확인
float32 place_here_dist_m     # 가장 가까운 등록 장소까지 m. 모르면 -1
string active_destination     # 가는 중인 곳 name. 없으면 ""
string last_destination       # 직전에 도착한 곳 name. 없으면 ""
int32 last_arrived_age_sec    # 그 도착 뒤 지난 초. 없으면 -1
string aborted_destination    # 확인까지 갔다가 안 간 곳 name. 없으면 ""
int32 wait_minutes            # 대기 요청 분. 없으면 -1
int32 wait_left_sec           # 대기 남은 초. 대기 중이 아니면 -1
int32 battery_pct             # 배터리 %. 모르면 -1
```

- [ ] **Step 3: 인터페이스 빌드와 확인**

```bash
cd /home/ji_w/VICA-smarthandle/vica_ros2_ws
source /opt/ros/humble/setup.bash && source install/setup.bash
colcon build --packages-select vica_interfaces 2>&1 | tail -3
source install/setup.bash
ros2 interface show vica_interfaces/msg/RobotState | grep -c "dialog_state\|battery_pct"
```
Expected: 빌드 `Summary: 1 package finished`, grep 결과 `2`.

- [ ] **Step 4: Commit**

```bash
git add src/vica_interfaces/msg/RobotState.msg
git commit -m "feat(interfaces): RobotState 에 로봇 대장 칸 10개 — 대화 단계·지금 있는 곳·가는 중·직전 도착·하려다 만 곳·대기·배터리

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: map_meta — 지도 한 장의 건물·층

**Files:**
- Create: `vica_ros2_ws/src/vica_mission_manager/vica_mission_manager/map_meta.py`
- Test: `vica_ros2_ws/src/vica_mission_manager/test/test_map_meta.py`

**Interfaces:**
- Produces: `MapMeta(building: str, floor: int)`(frozen dataclass, 기본 `""`, `-1`), `load_map_meta(destinations_path: str) -> MapMeta` — `destinations.yaml` 과 같은 폴더의 `map.yaml`(`building`, `floor`)을 읽는다. 없거나 깨지면 기본값.

- [ ] **Step 1: 실패하는 시험 쓰기**

`test/test_map_meta.py`:
```python
"""지도 한 장 = 한 층. 목적지 폴더의 map.yaml 에서 건물·층을 읽는다."""
from vica_mission_manager.map_meta import MapMeta, load_map_meta


def test_reads_building_and_floor(tmp_path):
    (tmp_path / "map.yaml").write_text("building: 로봇관\nfloor: 4\n", encoding="utf-8")
    meta = load_map_meta(str(tmp_path / "destinations.yaml"))
    assert meta == MapMeta(building="로봇관", floor=4)


def test_missing_file_is_unknown(tmp_path):
    assert load_map_meta(str(tmp_path / "destinations.yaml")) == MapMeta(building="", floor=-1)


def test_bad_values_fall_back(tmp_path):
    (tmp_path / "map.yaml").write_text("building: 5\nfloor: 사층\n", encoding="utf-8")
    meta = load_map_meta(str(tmp_path / "destinations.yaml"))
    assert meta.building == "5" and meta.floor == -1


def test_non_mapping_yaml_is_unknown(tmp_path):
    (tmp_path / "map.yaml").write_text("- 1\n- 2\n", encoding="utf-8")
    assert load_map_meta(str(tmp_path / "destinations.yaml")) == MapMeta()
```

- [ ] **Step 2: 실패 확인**

```bash
cd /home/ji_w/VICA-smarthandle/vica_ros2_ws/src/vica_mission_manager
PYTHONPATH=. python3 -m pytest -q test/test_map_meta.py
```
Expected: `ModuleNotFoundError: vica_mission_manager.map_meta`.

- [ ] **Step 3: 구현**

`vica_mission_manager/map_meta.py`:
```python
"""지도 한 장 = 한 층 (스펙 3.1). 목적지 폴더의 `map.yaml` 에서 건물·층을 읽는다.

    ~/vica_data/destinations/<map_id>/destinations.yaml   목적지 카탈로그 (기존)
    ~/vica_data/destinations/<map_id>/home.yaml           홈 위치 (기존)
    ~/vica_data/destinations/<map_id>/map.yaml            건물·층 (이 모듈)   예) building: 로봇관 / floor: 4

없거나 깨지면 "모름"(빈 문자열·-1)이다 — 층을 몰라도 미션은 돌아야 한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class MapMeta:
    building: str = ""
    floor: int = -1  # -1 = 모름


def load_map_meta(destinations_path: str) -> MapMeta:
    path = Path(destinations_path).expanduser().parent / "map.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return MapMeta()
    if not isinstance(data, dict):
        return MapMeta()
    building = str(data.get("building") or "").strip()
    try:
        floor = int(data.get("floor"))
    except (TypeError, ValueError):
        floor = -1
    return MapMeta(building=building, floor=floor)
```

- [ ] **Step 4: 통과 확인**

```bash
PYTHONPATH=. python3 -m pytest -q test/test_map_meta.py
```
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add vica_mission_manager/map_meta.py test/test_map_meta.py
git commit -m "feat(mission): map.yaml 에서 건물·층을 읽는 map_meta — 지도 한 장 = 한 층

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: ledger — 대장 자료형, 좌표→"OO 앞/사이", JSON 보존, 메시지 칸 파생

**Files:**
- Create: `vica_ros2_ws/src/vica_mission_manager/vica_mission_manager/ledger.py`
- Test: `vica_ros2_ws/src/vica_mission_manager/test/test_ledger.py`

**Interfaces:**
- Consumes: `mission_logic.Destination`, `mission_logic.Pose2D`.
- Produces:
  - `Ledger` dataclass: `active_destination: str = ""`, `last_destination: str = ""`, `last_arrived_at: Optional[float] = None`(epoch), `aborted_destination: str = ""`.
  - `Ledger.to_json() -> str`, `Ledger.from_json(text: str) -> Ledger`(복원은 `last_destination`·`last_arrived_at`·`aborted_destination` 만; `active_destination` 은 항상 빈 값).
  - `place_here(pose: Optional[Pose2D], destinations: Dict[str, Destination], cov_xy: float = 0.0, near_m: float = NEAR_M) -> tuple[str, float]`.
  - `LedgerStore(destinations_path: str)` with `.path: Path`, `.read() -> Ledger`, `.write(ledger) -> bool`.
  - `state_fields(ledger, dialog_state: str, pose, cov_xy, destinations, now_epoch: float, wait_minutes: int, wait_left_sec: int, battery_pct: int = -1) -> dict` — 메시지 칸 이름을 키로 한 dict.

- [ ] **Step 1: 실패하는 시험 쓰기**

`test/test_ledger.py`:
```python
"""로봇 대장: 좌표→장소 라벨, JSON 보존, 메시지 칸 파생. 판단 없음."""
import json

from vica_mission_manager.ledger import Ledger, LedgerStore, NEAR_M, place_here, state_fields
from vica_mission_manager.mission_logic import Destination, Pose2D


def _dest(id_, name, x, y):
    return Destination(id=id_, name=name, pose=Pose2D(x=x, y=y, yaw_deg=0.0), calibrated=True)


DESTS = {
    "a": _dest("a", "407호", 0.0, 0.0),
    "b": _dest("b", "화장실", 10.0, 0.0),
    "__home__": _dest("__home__", "홈", 50.0, 50.0),
}


class TestPlaceHere:
    def test_within_near_is_front_of(self):
        label, dist = place_here(Pose2D(1.0, 0.0, 0.0), DESTS)
        assert label == "407호 앞" and abs(dist - 1.0) < 1e-6

    def test_far_is_between_two_nearest(self):
        label, dist = place_here(Pose2D(5.0, 0.0, 0.0), DESTS)
        assert label == "407호와 화장실 사이" and abs(dist - 5.0) < 1e-6

    def test_josa_gwa_after_consonant(self):
        dests = {"a": _dest("a", "화장실", 0.0, 0.0), "b": _dest("b", "407호", 10.0, 0.0)}
        label, _ = place_here(Pose2D(4.0, 0.0, 0.0), dests)
        assert label == "화장실과 407호 사이"

    def test_home_is_ignored_for_label(self):
        label, _ = place_here(Pose2D(49.0, 50.0, 0.0), DESTS)
        assert "홈" not in label

    def test_no_pose_or_big_covariance_is_unknown(self):
        assert place_here(None, DESTS) == ("", -1.0)
        assert place_here(Pose2D(1.0, 0.0, 0.0), DESTS, cov_xy=5.0) == ("", -1.0)

    def test_single_destination_far_is_near(self):
        label, _ = place_here(Pose2D(20.0, 0.0, 0.0), {"a": DESTS["a"]})
        assert label == "407호 근처"


class TestStore:
    def test_roundtrip_restores_only_past_facts(self, tmp_path):
        store = LedgerStore(str(tmp_path / "destinations.yaml"))
        led = Ledger(active_destination="화장실", last_destination="407호", last_arrived_at=1000.0,
                     aborted_destination="세미나실")
        assert store.write(led) is True
        back = store.read()
        assert back.last_destination == "407호" and back.last_arrived_at == 1000.0
        assert back.aborted_destination == "세미나실"
        assert back.active_destination == ""          # 가는 중은 복원하지 않는다
        assert store.path.name == "ledger.json"

    def test_missing_or_broken_file_is_empty(self, tmp_path):
        store = LedgerStore(str(tmp_path / "destinations.yaml"))
        assert store.read() == Ledger()
        store.path.write_text("{not json", encoding="utf-8")
        assert store.read() == Ledger()

    def test_write_failure_returns_false(self, tmp_path):
        store = LedgerStore(str(tmp_path / "nope" / "destinations.yaml"))
        store.path = tmp_path            # 디렉터리에 쓰기 → 실패
        assert store.write(Ledger()) is False


class TestStateFields:
    def test_fields_are_derived_not_judged(self):
        led = Ledger(active_destination="", last_destination="407호", last_arrived_at=1000.0,
                     aborted_destination="화장실")
        f = state_fields(led, "waiting", Pose2D(1.0, 0.0, 0.0), 0.0, DESTS, now_epoch=1660.0,
                         wait_minutes=10, wait_left_sec=340)
        assert f == {
            "dialog_state": "waiting", "place_here": "407호 앞", "place_here_dist_m": 1.0,
            "active_destination": "", "last_destination": "407호", "last_arrived_age_sec": 660,
            "aborted_destination": "화장실", "wait_minutes": 10, "wait_left_sec": 340, "battery_pct": -1,
        }

    def test_unknowns_are_minus_one_or_empty(self):
        f = state_fields(Ledger(), "idle", None, 0.0, {}, now_epoch=5.0, wait_minutes=-1, wait_left_sec=-1)
        assert f["place_here"] == "" and f["place_here_dist_m"] == -1.0
        assert f["last_arrived_age_sec"] == -1 and f["battery_pct"] == -1
```

- [ ] **Step 2: 실패 확인**

```bash
PYTHONPATH=. python3 -m pytest -q test/test_ledger.py
```
Expected: `ModuleNotFoundError: vica_mission_manager.ledger`.

- [ ] **Step 3: 구현**

`vica_mission_manager/ledger.py`:
```python
"""로봇 대장(스펙 3절) — 미션이 적고 /vica/robot_state 로 방송하는 **사실**. 판단은 없다.

- Ledger: 직전에 간 곳·도착 시각·하려다 만 곳·가는 중인 곳. 파일(ledger.json)에 남겨
  재부팅 뒤 "직전에 간 곳·하려다 만 곳"만 복원한다(가는 중은 복원하지 않는다).
- place_here: AMCL 좌표를 등록 목적지와 견줘 "407호 앞"(NEAR_M 안) / "407호와 화장실 사이"
  (그 밖, 가장 가까운 두 곳) / "407호 근처"(등록 장소가 하나) / ""(좌표 없음·공분산 큼).
- state_fields: 위 사실을 RobotState 메시지 칸 이름 그대로 dict 로 편다. 노드는 복사만 한다.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from .mission_logic import Destination, Pose2D

NEAR_M = 3.0            # 이 안이면 "OO 앞"
COV_UNKNOWN = 2.0       # x·y 분산 합이 이보다 크면 위치 미확인(초기 위치 전)
HOME_ID = "__home__"    # 홈은 장소 라벨에서 뺀다 — "홈 앞"은 사용자에게 뜻이 없다


@dataclass
class Ledger:
    active_destination: str = ""
    last_destination: str = ""
    last_arrived_at: Optional[float] = None   # epoch 초
    aborted_destination: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Ledger":
        try:
            data = json.loads(text)
        except (TypeError, ValueError):
            return cls()
        if not isinstance(data, dict):
            return cls()
        arrived = data.get("last_arrived_at")
        return cls(
            active_destination="",
            last_destination=str(data.get("last_destination") or ""),
            last_arrived_at=float(arrived) if isinstance(arrived, (int, float)) else None,
            aborted_destination=str(data.get("aborted_destination") or ""),
        )


def _josa_wa(name: str) -> str:
    """받침 있으면 '과', 없으면 '와'. 한글이 아니면 '와'."""
    if not name:
        return "와"
    ch = name[-1]
    if "가" <= ch <= "힣":
        return "과" if (ord(ch) - 0xAC00) % 28 else "와"
    return "와"


def place_here(pose: Optional[Pose2D], destinations: Dict[str, Destination],
               cov_xy: float = 0.0, near_m: float = NEAR_M) -> Tuple[str, float]:
    """(라벨, 가장 가까운 등록 장소까지 m). 모르면 ("", -1.0)."""
    if pose is None or cov_xy > COV_UNKNOWN:
        return "", -1.0
    ranked = sorted(
        ((math.hypot(d.pose.x - pose.x, d.pose.y - pose.y), d)
         for d in destinations.values() if d.id != HOME_ID),
        key=lambda t: t[0],
    )
    if not ranked:
        return "", -1.0
    dist, nearest = ranked[0]
    if dist <= near_m:
        return f"{nearest.name} 앞", dist
    if len(ranked) >= 2:
        return f"{nearest.name}{_josa_wa(nearest.name)} {ranked[1][1].name} 사이", dist
    return f"{nearest.name} 근처", dist


class LedgerStore:
    """`<목적지 폴더>/ledger.json`. home.yaml 과 같은 자리(스펙과 다른 점: state/ 폴더 대신)."""

    def __init__(self, destinations_path: str) -> None:
        self.path = Path(destinations_path).expanduser().parent / "ledger.json"

    def read(self) -> Ledger:
        try:
            return Ledger.from_json(self.path.read_text(encoding="utf-8"))
        except OSError:
            return Ledger()

    def write(self, ledger: Ledger) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(ledger.to_json(), encoding="utf-8")
            tmp.replace(self.path)
            return True
        except OSError:
            return False


def state_fields(ledger: Ledger, dialog_state: str, pose: Optional[Pose2D], cov_xy: float,
                 destinations: Dict[str, Destination], now_epoch: float,
                 wait_minutes: int, wait_left_sec: int, battery_pct: int = -1) -> dict:
    label, dist = place_here(pose, destinations, cov_xy)
    age = int(now_epoch - ledger.last_arrived_at) if ledger.last_arrived_at is not None else -1
    return {
        "dialog_state": dialog_state,
        "place_here": label,
        "place_here_dist_m": float(dist),
        "active_destination": ledger.active_destination,
        "last_destination": ledger.last_destination,
        "last_arrived_age_sec": age,
        "aborted_destination": ledger.aborted_destination,
        "wait_minutes": int(wait_minutes),
        "wait_left_sec": int(wait_left_sec),
        "battery_pct": int(battery_pct),
    }
```

- [ ] **Step 4: 통과 확인**

```bash
PYTHONPATH=. python3 -m pytest -q test/test_ledger.py
```
Expected: `11 passed`.

- [ ] **Step 5: Commit**

```bash
git add vica_mission_manager/ledger.py test/test_ledger.py
git commit -m "feat(mission): 로봇 대장 자료형·좌표→장소 라벨·ledger.json 보존·메시지 칸 파생 (순수 모듈)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: 미션 로직 접근자 — 대기 요청 분·남은 초

**Files:**
- Modify: `vica_ros2_ws/src/vica_mission_manager/vica_mission_manager/mission_logic.py:1032`(초기화), `:2017-2023`(`_enter_waiting`), `:1185` 근처(접근자 추가), `_reset_arrival_dialog`(`_wait_until = None` 자리)
- Test: `vica_ros2_ws/src/vica_mission_manager/test/test_arrival_dialog.py`(끝에 추가)

**Interfaces:**
- Produces: `MissionLogic.wait_minutes_requested() -> int`(대기 중 아니면 -1), `MissionLogic.wait_left_sec(now: float) -> int`(WAITING 아니면 -1, 남은 초 ≥ 0).

- [ ] **Step 1: 실패하는 시험 쓰기**

`test/test_arrival_dialog.py` 끝에:
```python


class TestLedgerAccessors:
    """대장(P1)이 읽는 대기 접근자 — 판단이 아니라 값 노출."""

    def test_wait_minutes_and_left(self):
        logic = arrive("")                                    # "여기서 대기할까요?"
        logic.on_arrival_answer(_intent("wait", wait_minutes=10), 3.0)
        assert logic.state == State.WAITING
        assert logic.wait_minutes_requested() == 10
        assert logic.wait_left_sec(63.0) == 540
        assert logic.wait_left_sec(3.0 + 601.0) == 0

    def test_not_waiting_is_minus_one(self):
        logic = MissionLogic()
        assert logic.wait_minutes_requested() == -1 and logic.wait_left_sec(0.0) == -1
        logic = arrive("")
        logic.on_arrival_answer(_intent("wait", wait_minutes=10), 3.0)
        logic.on_wake(10.0)                                   # 대기 접음
        assert logic.wait_minutes_requested() == -1 and logic.wait_left_sec(10.0) == -1
```

- [ ] **Step 2: 실패 확인**

```bash
PYTHONPATH=. python3 -m pytest -q test/test_arrival_dialog.py -k Ledger
```
Expected: `AttributeError: 'MissionLogic' object has no attribute 'wait_minutes_requested'`.

- [ ] **Step 3: 구현**

`mission_logic.py` — 초기화(1032행 `self._wait_until` 바로 아래):
```python
        self._wait_minutes_requested = -1    # 대장(P1): 대기 요청 분. WAITING 밖에서는 -1
```
`_enter_waiting`(2020행 `self._wait_until = now + minutes * 60.0` 바로 아래):
```python
        self._wait_minutes_requested = int(minutes)
```
`_reset_arrival_dialog`(`self._wait_until = None` 줄 바로 아래)와 `on_wake` 의 WAITING 분기(`self._wait_until = None` 바로 아래)에 각각:
```python
        self._wait_minutes_requested = -1
```
접근자(`confirming_dest_id` 프로퍼티 바로 아래, 1190행 근처):
```python
    def wait_minutes_requested(self) -> int:
        """대장(P1): 대기 요청 분. WAITING 이 아니면 -1."""
        return self._wait_minutes_requested if self.state == State.WAITING else -1

    def wait_left_sec(self, now: float) -> int:
        """대장(P1): 대기 남은 초(0 이상). WAITING 이 아니면 -1."""
        if self.state != State.WAITING or self._wait_until is None:
            return -1
        return max(0, int(self._wait_until - now))
```

- [ ] **Step 4: 통과 확인**

```bash
PYTHONPATH=. python3 -m pytest -q test/test_arrival_dialog.py
```
Expected: 기존 57 + 2 = `59 passed`.

- [ ] **Step 5: Commit**

```bash
git add vica_mission_manager/mission_logic.py test/test_arrival_dialog.py
git commit -m "feat(mission): 대장용 접근자 — 대기 요청 분·남은 초

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: 노드 통합 — 대장을 채우고 방송하고 보존한다

**Files:**
- Modify: `vica_ros2_ws/src/vica_mission_manager/vica_mission_manager/mission_manager_node.py` — import 블록, `__init__`(196행 `self.destinations = load_destinations(...)` 뒤), `_on_amcl_pose`(1053행), `_tick`(1336행), `_publish_robot_state`(1350행), `_publish_goal_event`(1625행)

**Interfaces:**
- Consumes: Task 2 `load_map_meta`, Task 3 `Ledger`/`LedgerStore`/`state_fields`, Task 4 접근자.
- Produces: `/vica/robot_state` 에 Task 1 의 열 칸이 채워져 1 Hz 로 나간다. `<목적지 폴더>/ledger.json` 이 도착·취소·거절 때 갱신된다.

- [ ] **Step 1: import 와 초기화**

import 블록(`from .home_storage import ...` 근처)에:
```python
from .ledger import Ledger, LedgerStore, state_fields
from .map_meta import load_map_meta
```
`__init__` 의 `self.destinations = load_destinations(self._destinations_path)` 바로 뒤:
```python
        # ---- 로봇 대장 (P1, 스펙 3절) — 사실만 적고 방송한다 ---------------------
        self._map_meta = load_map_meta(self._destinations_path)
        self._ledger_store = LedgerStore(self._destinations_path)
        self._ledger: Ledger = self._ledger_store.read()
        self._ledger_prev_confirming: Optional[str] = None   # 확인 대기 → 안 감 전이 감지용
        self._pose_cov_xy = 0.0                               # AMCL x·y 분산 합(초기 위치 전 판정)
        self.get_logger().info(
            f"대장: 건물 {self._map_meta.building or '?'} {self._map_meta.floor}층, "
            f"직전 도착 {self._ledger.last_destination or '-'}, 파일 {self._ledger_store.path}")
```
(`Optional` 은 이미 typing 에서 import 돼 있다. `time` 모듈이 import 돼 있지 않으면 `import time` 을 더한다.)

- [ ] **Step 2: AMCL 공분산 기억**

`_on_amcl_pose` 의 `self._robot_pose = Pose2D(...)` 뒤에:
```python
        cov = msg.pose.covariance
        self._pose_cov_xy = float(cov[0] + cov[7]) if len(cov) >= 8 else 0.0
```

- [ ] **Step 3: goal 사건에서 대장 갱신**

`_publish_goal_event(self, event, destination, reason="")` 본문 맨 앞(`msg = String()` 앞)에:
```python
        # 대장(P1): 사건 → 사실. 홈 복귀는 목적지가 아니다.
        name = destination.name if (destination and destination.id != "__home__") else ""
        if event in ("goal_sent", "goal_accepted") and name:
            self._ledger.active_destination = name
            self._ledger.aborted_destination = ""
        elif event == "goal_succeeded" and name:
            self._ledger.last_destination = name
            self._ledger.last_arrived_at = time.time()
            self._ledger.active_destination = ""
            self._save_ledger()
        elif event in ("goal_failed", "goal_rejected", "goal_canceled"):
            if name:
                self._ledger.aborted_destination = name
            self._ledger.active_destination = ""
            self._save_ledger()
        elif event in ("return_home_sent", "return_home_succeeded", "return_home_failed",
                       "return_home_canceled", "state_idle"):
            self._ledger.active_destination = ""
```
그리고 클래스 어딘가(`_publish_goal_event` 바로 아래)에:
```python
    def _save_ledger(self) -> None:
        """대장 파일 갱신. 실패는 경고만 — 파일 오류로 노드가 죽지 않는다."""
        if not self._ledger_store.write(self._ledger):
            self.get_logger().warning(f"대장 파일 쓰기 실패: {self._ledger_store.path}")
```

- [ ] **Step 4: tick 에서 "확인까지 갔다가 안 감" 전이 감지**

`_tick` 의 `actions = self.logic.on_tick(self._now(), status, distance)` 바로 뒤:
```python
        # 대장(P1): 확인 대기(CONFIRMING)였다가 출발 없이 접혔으면(거절·시간초과·호출로 접음)
        # 그 목적지가 "하려다 만 곳"이다. 상태 전이만 보고 적는다 — 판단이 아니다.
        confirming = self.logic.confirming_dest_id
        if (self._ledger_prev_confirming and not confirming
                and self.logic.state not in (State.NAVIGATING, State.CONFIRMING)
                and self.logic.active_destination is None):
            prev = self.destinations.get(self._ledger_prev_confirming)
            if prev is not None:
                self._ledger.aborted_destination = prev.name
                self._save_ledger()
        self._ledger_prev_confirming = confirming
```

- [ ] **Step 5: 방송 채우기**

`_publish_robot_state` 를 이렇게 바꾼다(기존 주석은 유지):
```python
    def _publish_robot_state(self) -> None:
        msg = RobotState()
        floor_param = int(self.get_parameter("current_floor").value)
        building_param = str(self.get_parameter("current_building").value)
        # 층·건물: launch 인자가 있으면 우선, 없으면 목적지 폴더의 map.yaml (지도 = 한 층)
        msg.current_floor = floor_param if floor_param >= 0 else self._map_meta.floor
        msg.current_building = building_param or self._map_meta.building
        msg.is_moving = self.logic.state in (State.NAVIGATING, State.SEEKING)
        msg.is_paused = self.logic.state == State.PAUSED
        fields = state_fields(
            self._ledger, self.logic.state.value, self._robot_pose, self._pose_cov_xy,
            self.destinations, now_epoch=time.time(),
            wait_minutes=self.logic.wait_minutes_requested(),
            wait_left_sec=self.logic.wait_left_sec(self._now()),
        )
        for key, value in fields.items():
            setattr(msg, key, value)
        self.pub_state.publish(msg)
```
(`msg.is_moving`·`is_paused` 위의 긴 주석 두 덩이는 그대로 둔다.)

- [ ] **Step 6: 시험·임포트 확인**

```bash
cd /home/ji_w/VICA-smarthandle/vica_ros2_ws/src/vica_mission_manager
PYTHONPATH=. python3 -m pytest -q test/ 2>&1 | tail -2
cd /home/ji_w/VICA-smarthandle/vica_ros2_ws && source /opt/ros/humble/setup.bash && source install/setup.bash
python3 -c "import vica_mission_manager.mission_manager_node as m; print('node import ok')"
```
Expected: 시험은 `3 failed`(기존 progress_narration 표류)·나머지 passed, `node import ok`.

- [ ] **Step 7: Commit**

```bash
git add vica_mission_manager/mission_manager_node.py
git commit -m "feat(mission): 로봇 대장을 채워 방송·보존 — 층은 map.yaml, 위치는 AMCL 좌표, 도착·취소·확인 접힘은 사건과 전이로

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: 로봇 데이터 — map.yaml 과 빌드·재기동

**Files:**
- Create (로봇 데이터, git 밖): `~/vica_data/destinations/vica_map_0903_d/map.yaml`
- Modify: `vica_ros2_ws/docs/vica_robot_bringup_manual.md`(경로 규약 표에 한 줄)

- [ ] **Step 1: 지도 메타 파일**

```bash
cat > ~/vica_data/destinations/vica_map_0903_d/map.yaml <<'EOF'
# 지도 한 장 = 한 층. 미션이 /vica/robot_state 의 층·건물로 방송한다 (P1, 2026-09-21).
building: 로봇관
floor: 4
EOF
```

- [ ] **Step 2: 문서 한 줄**

`vica_ros2_ws/docs/vica_robot_bringup_manual.md` 에서 `home.yaml` 경로 규약을 설명하는 표(또는 줄) 아래에 추가:
```
| `<storage_root>/<map_id>/map.yaml` | 건물·층 한 줄(`building`, `floor`). 미션이 읽어 `/vica/robot_state` 로 방송. 없으면 층 -1 |
| `<storage_root>/<map_id>/ledger.json` | 로봇 대장(직전 도착·하려다 만 곳). 미션이 쓴다. 손으로 고치지 않는다 |
```

- [ ] **Step 3: 빌드**

```bash
cd /home/ji_w/VICA-smarthandle/vica_ros2_ws
source /opt/ros/humble/setup.bash && source install/setup.bash
colcon build --packages-select vica_interfaces vica_mission_manager 2>&1 | tail -3
```
Expected: `Summary: 2 packages finished`.

- [ ] **Step 4: 방송 확인(스택이 떠 있을 때)**

⑩ mission 칸을 Ctrl-C 후 다시 띄운 뒤:
```bash
source /opt/ros/humble/setup.bash && source install/setup.bash && export ROS_DOMAIN_ID=7
timeout 5 ros2 topic echo /vica/robot_state --once
```
Expected: `current_building: 로봇관`, `current_floor: 4`, `dialog_state: idle`, `place_here: ''`(초기 위치 전) 또는 `'OO 앞'`.

- [ ] **Step 5: Commit**

```bash
git add docs/vica_robot_bringup_manual.md
git commit -m "docs(mission): map.yaml·ledger.json 경로 규약

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: 음성 — pydantic RobotState 칸과 변환

**Files:**
- Modify: `vica-voice-llm/src/schema.py:94-106`, `vica-voice-llm/src/ros_convert.py:54-62`
- Test: `vica-voice-llm/tests/test_ledger_view.py`(새 파일, Task 8 과 공유)

**Interfaces:**
- Produces: `RobotState` 에 `dialog_state: str = ""`, `place_here: str = ""`, `place_here_dist_m: float = -1.0`, `active_destination: str = ""`, `last_destination: str = ""`, `last_arrived_age_sec: int = -1`, `aborted_destination: str = ""`, `wait_minutes: int = -1`, `wait_left_sec: int = -1`, `battery_pct: int = -1`. `msg_to_robot_state` 는 옛 메시지(칸 없음)도 `getattr` 기본값으로 받는다.

- [ ] **Step 1: 실패하는 시험 쓰기**

`tests/test_ledger_view.py`:
```python
"""대장 방송(RobotState) → 지시문 [지금 상황] 블록. 옛 미션(칸 없음)이면 빈 문자열."""
from types import SimpleNamespace

from src.ros_convert import msg_to_robot_state
from src.schema import RobotState


def _msg(**kw):
    base = dict(current_floor=4, current_building="로봇관", is_moving=False, is_paused=False)
    base.update(kw)
    return SimpleNamespace(**base)


class TestConvert:
    def test_new_fields_are_converted(self):
        st = msg_to_robot_state(_msg(dialog_state="waiting", place_here="407호 앞", place_here_dist_m=1.5,
                                     active_destination="", last_destination="407호", last_arrived_age_sec=660,
                                     aborted_destination="화장실", wait_minutes=10, wait_left_sec=340, battery_pct=-1))
        assert st.dialog_state == "waiting" and st.place_here == "407호 앞" and st.last_destination == "407호"
        assert st.wait_minutes == 10 and st.wait_left_sec == 340 and st.battery_pct == -1
        assert st.current_floor == 4

    def test_old_message_without_fields_still_converts(self):
        st = msg_to_robot_state(_msg())
        assert st.dialog_state == "" and st.place_here == "" and st.last_arrived_age_sec == -1
        assert st == RobotState(current_floor=4, current_building="로봇관")
```

- [ ] **Step 2: 실패 확인**

```bash
cd /home/ji_w/VICA-smarthandle/vica-voice-llm && .venv/bin/python -m pytest -q tests/test_ledger_view.py
```
Expected: `AttributeError` 또는 pydantic 검증 오류(`dialog_state` 없음).

- [ ] **Step 3: 구현**

`src/schema.py` 의 `RobotState` 에 `is_paused` 아래 추가:
```python
    # ---- 로봇 대장 (P1, 스펙 3절). 미션이 적고 여기선 읽기만. 없으면 ""/-1 ----
    dialog_state: str = ""
    place_here: str = ""
    place_here_dist_m: float = -1.0
    active_destination: str = ""
    last_destination: str = ""
    last_arrived_age_sec: int = -1
    aborted_destination: str = ""
    wait_minutes: int = -1
    wait_left_sec: int = -1
    battery_pct: int = -1
```
`src/ros_convert.py` 의 `msg_to_robot_state` 를:
```python
def msg_to_robot_state(msg: RobotStateMsg) -> RobotState:
    """ROS2 RobotState 메시지 -> pydantic RobotState. (-1 층은 '알 수 없음' = None)

    대장 칸(P1)은 옛 미션 메시지에 없을 수 있어 getattr 기본값으로 받는다 — 그러면
    ledger_view 가 빈 블록을 돌려주고 노드는 goal-event 상황판으로 폴백한다.
    """
    return RobotState(
        current_floor=None if msg.current_floor < 0 else msg.current_floor,
        current_building=msg.current_building,
        is_moving=msg.is_moving,
        is_paused=msg.is_paused,
        dialog_state=str(getattr(msg, "dialog_state", "") or ""),
        place_here=str(getattr(msg, "place_here", "") or ""),
        place_here_dist_m=float(getattr(msg, "place_here_dist_m", -1.0)),
        active_destination=str(getattr(msg, "active_destination", "") or ""),
        last_destination=str(getattr(msg, "last_destination", "") or ""),
        last_arrived_age_sec=int(getattr(msg, "last_arrived_age_sec", -1)),
        aborted_destination=str(getattr(msg, "aborted_destination", "") or ""),
        wait_minutes=int(getattr(msg, "wait_minutes", -1)),
        wait_left_sec=int(getattr(msg, "wait_left_sec", -1)),
        battery_pct=int(getattr(msg, "battery_pct", -1)),
    )
```

- [ ] **Step 4: 통과 확인**

```bash
.venv/bin/python -m pytest -q tests/test_ledger_view.py tests/test_robot_sim.py
```
Expected: 전부 passed.

- [ ] **Step 5: Commit**

```bash
git add src/schema.py src/ros_convert.py tests/test_ledger_view.py
git commit -m "feat(voice): RobotState 에 대장 칸 10개 — 옛 메시지 호환 변환

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: ledger_view — 방송을 [지금 상황] 블록으로

**Files:**
- Create: `vica-voice-llm/src/ledger_view.py`
- Test: `vica-voice-llm/tests/test_ledger_view.py`(추가)

**Interfaces:**
- Consumes: Task 7 `RobotState`.
- Produces: `render_ledger(state: RobotState, *, awaiting_answer: bool, now_text: str) -> str` — `state.dialog_state == ""` 이면 `""`(폴백 신호). 그 외 `"\n[지금 상황] …\n- …\n"` 블록. `DIALOG_KO: dict[str, str]`.

- [ ] **Step 1: 실패하는 시험 쓰기**

`tests/test_ledger_view.py` 끝에:
```python


from src.ledger_view import render_ledger  # noqa: E402


class TestRender:
    def test_full_board(self):
        st = RobotState(current_floor=4, current_building="로봇관", dialog_state="waiting", place_here="407호 앞",
                        place_here_dist_m=1.5, last_destination="407호", last_arrived_age_sec=660,
                        aborted_destination="화장실", wait_minutes=10, wait_left_sec=340, battery_pct=-1)
        text = render_ledger(st, awaiting_answer=False, now_text="15:40")
        assert text.startswith("\n[지금 상황]")
        for line in ("- 건물/층: 로봇관 4층", "- 지금 있는 곳: 407호 앞", "- 안내 중: 없음",
                     "- 직전에 간 곳: 407호 (11분 전 도착)", "- 하려다 만 곳: 화장실",
                     "- 대화 단계: 대기 중", "- 대기: 10분 요청, 5분 40초 남음",
                     "- 시각: 15:40", "- 배터리: 모름"):
            assert line in text, line

    def test_navigating_and_unknowns(self):
        st = RobotState(dialog_state="navigating", active_destination="화장실", is_moving=True)
        text = render_ledger(st, awaiting_answer=True, now_text="09:01")
        assert "- 건물/층: 모름" in text and "- 지금 있는 곳: 위치 미확인" in text
        assert "- 안내 중: 화장실로 이동 중" in text and "- 직전에 간 곳: 아직 없음" in text
        assert "- 대화 단계: 안내 중(주행)" in text and "- 대기: 없음" in text
        assert "답을 기다리는 중: 예" in text

    def test_old_mission_gives_empty(self):
        assert render_ledger(RobotState(current_floor=4), awaiting_answer=False, now_text="09:01") == ""

    def test_unknown_state_word_passes_through(self):
        text = render_ledger(RobotState(dialog_state="turning"), awaiting_answer=False, now_text="09:01")
        assert "- 대화 단계: 회전 중" in text
```

- [ ] **Step 2: 실패 확인**

```bash
.venv/bin/python -m pytest -q tests/test_ledger_view.py
```
Expected: `ModuleNotFoundError: src.ledger_view`.

- [ ] **Step 3: 구현**

`src/ledger_view.py`:
```python
"""대장 방송(/vica/robot_state) → 지시문 맨 뒤 `[지금 상황]` 블록 (스펙 3.2).

미션이 적은 사실을 옮겨 적을 뿐 판단하지 않는다. 옛 미션(대장 칸이 비어 옴)이면 빈
문자열을 돌려주고, 노드는 goal-event 추정 상황판(situation_board)으로 폴백한다.
"""
from __future__ import annotations

from .schema import RobotState

DIALOG_KO = {
    "idle": "대기(안내 없음)",
    "awaiting_user": "인사 답 대기",
    "confirming": "목적지 확인 대기",
    "seeking": "호출 방향으로 회전 중",
    "turning": "회전 중",
    "approaching": "사람에게 다가가는 중",
    "navigating": "안내 중(주행)",
    "paused": "일시정지",
    "arrived": "도착",
    "asking_next": "도착 질문 중",
    "asking_wait_time": "대기 시간 질문 중",
    "waiting": "대기 중",
    "returning": "제자리로 복귀 중",
    "estopped": "비상 정지",
    "failed": "이동 실패",
}

HEADER = ("\n[지금 상황] (미션이 확인한 사실 — 대화 이력이 비어 있어도 이것은 맞다. "
          "\"지금 어디 가?\"·\"아까 어디 갔었지?\"·\"몇 층이야?\"·\"몇 시야?\"는 이것으로 답한다)\n")


def _ago(sec: int) -> str:
    if sec < 60:
        return "방금"
    if sec < 3600:
        return f"{sec // 60}분 전"
    return f"{sec // 3600}시간 전"


def _left(sec: int) -> str:
    return f"{sec // 60}분 {sec % 60}초" if sec >= 60 else f"{sec}초"


def render_ledger(state: RobotState, *, awaiting_answer: bool, now_text: str) -> str:
    if not state.dialog_state:
        return ""
    lines = []
    if state.current_building or state.current_floor is not None:
        floor = f" {state.current_floor}층" if state.current_floor is not None else ""
        lines.append(f"- 건물/층: {state.current_building or '건물 모름'}{floor}")
    else:
        lines.append("- 건물/층: 모름")
    lines.append(f"- 지금 있는 곳: {state.place_here or '위치 미확인'}")
    if state.active_destination:
        lines.append(f"- 안내 중: {state.active_destination}로 이동 중")
    else:
        lines.append("- 안내 중: 없음")
    if state.last_destination:
        lines.append(f"- 직전에 간 곳: {state.last_destination} ({_ago(state.last_arrived_age_sec)} 도착)")
    else:
        lines.append("- 직전에 간 곳: 아직 없음")
    if state.aborted_destination:
        lines.append(f"- 하려다 만 곳: {state.aborted_destination}")
    lines.append(f"- 대화 단계: {DIALOG_KO.get(state.dialog_state, state.dialog_state)}")
    if state.wait_minutes >= 0:
        left = f", {_left(state.wait_left_sec)} 남음" if state.wait_left_sec >= 0 else ""
        lines.append(f"- 대기: {state.wait_minutes}분 요청{left}")
    else:
        lines.append("- 대기: 없음")
    lines.append(f"- 시각: {now_text}")
    lines.append(f"- 배터리: {state.battery_pct}%" if state.battery_pct >= 0 else "- 배터리: 모름")
    if awaiting_answer:
        lines.append("- 로봇이 방금 질문하고 답을 기다리는 중: 예 (지금 들리는 말은 그 답일 가능성이 높다)")
    return HEADER + "\n".join(lines) + "\n"
```

- [ ] **Step 4: 통과 확인**

```bash
.venv/bin/python -m pytest -q tests/test_ledger_view.py
```
Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/ledger_view.py tests/test_ledger_view.py
git commit -m "feat(voice): 대장 방송 → [지금 상황] 블록 렌더 (옛 미션이면 빈 블록)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: 노드 연결 — 대장 우선, 없으면 옛 상황판

**Files:**
- Modify: `vica-voice-llm/src/ros_node.py`(import, `_on_user_audio` 의 `situation=` 인자), `vica-voice-llm/src/langchain_intent_parser.py`(`build_audio_prompt` 의 state_block 중복 제거)
- Test: `vica-voice-llm/tests/test_parser_audio.py`(추가)

**Interfaces:**
- Consumes: Task 8 `render_ledger`, 기존 `SituationBoard.render(awaiting_answer=)`.
- Produces: 소리 모드 지시문 맨 뒤에 `[지금 상황]` 이 있으면 `[현재 로봇 상태]` 블록은 생략된다(중복 방지).

- [ ] **Step 1: 실패하는 시험 쓰기**

`tests/test_parser_audio.py` 의 `TestAudioPrompt` 에:
```python
    def test_state_block_is_skipped_when_situation_has_ledger(self):
        from src.schema import RobotState
        st = RobotState(current_floor=4, current_building="로봇관", dialog_state="idle")
        with_ledger = parser.build_audio_prompt([DEST], st, situation="\n[지금 상황]\n- 건물/층: 로봇관 4층\n")
        assert "[현재 로봇 상태]" not in with_ledger and "[지금 상황]" in with_ledger
        without = parser.build_audio_prompt([DEST], st, situation="")
        assert "[현재 로봇 상태]" in without
```

- [ ] **Step 2: 실패 확인**

```bash
.venv/bin/python -m pytest -q tests/test_parser_audio.py -k ledger
```
Expected: FAIL(`[현재 로봇 상태]` 가 들어 있음).

- [ ] **Step 3: 구현**

`src/langchain_intent_parser.py` `build_audio_prompt` 의 `state_block = _format_robot_state(robot_state)` 를:
```python
    # 대장 블록([지금 상황], ledger_view)이 있으면 층·이동 중이 거기 있으므로 옛 상태 블록은 뺀다.
    state_block = "" if "[지금 상황]" in situation else _format_robot_state(robot_state)
```
`src/ros_node.py` import 에 `from .ledger_view import render_ledger` 를 더하고, `_on_user_audio` 의
```python
                situation=self._board.render(awaiting_answer=time.time() < self._followup_until))
```
를
```python
                situation=self._situation_block())
```
로 바꾼 뒤 메서드를 추가한다(`_shadow_text` 위):
```python
    def _situation_block(self) -> str:
        """지시문 맨 뒤 [지금 상황]: 미션 대장(P1)이 오면 그것, 옛 미션이면 goal-event 상황판."""
        awaiting = time.time() < self._followup_until
        text = render_ledger(self._robot_state, awaiting_answer=awaiting, now_text=time.strftime("%H:%M"))
        return text or self._board.render(awaiting_answer=awaiting)
```

- [ ] **Step 4: 통과·임포트 확인**

```bash
.venv/bin/python -m pytest -q 2>&1 | tail -1
source /opt/ros/humble/setup.bash && source /home/ji_w/VICA-smarthandle/vica_ros2_ws/install/setup.bash && .venv/bin/python -c "import src.ros_node; print('ros_node import ok')"
```
Expected: 전부 passed, `ros_node import ok`.

- [ ] **Step 5: Commit**

```bash
git add src/ros_node.py src/langchain_intent_parser.py tests/test_parser_audio.py
git commit -m "feat(voice): 소리 모드 [지금 상황]은 미션 대장 우선, 옛 미션이면 goal-event 상황판

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: 건물 디렉터리 — 다른 층 장소를 안다

**Files:**
- Create: `vica-voice-llm/src/building_directory.py`
- Modify: `vica-voice-llm/src/langchain_intent_parser.py`(`build_audio_prompt` 에 `directory_block` 인자와 규칙), `vica-voice-llm/src/ros_node.py`(로드·전달), `vica-voice-llm/.env.example`
- Test: `vica-voice-llm/tests/test_building_directory.py`
- Create (로봇 데이터, git 밖): `~/vica_data/destinations/directory.yaml`

**Interfaces:**
- Produces: `DirectoryEntry(name: str, building: str, floor: int)`, `load_directory(path: str) -> list[DirectoryEntry]`(없으면 `[]`), `format_directory_block(entries, current_building: str, current_floor: Optional[int]) -> str`(현재 층 항목은 빼고 `"\n[다른 층 장소] …"` 블록, 비면 `""`), `DEFAULT_DIRECTORY_YAML = "~/vica_data/destinations/directory.yaml"`, env `VICA_DIRECTORY_YAML`.
- `build_audio_prompt(destinations, robot_state=None, situation="", directory_block="")`.

- [ ] **Step 1: 실패하는 시험 쓰기**

`tests/test_building_directory.py`:
```python
"""건물 디렉터리: 다른 층 장소를 알되 갈 수는 없다 → 엘리베이터 제안 (스펙 3.1)."""
from src.building_directory import DirectoryEntry, format_directory_block, load_directory


def test_load_and_skip_bad_rows(tmp_path):
    p = tmp_path / "directory.yaml"
    p.write_text("- name: 세미나실\n  building: 로봇관\n  floor: 3\n- name: 잘못\n  floor: x\n- 5\n", encoding="utf-8")
    assert load_directory(str(p)) == [DirectoryEntry(name="세미나실", building="로봇관", floor=3)]


def test_missing_file_is_empty(tmp_path):
    assert load_directory(str(tmp_path / "none.yaml")) == []


def test_block_lists_only_other_floors():
    entries = [DirectoryEntry("세미나실", "로봇관", 3), DirectoryEntry("407호", "로봇관", 4),
               DirectoryEntry("식당", "학생회관", 1)]
    block = format_directory_block(entries, "로봇관", 4)
    assert block.startswith("\n[다른 층 장소]")
    assert "세미나실: 로봇관 3층" in block and "식당: 학생회관 1층" in block and "407호" not in block


def test_block_empty_when_nothing_else():
    assert format_directory_block([DirectoryEntry("407호", "로봇관", 4)], "로봇관", 4) == ""
    assert format_directory_block([], "로봇관", 4) == ""
```
그리고 `tests/test_parser_audio.py` 의 `TestAudioPrompt` 에:
```python
    def test_directory_block_and_elevator_rule(self):
        text = parser.build_audio_prompt([DEST], directory_block="\n[다른 층 장소]\n- 세미나실: 로봇관 3층\n")
        assert "[다른 층 장소]" in text and "엘리베이터" in text
```

- [ ] **Step 2: 실패 확인**

```bash
.venv/bin/python -m pytest -q tests/test_building_directory.py tests/test_parser_audio.py -k "directory"
```
Expected: `ModuleNotFoundError: src.building_directory`.

- [ ] **Step 3: 구현**

`src/building_directory.py`:
```python
"""건물 디렉터리 (스펙 3.1): 이 지도(층) 밖의 장소를 이름·건물·층으로만 안다.

    ~/vica_data/destinations/directory.yaml
    - name: 세미나실
      building: 로봇관
      floor: 3

갈 수는 없다(지도 = 한 층). 모델은 "3층에 있다, 층 이동은 못 한다"고 말하고 등록 목적지
'엘리베이터'가 있으면 그리로 안내를 제안한다(지시문 규칙). 파일이 없으면 빈 목록 — 그러면
블록도 없고 모델은 지금처럼 "모른다"고 답한다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

DEFAULT_DIRECTORY_YAML = "~/vica_data/destinations/directory.yaml"


@dataclass(frozen=True)
class DirectoryEntry:
    name: str
    building: str
    floor: int


def directory_path() -> str:
    return os.environ.get("VICA_DIRECTORY_YAML", DEFAULT_DIRECTORY_YAML)


def load_directory(path: str) -> list[DirectoryEntry]:
    try:
        data = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []
    if not isinstance(data, list):
        return []
    entries: list[DirectoryEntry] = []
    for row in data:
        if not isinstance(row, dict) or not row.get("name"):
            continue
        try:
            floor = int(row.get("floor"))
        except (TypeError, ValueError):
            continue
        entries.append(DirectoryEntry(name=str(row["name"]).strip(),
                                      building=str(row.get("building") or "").strip(), floor=floor))
    return entries


def format_directory_block(entries: list[DirectoryEntry], current_building: str,
                           current_floor: Optional[int]) -> str:
    others = [e for e in entries
              if not (e.building == current_building and current_floor is not None and e.floor == current_floor)]
    if not others:
        return ""
    lines = [f"- {e.name}: {e.building} {e.floor}층".replace(":  ", ": ") for e in others]
    return "\n[다른 층 장소] (이 지도에 없어 직접 갈 수 없다)\n" + "\n".join(lines) + "\n"
```
`src/langchain_intent_parser.py` `build_audio_prompt` 시그니처를
```python
def build_audio_prompt(
    destinations: Sequence[DestinationData], robot_state: Optional[RobotState] = None,
    situation: str = "", directory_block: str = "",
) -> str:
```
로 바꾸고, 지시문의 `[목적지 목록] … {dest_block}` 바로 뒤에:
```
{directory_block}
[다른 층] 위 "다른 층 장소"에 있는 곳을 물으면 "OO는 N층에 있어요"라고 알려주고, 층 이동은 못 한다고
말한 뒤 목록에 "엘리베이터"가 있으면 그리로 안내를 제안한다(navigate, need_confirm=true). 없으면
"엘리베이터로 가시면 돼요" 한 문장.
```
(`{directory_block}` 은 f-string 안이므로 그대로 치환된다.)

`src/ros_node.py`: import `from .building_directory import directory_path, format_directory_block, load_directory`; `__init__` 의 `self._reload_destinations_if_changed(force=True)` 뒤에
```python
        self._directory = load_directory(directory_path())
        self.get_logger().info(f"건물 디렉터리 {len(self._directory)}곳: {directory_path()}")
```
`_reload_destinations_if_changed` 가 목적지를 다시 읽는 자리(YAML 교체 감지 분기 안)에 `self._directory = load_directory(directory_path())` 를 한 줄 더한다. `_on_user_audio` 의 `parse_intent_audio(...)` 호출에 인자
```python
                directory_block=format_directory_block(
                    self._directory, self._robot_state.current_building, self._robot_state.current_floor),
```
를 더하고, `parse_intent_audio` 시그니처에 `directory_block: str = ""` 를 추가해 `build_audio_prompt(destinations, robot_state, situation, directory_block)` 로 넘긴다.

`.env.example` 에:
```
# 건물 디렉터리(다른 층 장소, P1). 없으면 모델은 이 층만 안다.
VICA_DIRECTORY_YAML=~/vica_data/destinations/directory.yaml
```

- [ ] **Step 4: 로봇 데이터 파일**

```bash
cat > ~/vica_data/destinations/directory.yaml <<'EOF'
# 건물 디렉터리 — 이 층 밖의 장소. 이름·건물·층만. (P1, 2026-09-21)
- name: 세미나실
  building: 로봇관
  floor: 3
EOF
```
(실제 목록은 사용자가 채운다. 3층 세미나실은 역할극의 예시다.)

- [ ] **Step 5: 통과·임포트 확인**

```bash
.venv/bin/python -m pytest -q 2>&1 | tail -1
source /opt/ros/humble/setup.bash && source /home/ji_w/VICA-smarthandle/vica_ros2_ws/install/setup.bash && .venv/bin/python -c "import src.ros_node; print('ros_node import ok')"
```
Expected: 전부 passed, import ok.

- [ ] **Step 6: Commit**

```bash
git add src/building_directory.py src/langchain_intent_parser.py src/ros_node.py .env.example tests/test_building_directory.py tests/test_parser_audio.py
git commit -m "feat(voice): 건물 디렉터리 — 다른 층 장소를 알고 엘리베이터 안내를 제안

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: 지시문 규칙 — 시각·배터리·대장 질문

**Files:**
- Modify: `vica-voice-llm/src/langchain_intent_parser.py`(`build_audio_prompt` 의 question 규칙)
- Test: `vica-voice-llm/tests/test_parser_audio.py`(추가)

**Interfaces:**
- Produces: 지시문에 `[지금 상황]` 의 줄 이름으로 답하는 규칙이 들어간다.

- [ ] **Step 1: 실패하는 시험 쓰기**

`TestAudioPrompt` 에:
```python
    def test_question_rule_names_ledger_lines(self):
        text = parser.build_audio_prompt([DEST])
        for key in ("몇 층", "지금 어디", "어디 가려고", "몇 시", "배터리"):
            assert key in text, key
        assert "안내 데스크" in text   # 조언 금지 문구는 유지
```

- [ ] **Step 2: 실패 확인**

```bash
.venv/bin/python -m pytest -q tests/test_parser_audio.py -k ledger_lines
```
Expected: FAIL(`몇 시` 없음).

- [ ] **Step 3: 구현**

`build_audio_prompt` 의 question 규칙에서 `"몇 층이야?"는 로봇 상태 블록의 층, 그게 없으면 마지막으로 도착한 목적지의 층으로 답한다.` 문장을 다음으로 바꾼다:
```
"몇 층이야?"는 [지금 상황]의 건물/층, "지금 어디 있어?"는 지금 있는 곳, "지금 어디 가?"는 안내 중,
"아까 어디 갔었지?"는 직전에 간 곳, "어디 가려고 했더라?"는 하려다 만 곳, "몇 시야?"는 시각,
"배터리 얼마나 남았어?"는 배터리 줄로 답한다. 줄이 "모름"이면 한 문장으로 모른다고 한다.
```

- [ ] **Step 4: 통과 확인**

```bash
.venv/bin/python -m pytest -q tests/test_parser_audio.py
```
Expected: 전부 passed.

- [ ] **Step 5: Commit**

```bash
git add src/langchain_intent_parser.py tests/test_parser_audio.py
git commit -m "feat(voice): 지시문 — 대장 줄 이름으로 층·위치·행선지·시각·배터리 질문에 답하기

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 12: 실 API 스모크 — 대장 블록으로 다섯 질문

**Files:**
- Modify: `vica-voice-llm/tools/realtime_intent_smoke.py`(케이스 추가)

- [ ] **Step 1: 케이스 추가**

`main()` 의 모델 전결 케이스 블록 뒤에:
```python
    # 대장(P1) 케이스: [지금 상황] 블록만으로 다섯 질문에 답하는지.
    from src.building_directory import DirectoryEntry, format_directory_block
    from src.ledger_view import render_ledger
    from src.schema import RobotState
    st = RobotState(current_floor=4, current_building="로봇관", dialog_state="waiting", place_here="407호 앞",
                    place_here_dist_m=1.2, last_destination="407호", last_arrived_age_sec=660,
                    aborted_destination="화장실", wait_minutes=10, wait_left_sec=300)
    situation = render_ledger(st, awaiting_answer=False, now_text="15:40")
    directory = format_directory_block([DirectoryEntry("세미나실", "로봇관", 3)], "로봇관", 4)
    for q in ("우리 몇 층이야", "지금 어디 있어", "아까 어디 갔었지", "어디 가려고 했더라", "지금 몇 시야",
              "세미나실 갈 수 있어"):
        pcm = synth(tts, q)
        try:
            intent, heard, dt, info = parse_intent_audio(pcm, dests, situation=situation, directory_block=directory)
            print(f"[대장] '{q}' → intent={intent.intent} reply='{intent.reply[:40]}' dt={dt:.2f}s")
        except Exception as exc:
            print(f"[대장 실패] '{q}': {type(exc).__name__}: {exc}")
```

- [ ] **Step 2: 실행**

```bash
cd /home/ji_w/VICA-smarthandle/vica-voice-llm && timeout 240 .venv/bin/python tools/realtime_intent_smoke.py "화장실로 가자" 2>/dev/null | grep "\[대장"
```
Expected: 여섯 줄. "몇 층" → 4층, "어디" → 407호 앞, "갔었지" → 407호, "가려고" → 화장실, "몇 시" → 15:40, "세미나실" → 3층·엘리베이터 언급. 틀리는 줄이 있으면 Task 11 의 문장을 그 질문에 맞게 보강하고 재실행(최대 2회).

- [ ] **Step 3: Commit**

```bash
git add tools/realtime_intent_smoke.py
git commit -m "test(voice): 스모크에 대장 다섯 질문 케이스

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 13: 실기 준비·문서

**Files:**
- Modify: `vica-voice-llm/docs/superpowers/specs/2026-09-21-single-mouth-ledger-journal-design.md`(3.2 파일 위치 세 곳)
- Modify: `vica-voice-llm/docs/worklog-2026-09-20-realtime-intent-field.md`(§10 P1 실기 대본 추가)

- [ ] **Step 1: 스펙의 파일 위치를 계획과 맞춘다**

3.2 의 `~/vica_data/state/<map_id>/ledger.json` → `~/vica_data/destinations/<map_id>/ledger.json`(home.yaml 과 같은 폴더). 3.1 의 "지도 폴더의 `map.yaml`" → "목적지 폴더 `~/vica_data/destinations/<map_id>/map.yaml`". 건물 디렉터리 `~/vica_data/destinations/<building>/directory.yaml` → `~/vica_data/destinations/directory.yaml`(하나, `building` 칸으로 구분). 시각은 LLM 노드가 넣고 미션 방송에는 없다(3.1 표의 "시각" 출처를 "LLM 노드 시스템 시계"로).

- [ ] **Step 2: 실기 대본**

worklog 끝에:
```
## 10. P1 대장 실기 대본 (합격선: 대장 질문 5종 정답)

준비: ⑩ mission(재빌드 뒤)·⑫ llm 재기동. `ros2 topic echo /vica/robot_state --once` 에서 건물·층·dialog_state 확인.
1. 초기 위치 찍기 전 "우리 지금 어디 있어?" → "위치 미확인" 계열 답. 찍은 뒤 → "OO 앞".
2. "우리 몇 층이야?" → "로봇관 4층".
3. 407호 안내 → 도착 → "십 분" 대기 → 3분 넘게 침묵 → "비카야, 아까 어디 갔었지?" → 407호. "지금 몇 시야?" → 시각.
4. 화장실 제안까지 갔다가 "아니" → 잠시 뒤 "어디 가려고 했더라?" → 화장실.
5. "3층 세미나실 갈 수 있어?" → 3층에 있다·층 이동 불가·엘리베이터 제안(엘리베이터가 목록에 있으면 navigate 제안).
6. 주행 중 "지금 어디 가?" → 목적지 이름.
관찰: `[RT]` 로그의 reply, `ledger.json` 내용(도착 뒤 last_destination), 미션 재기동 뒤 "아까 어디 갔었지?"가 파일 복원으로 답하는지.
```

- [ ] **Step 3: Commit(두 저장소)**

```bash
cd /home/ji_w/VICA-smarthandle/vica-voice-llm
git add docs/superpowers/specs/2026-09-21-single-mouth-ledger-journal-design.md docs/worklog-2026-09-20-realtime-intent-field.md
git commit -m "docs(voice): P1 파일 위치 확정(목적지 폴더)·실기 대본

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-Review

- **스펙 3절 대조**: 건물·층(Task 2·5·6) / 지금 있는 곳(Task 3·5) / 가는 중·직전·하려다 만 곳(Task 3·5) / 대화 단계(Task 5) / 대기(Task 4·5) / 시각(Task 8·9, LLM 노드 시계) / 배터리(칸만, -1) / 건물 디렉터리·엘리베이터(Task 10) / RobotState 칸(Task 1·7) / 파일 보존·복원(Task 3·5) / LedgerView 호환 폴백(Task 8·9) / 질문→출처(Task 11·12) / 원천 차단(대화 단계가 블록에 들어감, Task 8). 빠진 것 없음.
- **자리표시자**: 없음. 모든 코드 블록은 실제 내용.
- **이름 일치**: `load_map_meta`/`MapMeta`(Task 2→5), `Ledger`/`LedgerStore`/`state_fields`/`place_here`(3→5), `wait_minutes_requested`/`wait_left_sec`(4→5), `render_ledger`(8→9·12), `format_directory_block`/`load_directory`/`directory_path`(10→12), `build_audio_prompt(..., situation, directory_block)`(9·10→12), `parse_intent_audio(..., directory_block=)`(10→12).

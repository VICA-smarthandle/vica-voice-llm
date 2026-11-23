# 빔포밍 회전 스위치 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 호출 방향 회전(`VICA_WAKE_DOA_TURN`)을 음성 재시작 없이 앱 대시보드 '빔포밍 회전' 스위치로 켜고 끈다.

**Architecture:** 음성 `ros_wakeword_node` 가 SetBool 서비스와 latched Bool 상태 토픽을 낸다(전방 초음파 스위치 11-18 과 같은 틀).
앱은 rosbridge 로 그 서비스를 부르고 상태 토픽만 보고 그린다. 미션·모드 노드는 그대로다.

**Tech Stack:** ROS 2 Humble rclpy(`std_srvs/SetBool`, `std_msgs/Bool`), Flutter(provider·rosbridge), pytest, flutter_test.

**Spec:** `vica-voice-llm/docs/superpowers/specs/2026-11-19-beamforming-turn-switch-design.md`

## Global Constraints

- 서비스 `/vica/wake_doa_turn/enable` (`std_srvs/srv/SetBool`), 상태 `/vica/wake_doa_turn/enabled` (`std_msgs/Bool`, RELIABLE·TRANSIENT_LOCAL·depth 1).
- 문구(캔버스 101번 보드, 사용자 승인): `빔포밍 회전` · `켜짐` · `잠시 꺼짐` · `회전 꺼짐`(11-19 검토 뒤 줄임) · `음성 꺼짐 · 바꿀 수 없음` · `빔포밍 회전을 켰습니다` · `빔포밍 회전을 껐습니다`.
- 기동 값은 `.env` `VICA_WAKE_DOA_TURN`(없으면 꺼짐). `.env` 값은 이 계획에서 바꾸지 않는다(실측 뒤 사용자 승인).
- 미션(`vica_ros2_ws`)·모드 노드(루트) 수정 없음.
- 음성 `after_feedback_waiting`(운영). 다른 세션 작업이 있어 **이 계획의 파일만** `git add`. 푸시·dev 머지 없음.
- 앱은 운영 브랜치 `after_feedback_waiting`(`VICA_Supervisor/`) 위에서 한다. **도킹 세션(feat/docking)과 수동 주행 세션(feat/app-manual-drive)이 운영 브랜치에 합치기를 끝냈다고 알린 뒤에만** Task 2·3 을 시작한다(11-19 도킹 세션 조율 — `supervisor_provider.dart` 가 겹침). 그 전엔 운영 앱 브랜치를 건드리지 않는다.
- `flutter test`·`flutter analyze`·`flutter build` 는 `systemctl --user is-active vica-drive.target` 가 `inactive` 일 때만(11-17 저널 정체 사고).

## Review Focus

1. 음성(⑫)이 꺼졌거나 다시 뜨는 중인데 앱이 마지막 latched 값(켜짐)을 들고 있다 → 카드는 `음성 꺼짐 · 바꿀 수 없음` + 스위치 잠금(모드 상태 `voice != 'active'` 도 모름으로 본다). Task 3 시험.
2. 서비스가 없는 옛 음성 노드(새 코드가 아직 안 뜬 ⑫) → 호출 실패 → 알림 `안 됐습니다: …`, 표시는 노드 상태 그대로. Task 2·3 시험.
3. 앱이 끊겼다 다시 붙음 → 상태를 모름으로 비웠다가 latched 값으로 다시 채운다. Task 2 시험.
4. 누른 직후 표시를 미리 바꾸지 않는다 — 표시는 노드가 낸 상태만 따른다. Task 3 시험.
5. `.env` 표기 변형(` ON `·`true`·`1`) → 켜짐, 그 밖(빈 값·`off`·`0`·없음) → 꺼짐. Task 1 시험.

---

### Task 1: 음성 노드 — 서비스·상태 토픽

**Files:**
- Create: `vica-voice-llm/src/wake_doa_turn.py`
- Modify: `vica-voice-llm/src/ros_wakeword_node.py` (import 44~64행, `_pub_wake_doa` 94행 근처, `_wake_doa_turn` 150~156행, 기동 로그 343~349행)
- Modify: `vica-voice-llm/.env.example` 64~67행 주석
- Test: `vica-voice-llm/tests/test_wake_doa_turn.py`

**Interfaces:**
- Produces (`src/wake_doa_turn.py`, ROS import 없음):
  - `WAKE_DOA_TURN_SERVICE = "/vica/wake_doa_turn/enable"`
  - `WAKE_DOA_TURN_STATE_TOPIC = "/vica/wake_doa_turn/enabled"`
  - `wake_doa_turn_from_env(value: str | None) -> bool`
  - `wake_doa_turn_reply(on: bool) -> str`
- Produces (노드): `_on_wake_doa_turn_set(self, request, response)`, `_publish_wake_doa_turn_state(self) -> None`

- [ ] **Step 1: 실패하는 시험 작성** — `tests/test_wake_doa_turn.py`. 노드는 `test_sw_aec_node_wiring.py` 처럼 AST 로 본다(rclpy 없이 돈다).

```python
from src.wake_doa_turn import (WAKE_DOA_TURN_SERVICE, WAKE_DOA_TURN_STATE_TOPIC,
                               wake_doa_turn_from_env, wake_doa_turn_reply)

def test_names_match_the_app():
    assert WAKE_DOA_TURN_SERVICE == "/vica/wake_doa_turn/enable"
    assert WAKE_DOA_TURN_STATE_TOPIC == "/vica/wake_doa_turn/enabled"

@pytest.mark.parametrize("value", ["on", "1", "true", " ON ", "True"])
def test_env_on(value): assert wake_doa_turn_from_env(value) is True

@pytest.mark.parametrize("value", [None, "", "off", "0", "false", "no"])
def test_env_off(value): assert wake_doa_turn_from_env(value) is False

def test_reply_text():
    assert wake_doa_turn_reply(True) == "빔포밍 회전을 켰습니다"
    assert wake_doa_turn_reply(False) == "빔포밍 회전을 껐습니다"
```

AST 시험(같은 파일, `_method`/`_calls` 도우미는 `test_sw_aec_node_wiring.py` 와 같게):
- `test_init_reads_env_through_helper`: `__init__` 에서 `self._wake_doa_turn` 대입의 값이 `wake_doa_turn_from_env(os.environ.get('VICA_WAKE_DOA_TURN'))`.
- `test_service_and_latched_state_are_created`: `__init__` 에 `self.create_service(SetBool, WAKE_DOA_TURN_SERVICE, self._on_wake_doa_turn_set)` 한 번, `create_publisher(Bool, WAKE_DOA_TURN_STATE_TOPIC, …)` 한 번이고 그 QoS 인자 글자에 `TRANSIENT_LOCAL` 이 있다. `_publish_wake_doa_turn_state()` 호출이 그 publisher 생성보다 뒤에 있다.
- `test_set_callback_updates_publishes_and_replies`: `_on_wake_doa_turn_set` 안에 `self._wake_doa_turn` 대입, `self._publish_wake_doa_turn_state()` 호출, `response.success` 대입, `wake_doa_turn_reply(` 호출, `return response`.
- `test_turn_off_still_keeps_doa_from_mission`: `_publish_wake_doa` 가 여전히 `if not self._wake_doa_turn:` 로 시작해 `return` 한다(기존 동작 잠금).
- `test_env_example_names_the_service`: `.env.example` 에 `/vica/wake_doa_turn/enable` 가 있다.

- [ ] **Step 2: 실패 확인**

Run: `cd vica-voice-llm && .venv/bin/python -m pytest tests/test_wake_doa_turn.py -q`
Expected: FAIL (`ModuleNotFoundError: src.wake_doa_turn`)

- [ ] **Step 3: 구현**
  - `src/wake_doa_turn.py`: 위 상수·두 함수. `from_env` 는 `strip().lower() in ("1", "on", "true")`.
  - 노드: `from std_srvs.srv import SetBool`, `from .wake_doa_turn import …`(상대 import, `.tts_queue` 다음 알파벳 순서 자리).
    `_wake_doa_turn = wake_doa_turn_from_env(os.environ.get("VICA_WAKE_DOA_TURN"))` — 기존 주석은 살리고 "앱 대시보드 '빔포밍 회전'으로 실행 중 바꿈(2026-11-19)" 한 줄 추가.
    publisher·service 는 `_wake_doa_turn` 이 정해진 뒤, 마이크 스레드 시작 전에 만들고 바로 `_publish_wake_doa_turn_state()`.
  - `_on_wake_doa_turn_set`: 값 대입 → 상태 발행 → 로그 `f"🧭 빔포밍 회전 {'켜짐' if on else '꺼짐'} (앱)"` → `success=True`, `message=wake_doa_turn_reply(on)`.
  - 기동 로그의 `호출 방향 회전 켜짐/꺼짐` → `빔포밍 회전 켜짐/꺼짐`.
  - `.env.example` 주석에 한 줄: 앱 '빔포밍 회전' 스위치 = `/vica/wake_doa_turn/enable`, 음성 재시작 때 이 값으로 돌아감.

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_wake_doa_turn.py -q` → PASS
Run: `.venv/bin/python -m pytest -q` → 전부 통과(기존 실패가 있으면 이 작업 전후 같은지 비교)

- [ ] **Step 5: 커밋**

```bash
git add src/wake_doa_turn.py src/ros_wakeword_node.py .env.example tests/test_wake_doa_turn.py
git commit -m "feat(voice): 빔포밍 회전 — 실행 중 켜고 끄기 /vica/wake_doa_turn/enable·enabled(11-19 사용자)"
```

### Task 2: 앱 provider — 상태 구독·서비스 호출

**Files:**
- Modify: `VICA_Supervisor/lib/providers/supervisor_provider.dart` (끊김 초기화의 `_frontUltrasonicOn = null` 옆, `frontUltrasonicStateTopic` 구독 옆, 전방 초음파 묶음 뒤에 새 묶음, `setModeStatusForTest` 옆 — 행 번호는 합치기 뒤 다시 찾는다)
- Test: `VICA_Supervisor/test/providers/wake_doa_turn_test.dart`

**Interfaces:**
- Produces:
  - `static const wakeDoaTurnStateTopic = '/vica/wake_doa_turn/enabled';`
  - `static const wakeDoaTurnService = '/vica/wake_doa_turn/enable';`
  - `bool? get wakeDoaTurnOn` — null = 모름
  - `Future<(bool, String)> setWakeDoaTurn(bool on)` — (받아들였나, 보일 문구). `setDetector` 와 같은 모양이라 카드의 `_run` 에 그대로 넣는다.
  - `@visibleForTesting void setWakeDoaTurnForTest(bool? on)`

- [ ] **Step 1: 실패하는 시험 작성** — `FakeRosServer`(`test/ros/fake_ros_socket.dart`)로 `service_call_messages_test.dart` 의 `run` 틀을 따른다.
  - `연결이 없으면 부르지 않고 (false, 'ROS Bridge에 연결되지 않았습니다.')`
  - `켜기: SetBool data=true 로 부르고 노드 문구를 그대로 돌려준다` — answer `{'success': true, 'message': '빔포밍 회전을 켰습니다'}` → `(true, '빔포밍 회전을 켰습니다')`.
  - `노드가 거절하면 (false, 노드 문구), 문구가 비면 '빔포밍 회전을 바꾸지 못했습니다.'`
  - `호출 자체가 실패하면(result: false) (false, …) 이고 연결 기록에 남는다`
  - `상태 토픽 true/false 를 받으면 wakeDoaTurnOn 이 따라간다` — `server.last.receive({'op': 'publish', 'topic': '/vica/wake_doa_turn/enabled', 'msg': {'data': true}})`.
  - `끊겼다 다시 붙으면 wakeDoaTurnOn 은 null 로 비워진다` — `reconnect_test.dart` 의 끊김 재현 방식을 따른다.

- [ ] **Step 2: 실패 확인** — `flutter test test/providers/wake_doa_turn_test.dart` → FAIL(없는 getter)

- [ ] **Step 3: 구현** — 전방 초음파(`setFrontUltrasonic`·`_handleFrontUltrasonicState`)와 같은 틀. 구독 type `std_msgs/Bool`, 끊김 때 `_wakeDoaTurnOn = null`. 성공 판정은 서비스 응답만 보고 `_wakeDoaTurnOn` 은 건드리지 않는다(상태 토픽이 정본). 실패만 `_addLog(LogFilter.connection, …)`.

- [ ] **Step 4: 통과 확인** — 같은 명령 PASS

- [ ] **Step 5: 커밋**

```bash
git add lib/providers/supervisor_provider.dart test/providers/wake_doa_turn_test.dart
git commit -m "feat(app): 빔포밍 회전 상태 구독·켜고 끄기 서비스(11-19 사용자)"
```

### Task 3: 앱 카드 — '빔포밍 회전' 줄

**Files:**
- Modify: `VICA_Supervisor/lib/widgets/drive_nodes_card.dart` (카메라 감지 줄·경고 칸 뒤, `Divider` 앞)
- Test: `VICA_Supervisor/test/widgets/drive_nodes_card_test.dart`

**Interfaces:**
- Consumes: Task 2 의 `wakeDoaTurnOn`, `setWakeDoaTurn`, `setWakeDoaTurnForTest`

- [ ] **Step 1: 실패하는 시험 작성** — `_FakeSupervisor` 에 `setWakeDoaTurn` 재정의(`calls.add(on ? 'beam_on' : 'beam_off')`, `reply` 반환). 기본값은 `wakeDoaTurnOn == null` 이라 기존 시험의 '켜짐' 개수는 그대로다.
  - `빔포밍 켜짐: '빔포밍 회전' 줄과 켜진 스위치, 경고 칸 없음`
  - `빔포밍 꺼짐: '잠시 꺼짐' 과 경고 칸 '회전 꺼짐'`
  - `스위치를 끄면 beam_off 를 보내고 노드 문구를 알림으로 띄운다` — reply `(true, '빔포밍 회전을 껐습니다')` → SnackBar 글자 `빔포밍 회전을 껐습니다`. **상태 주입을 안 바꿨으므로 줄은 여전히 '켜짐'**(Review Focus 4).
  - `거절되면 '안 됐습니다: …' 를 띄운다` — reply `(false, '서비스 없음')`.
  - `상태 모름이면 '음성 꺼짐 · 바꿀 수 없음' 이고 스위치를 눌러도 아무것도 안 보낸다`
  - `음성 유닛이 active 가 아니면 latched 값이 true 여도 모름으로 그린다` — `drive()` 의 `voice: 'inactive'` + `setWakeDoaTurnForTest(true)`.
  - 아이콘: 켜짐 `Icons.explore_outlined`, 꺼짐 `Icons.explore_off_outlined`.

- [ ] **Step 2: 실패 확인** — `flutter test test/widgets/drive_nodes_card_test.dart` → FAIL

- [ ] **Step 3: 구현** — 기존 `_Row`·`_Note`·`_run` 을 그대로 쓴다. 모름 = `wakeDoaTurnOn == null || status.voice != 'active'`.
  색: 켜짐 `VicaColors.green`, 꺼짐 `Color(0xFF8A5F1C)`(카메라 감지 '잠시 꺼짐'과 같음), 모름 `VicaColors.muted`. 파일 머리 주석에 11-19 요청 한 줄.

- [ ] **Step 4: 통과 확인**
  - `flutter test test/widgets/drive_nodes_card_test.dart test/screens/autostart_layout_test.dart test/providers/wake_doa_turn_test.dart` → PASS
  - `flutter analyze lib/widgets/drive_nodes_card.dart lib/providers/supervisor_provider.dart` → No issues

- [ ] **Step 5: 커밋**

```bash
git add lib/widgets/drive_nodes_card.dart test/widgets/drive_nodes_card_test.dart
git commit -m "feat(app): 대시보드 주행 설정에 '빔포밍 회전' 스위치(11-19 사용자, 캔버스 101)"
```

### Task 4: 올리기와 켜짐 확인 (사용자와 함께)

각 단계는 실행 전 사용자 승인.

- [ ] **Step 1: ⑫ 음성 재시작** — `systemctl --user restart vica-llm`. 음성이 약 10 s 끊긴다.
  확인: `journalctl --user -u vica-llm -n 200 --no-pager | grep -e '빔포밍 회전'` 에 기동 줄 `빔포밍 회전 꺼짐`(`.env` 에 키 없음).
- [ ] **Step 2: 서비스 확인** — `ros2 service call /vica/wake_doa_turn/enable std_srvs/srv/SetBool "{data: true}"` (timeout 으로 감싸지 않는다)
  → `success=True, message='빔포밍 회전을 켰습니다'`. 이어서 `{data: false}` 로 되돌린다.
- [ ] **Step 3: 앱 반영** — 웹(5000) 빌드는 주행 묶음이 꺼져 있을 때, APK 는 노트북에서.
- [ ] **Step 4: 방향 기준 재측정(설계 실기 순서 1)** — 스위치 꺼진 채, 로봇은 움직이지 않음.
  정면·손잡이·좌우를 걸으며 부르고 `journalctl --user -u vica-llm -f | grep '🧭'` 의 각도를 본다. 기준: 정면 ≈0°, 손잡이 ≈180°, 오른쪽으로 걸으면 각도 증가.
- [ ] **Step 5: 켜짐 확인** — 미션(⑩)이 떠 있고 대기(IDLE)일 때, 주변을 비우고:
  1. 앱에서 켜면 줄이 `켜짐`(노드가 낸 상태), 음성 로그 `🧭 빔포밍 회전 켜짐 (앱)`.
  2. 로봇 옆에서 "비카야" → 음성 로그 `🧭 호출 방향 N°` (뒤에 `(방향 회전 꺼짐 — 미션에 안 보냄)` 이 **없어야** 함).
  3. 미션 로그 `'비카야' 방향 doa=N° yaw=M°: idle -> seeking` → 로봇이 부른 쪽으로 돈다. 손잡이 쪽에서 부르면 돌지 않고 접근 질문.
  4. 앱에서 끄면 다시 부를 때 꼬리 `(방향 회전 꺼짐 — 미션에 안 보냄)`, 회전 없음.
  5. (11-19 검토) 앱을 먼저 연결해 둔 채 주행을 켜도, 음성이 뜨고 몇 초 안에 카드가 '음성 꺼짐'에서
     켜짐/잠시 꺼짐으로 바뀐다(1 s 재발행).
- [ ] **Step 6: 통과하면** `.env` `VICA_WAKE_DOA_TURN=on` 을 넣을지 사용자에게 묻는다.

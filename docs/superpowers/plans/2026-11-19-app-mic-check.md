# 앱 마이크 점검 · 음성 노드 켜기/끄기 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 앱에서 음성 노드(`vica-llm.service`)만 끄고 켜고, 마이크 현장 점검을 앱 화면으로만 한다.

**Architecture:** 모드 노드(루트 `scripts/autostart`)가 음성 유닛 start/stop 과 점검기 일회성 유닛(`systemd-run --user --unit=vica-miccheck`)을
맡고, 거절 규칙은 `mode_logic.py` 순수 함수로 둔다. 점검기(`vica-voice-llm/tools/mic_venue_check.py`)는 판정 로직을 그대로 두고
휴대폰 페이지를 지워 `/ws`(8770)만 연다. 앱(`VICA_Supervisor`)은 그 WebSocket 에 직접 붙는 Flutter 점검 화면을 새로 만든다.

**Tech Stack:** Python 3.10(rclpy·unittest/pytest) · systemd user 유닛 · websockets 15 · Flutter/Dart(`web_socket_channel` 3, `provider`)

**Spec:** `docs/superpowers/specs/2026-11-19-app-mic-check-design.md`(이 worktree, 9e30ba6·9ad4eaf). 화면 배치·문구는
"앱 UI 사전확인" 캔버스 보드 104~111(https://claude.ai/artifact/3gzpfMHhQZ2A9wkYG59ewU, 11-19 승인)이 정본이다.

## 작업 위치

| 저장소 | worktree | 브랜치 | 시험 명령 |
|---|---|---|---|
| 음성 | `~/wt-mic-check-voice` | `feat/app-mic-check`(있음) | `/home/ji_w/VICA-smarthandle/vica-voice-llm/.venv/bin/python -m pytest <파일> -q` |
| 루트(모드 노드) | `~/wt-autostart` | `feat/autostart`(있음) | `python3 -m pytest scripts/tests/<파일> -q` |
| 앱 | `~/wt-mic-check-app` (Task 0 에서 만듦) | `feat/app-mic-check` | `flutter test <파일>` · `flutter analyze` |

## Global Constraints

- 사용자 문구는 캔버스 보드 104~111 승인 문구 그대로. 이 계획에 따옴표로 적은 문장은 바꾸지 않는다.
- 새 의존성 없음(앱은 `web_socket_channel` 이미 있음, 음성은 `websockets` 15 이미 있음).
- 주행 묶음이 켜져 있으면 `flutter test`·`flutter build`·`flutter analyze`·colcon 을 돌리지 않는다(11-17 Nav2 기동 불가 사고).
  돌리기 전 `systemctl --user is-active vica-drive.target` 이 `inactive` 인지 본다. 켜져 있으면 멈추고 사용자에게 묻는다.
- 스택·유닛을 임의로 켜거나 재시작하지 않는다(사용자 11-18). `bin` 반영·`vica-mode` 재시작·운영 브랜치 합치기·웹 5000 교체는
  배포 Task 에서 사용자 승인을 받은 뒤에만.
- 푸시하지 않는다. 커밋 메시지는 한국어, 기존 문체(`feat(…)`/`fix(…)`/`docs(…)`), 끝에 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- 고정 값: 점검 유닛 `vica-miccheck.service` · 포트 `8770` · 실행 표지 환경 변수 `VICA_MIC_CHECK_LAUNCHER=mode` ·
  '움직이는 중' 판정 창 `1.0 s`(`/cmd_vel_safe` 0 아님) · `Twist` 0 판정 문턱 `1e-3`.
- 실제 마이크·스피커·ROS 에 기대는 시험은 자동화하지 않는다(음성 CLAUDE.md). 노드 파일은 정적(문자열) 시험만.
- 판정 로직(`mic_check_rules.judge`·`mic_check_session` 채점)과 서버↔화면 메시지 형식은 바꾸지 않는다.

## Review Focus

1. **점검기가 준비 중에 죽음**(8770 사용 중·reSpeaker 없음 → 유닛이 곧 끝남): 앱 화면이 '준비 중'에 멈춰 있으면 안 된다.
   모드 상태 `mic_check.state` 가 활성 → `inactive`/`failed` 로 바뀌면 "점검기가 끝났습니다"를 띄우고 화면을 닫는다. → Task 12 시험.
2. **실패한 일회성 유닛이 남아 다음 시작을 막음**: `systemd-run` 은 같은 이름의 실패 유닛이 남아 있으면 거절한다. 명령에 `--collect`. → Task 7 시험.
3. **끊긴 동안 누른 버튼**: 보내지 못하면 조용히 버리지 않고 "연결이 끊겨 보내지 못했습니다 · 다시 붙은 뒤 누르세요" 스낵. → Task 10·11 시험.
4. **점검기가 내려가는 중(칩 되돌리는 중)에 음성 켜기**: `deactivating` 도 점검 중으로 보고 거절한다. → Task 1 시험.
5. **점검 중 로봇 끄기**: 끄기 스크립트가 `vica-miccheck` 를 주행 묶음보다 먼저 멈춘다(칩 되돌리기). → Task 8 시험.

---

## 1단계 — 음성 노드 스위치 (따로 먼저 운영 반영)

### Task 0: 앱 worktree 만들기

- [ ] **Step 1:** `git -C /home/ji_w/VICA-smarthandle/VICA_Supervisor worktree add -b feat/app-mic-check /home/ji_w/wt-mic-check-app after_feedback_waiting`
  Expected: `Preparing worktree (new branch 'feat/app-mic-check')`, HEAD = 운영 `after_feedback_waiting` 최신(11-19 21:39 기준 7a938be).
- [ ] **Step 2:** `cd /home/ji_w/wt-mic-check-app && flutter pub get` (주행 묶음 꺼짐 확인 뒤). Expected: `Got dependencies!`

### Task 1: 거절 규칙 — 움직임·음성 노드 (`mode_logic.py`)

**Files:**
- Modify: `~/wt-autostart/scripts/autostart/mode_logic.py` (Snapshot 46-80, 상수는 `DRIVE_TARGET` 근처, 함수는 `voice_switch_refusal` 414 뒤)
- Test: `~/wt-autostart/scripts/tests/test_mode_logic.py` (새 클래스 `VoiceNodeRefusalTest`)

**Interfaces:**
- Produces:
  - `Snapshot.cmd_moving: bool = False` — 최근 `CMD_MOVING_SEC` 안에 `/cmd_vel_safe` 가 0 이 아니었다.
  - `CMD_MOVING_SEC = 1.0` · `MIC_CHECK_UNIT = "vica-miccheck.service"` · `MOVING_REFUSAL = "로봇이 움직이는 중입니다. 멈춘 뒤 다시 누르세요."`
  - `twist_moving(linear_x: float, angular_z: float) -> bool` — `abs(x) > 1e-3 or abs(z) > 1e-3`
  - `robot_in_motion(snap: Snapshot) -> bool` — `bool(snap.robot_moving) or snap.cmd_moving`
  - `mic_check_active(snap: Snapshot) -> bool` — `_state(snap, MIC_CHECK_UNIT)` 가 `active`·`activating`·`deactivating`
  - `voice_node_refusal(snap: Snapshot, manifest: dict, on: bool) -> str | None`

- [ ] **Step 1: 실패하는 시험 쓰기** — `VoiceNodeRefusalTest`:
  - `voice_node_refusal(snapshot(), MANIFEST, on=False)` → `None`
  - `snapshot(robot_moving=True)` → `MOVING_REFUSAL` (on=False·True 둘 다)
  - `snapshot(robot_moving=False, cmd_moving=True)` → `MOVING_REFUSAL` (사람 접근·수동 주행·도킹은 `is_moving` 거짓)
  - `idle_snapshot()` → `"주행 모드가 아닙니다."`
  - 점검 유닛 `("deactivating","stop-sigterm","")` 를 `units` 에 넣은 `snapshot` 으로 on=True → `"마이크 점검 중입니다. 점검을 끝낸 뒤 켜세요."`, on=False → `None`
  - `llm` 을 뺀 manifest → `"음성 유닛이 없습니다."`
  - `twist_moving(0.0, 0.0)` 거짓, `twist_moving(0.0005, 0.0)` 거짓, `twist_moving(0.0, -0.2)` 참
- [ ] **Step 2:** `cd ~/wt-autostart && python3 -m pytest scripts/tests/test_mode_logic.py -q -k VoiceNode` → FAIL(`AttributeError`)
- [ ] **Step 3: 구현** — 거절 순서: 유닛 없음 → 주행 모드 아님 → 움직임 → (on 일 때만) 점검 중.
- [ ] **Step 4:** `python3 -m pytest scripts/tests/test_mode_logic.py -q` → 전부 PASS(기존 포함)
- [ ] **Step 5: 커밋** `feat(autostart): 음성 노드 켜기·끄기 거절 규칙 — 움직임에 /cmd_vel_safe 1 s 포함(11-19 설계)`

### Task 2: 모드 노드 — `/cmd_vel_safe` 구독·`voice_on|off` 서비스

**Files:**
- Modify: `~/wt-autostart/scripts/autostart/vica_mode_node.py` (구독 135-139, 서비스 141-148, `collect` 의 `systemctl show` 대상·`Snapshot(...)`, `on_detector_off` 뒤에 새 함수)
- Test: Create `~/wt-autostart/scripts/tests/test_mode_node_voice_mic.py` (노드 원문 정적 시험 — `test_autostart_shutdown.py` 40행과 같은 방식)

**Interfaces:**
- Consumes: Task 1 의 `ml.twist_moving`·`ml.CMD_MOVING_SEC`·`ml.voice_node_refusal`·`ml.MIC_CHECK_UNIT`
- Produces: 서비스 `/vica/mode/voice_on`·`/vica/mode/voice_off`(std_srvs/Trigger). 받으면 `"음성 노드를 켭니다"` / `"음성 노드를 껐습니다"`.

- [ ] **Step 1: 실패하는 시험** — 노드 원문에 `'"/vica/mode/voice_on"'`·`'"/vica/mode/voice_off"'`·`'"/cmd_vel_safe"'`·`"ml.voice_node_refusal"`·
  `"ml.MIC_CHECK_UNIT"`·`"cmd_moving="` 가 있다. `from geometry_msgs.msg import Twist` 가 있다.
- [ ] **Step 2:** `python3 -m pytest scripts/tests/test_mode_node_voice_mic.py -q` → FAIL
- [ ] **Step 3: 구현**
  - `on_cmd_safe(msg: Twist)`: `ml.twist_moving(msg.linear.x, msg.angular.z)` 이면 `self.cmd_nonzero_at = time.monotonic()`(lock 안).
  - `collect`: `systemctl show` 대상에 `ml.MIC_CHECK_UNIT` 추가, `cmd_moving = cmd_nonzero_at is not None and now - cmd_nonzero_at < ml.CMD_MOVING_SEC`.
  - `on_voice_node(response, on: bool)`: `self.run(response, ml.voice_node_refusal(snap, self.manifest, on), ["start"|"stop", "--no-block", llm 유닛], 문구)`.
- [ ] **Step 4:** `python3 -m pytest scripts/tests -q` → PASS · `python3 -m py_compile scripts/autostart/vica_mode_node.py` → 출력 없음
- [ ] **Step 5: 커밋** `feat(autostart): 모드 노드 voice_on|off 서비스와 /cmd_vel_safe 움직임 판정`

### Task 3: 앱 — 음성 노드 스위치 (보드 104 ①②③)

**Files:**
- Modify: `~/wt-mic-check-app/lib/core/mode_contract.dart`, `lib/models/mode_status.dart`, `lib/providers/supervisor_provider.dart`(943 근처),
  `lib/widgets/drive_nodes_card.dart`(카메라 감지 줄 뒤)
- Test: `test/models/mode_status_test.dart`, `test/widgets/drive_nodes_card_test.dart`

**Interfaces:**
- Produces: `kVoiceNodeOnService = '/vica/mode/voice_on'` · `kVoiceNodeOffService = '/vica/mode/voice_off'` ·
  `ModeStatus.voiceOn` (`voice` 가 `active`·`activating`) · `SupervisorProvider.setVoiceNode(bool on) → Future<(bool, String)>`
  (`_callModeService(…, '음성 노드 켜기'|'음성 노드 끄기')`)

- [ ] **Step 1: 실패하는 시험**
  - `mode_status_test`: `voice: 'active'`·`'activating'` → `voiceOn` 참, `'inactive'`·`'failed'`·`'none'` → 거짓.
  - `drive_nodes_card_test`(기존 감지기 스위치 시험의 pump 도우미를 따름):
    - `voice: 'active'` → `ValueKey('drive_nodes_voice_node_switch')` 스위치 값 참, 제목 `'음성 노드'`, 상태 `'켜짐'`.
    - `voice: 'inactive'` → 스위치 거짓, 상태 `'꺼짐'`, 안내 `"호출어·‘멈춰’ 안 됨"` 보임.
    - `voice: 'deactivating'` → 스위치 `onChanged` null.
    - 스위치를 누르면 `setVoiceNode(false)` 가 불리고 받은 문구가 스낵으로, 거절이면 `'안 됐습니다: '` + 문구.
- [ ] **Step 2:** `flutter test test/models/mode_status_test.dart test/widgets/drive_nodes_card_test.dart` → FAIL
- [ ] **Step 3: 구현** — 줄 순서: 카메라 감지 · 음성 노드 · 빔포밍 회전 · 음성 LLM(보드 104 ①). 아이콘 `Icons.graphic_eq`.
  꺼짐 안내는 기존 `_Note(warning: true, …)`. 상태 글자: `켜짐`·`꺼짐`·`켜는 중`(activating)·`끄는 중`(deactivating).
- [ ] **Step 4:** 같은 명령 PASS · `flutter analyze` → `No issues found!`
- [ ] **Step 5: 커밋** `feat(app): 주행 설정 칸에 음성 노드 켜기·끄기 스위치(캔버스 104)`

### Task 4: 1단계 운영 반영 (사용자 승인 뒤)

- [ ] **Step 1:** 사용자에게 승인 받기: ① 모드 노드 `bin` 반영 + `vica-mode` 재시작 ② 앱 `feat/app-mic-check` → 운영 `after_feedback_waiting` 합치기 + 웹 5000 빌드.
  주행 묶음이 꺼져 있어야 한다.
- [ ] **Step 2:** `cp -a ~/.config/vica-terminator/bin ~/vica_data/autostart_bin_backup/2026-11-19_before_voice_node` 뒤
  `install -m 0644 ~/wt-autostart/scripts/autostart/{mode_logic.py,vica_mode_node.py} ~/.config/vica-terminator/bin/` →
  `systemctl --user restart vica-mode` → `journalctl --user -u vica-mode -n 5 --no-pager` 에 `모드 노드 시작` 줄.
- [ ] **Step 3:** 운영 앱 체크아웃에서 `git merge --no-ff feat/app-mic-check` → `flutter build web --release` → 앱에서 새로고침.
- [ ] **Step 4: 실기** — 주행 켬 → 스위치 끔 → `systemctl --user is-active vica-llm` = `inactive` → 켬 → `active`. 움직이는 중 거절 문구 확인.
  결과를 `~/vica_data/field_test_20261119_mic_check/RESULT.md` 에 적는다.

## 2단계 — 점검기를 앱 전용으로 · 모드 노드 점검 서비스

### Task 5: 점검기 — 실행 표지·페이지 폐지 (`vica-voice-llm`)

**Files:**
- Modify: `~/wt-mic-check-voice/tools/mic_venue_check.py`(main 346-378, 머리 문서), `tools/mic_check_web.py`(생성자·`_process_request` 88-96)
- Delete: `tools/mic_check_page.html`, `tests/test_mic_check_page.py`
- Test: `tests/test_mic_check_web.py`, `tests/test_mic_venue_check_tool.py`

**Interfaces:**
- Produces: `LAUNCHER_ENV = "VICA_MIC_CHECK_LAUNCHER"` · `launched_by_mode(environ: Mapping[str, str]) -> bool`(값이 `"mode"` 일 때만 참) ·
  `LiveServer(host: str = "0.0.0.0", port: int = DEFAULT_PORT)` — 페이지 인자 없음.

- [ ] **Step 1: 실패하는 시험**
  - `test_mic_check_web.py`: `test_page_and_404` → `test_root_is_404_and_only_ws_opens` — `GET /` 가 404, 본문에 `"앱에서"`. 나머지 6개는 `LiveServer(port=0)` 로.
  - `test_mic_venue_check_tool.py`: `launched_by_mode({})`·`({"VICA_MIC_CHECK_LAUNCHER": "x"})` 거짓, `"mode"` 참.
    환경 변수 없이 `main([])` → 2 를 돌려주고 출력에 `"마이크 점검은 앱에서만 합니다"`, 프로세스 검사(`_proc_cmdlines`)·서버를 부르기 전에 끝난다(monkeypatch 로 부르면 실패하게).
    `main(["--restore"])` 은 표지 없이도 `_restore_cli` 로 간다.
- [ ] **Step 2:** `…/python -m pytest tests/test_mic_check_web.py tests/test_mic_venue_check_tool.py -q` → FAIL
- [ ] **Step 3: 구현** — 표지 검사는 `--restore`·`--demo` 처리 뒤, `voice_stack_running` 앞. 거절 출력:
  `"마이크 점검은 앱에서만 합니다. (화면 개발: --demo · 비상 되돌리기: --restore)"`. `PAGE` 읽기와 `"휴대폰으로 여세요"` 출력을
  `"  앱 연결 대기: ws://<이 기기>:{port}/ws"` 로 바꾼다. 404 본문 `"마이크 점검은 앱에서 합니다\n"`.
- [ ] **Step 4:** `…/python -m pytest tests -q -k "mic_check or venue or env_file"` → PASS
- [ ] **Step 5: 커밋** `feat(voice): 마이크 점검기 — 앱 전용(모드 노드 표지 VICA_MIC_CHECK_LAUNCHER), 휴대폰 페이지 폐지`

### Task 6: 점검기 문구·문서

**Files:**
- Modify: `src/mic_check_session.py`(278-283 바쁨, 312 칩 실패, 652 저장, 670 되돌리기), `src/mic_check_rules.py:57` `TITLES`,
  `tools/mic_venue_check.py` 출력 문구(256·407·452), `docs/mic-venue-check.md`(앱 기준으로 다시 씀),
  `docs/superpowers/specs/2026-10-12-mic-venue-check-design.md`(머리에 "11-19 휴대폰 페이지 폐지 — 2026-11-19-app-mic-check-design.md 참조")
- Test: `tests/test_mic_check_rules.py:37`, `tests/test_mic_check_session.py`(옛 문구를 단언하는 곳 — `grep -n "⑫를 다시\|시연 가능\|도구를 다시\|끝난 뒤 다시 누르세요" tests`)

- [ ] **Step 1: 시험을 새 문구로 바꾸기**
  - `TITLES[PASS] == "사용 가능"`, `TITLES[WARN] == "사용 가능 · 확인 필요"`(나머지 셋 그대로)
  - 저장 토스트 `"저장했습니다 · 음성 노드를 다시 켜야 적용됩니다"`, 되돌리기 `f"{이름} 으로 되돌렸습니다 · 음성 노드를 다시 켜야 적용됩니다"`
  - 바쁨 토스트 `"지금은 녹음 중입니다 · 끝난 뒤 다시 누르세요"`(로봇은 `"지금은 로봇이 말하는 중입니다 · …"`)
  - 칩 실패 토스트 끝 `" · 끝내기 뒤 다시 시작하세요"`
- [ ] **Step 2:** `…/python -m pytest tests -q -k "mic_check"` → FAIL
- [ ] **Step 3:** 문구 바꾸기 · 문서 고치기(설치·실행은 "앱 → 주행 설정 칸 또는 모드 선택 → 마이크 점검", CLI 는 `--demo`·`--restore` 만)
- [ ] **Step 4:** `…/python -m pytest tests -q` → 전부 PASS
- [ ] **Step 5: 커밋** `feat(voice): 점검기 문구 — '사용 가능'·'음성 노드' 로, 사용법 문서를 앱 기준으로(11-19 캔버스 110·111)`

### Task 7: 거절 규칙 — 마이크 점검 (`mode_logic.py`)

**Files:**
- Modify: `~/wt-autostart/scripts/autostart/mode_logic.py` (`_derive_core` status, `drive_start_refusal` 320, `map_change_block` 542, 새 함수)
- Test: `scripts/tests/test_mode_logic.py` (새 클래스 `MicCheckTest`)

**Interfaces:**
- Consumes: Task 1 의 `MIC_CHECK_UNIT`·`mic_check_active`·`robot_in_motion`·`MOVING_REFUSAL`
- Produces:
  - `MIC_CHECK_PORT = 8770` · `MIC_CHECK_LAUNCHER_ENV = "VICA_MIC_CHECK_LAUNCHER"`
  - `mic_check_start_refusal(snap, manifest) -> str | None` — 순서: `"이미 점검 중입니다."` → `"지도를 그리는 중입니다."`(mapping_state 가
    `MAPPING_IDLE` 밖이거나 cartographer 프로세스) → `MOVING_REFUSAL` → `"음성 노드가 켜져 있습니다. 먼저 끄세요."`(llm 유닛
    active·activating·deactivating) → `"터미널에서 띄운 음성이 있습니다. 그 칸을 끈 뒤 다시 누르세요."`(llm `guard` 프로세스나 `guard_cmdline` 적중)
  - `mic_check_stop_refusal(snap) -> str | None` — 점검 중 아니면 `"점검 중이 아닙니다."`
  - `mic_check_side_stops(snap, manifest) -> list[str]` — 주행 켜짐이고 감지기 유닛이 active·activating 이면 `["vica-detector.service"]`, 아니면 `[]`
  - `mic_check_command(voice_root: str, map_id: str, destinations: str | None) -> list[str]`:
    `["systemd-run", "--user", "--collect", f"--unit={MIC_CHECK_UNIT}", f"--setenv={MIC_CHECK_LAUNCHER_ENV}=mode",
    "--setenv=PYTHONUNBUFFERED=1", f"--working-directory={voice_root}", f"{voice_root}/.venv/bin/python", "-m",
    "tools.mic_venue_check", "--name", map_id or "점검"]` + (`["--destinations", destinations]` 이 있을 때만)
  - `/vica/mode_status` 에 `"mic_check": {"state": _state(snap, MIC_CHECK_UNIT), "port": MIC_CHECK_PORT}`
  - `drive_start_refusal`: `drive_on` 검사 바로 뒤 점검 중이면 `"마이크 점검 중입니다. 점검을 끝낸 뒤 켜세요."`
  - `map_change_block`: 맨 앞에서 점검 중이면 `"마이크 점검 중입니다. 점검을 끝낸 뒤 바꾸세요."`

- [ ] **Step 1: 실패하는 시험** — 위 문구 하나하나에 한 시험씩 + 아래:
  - 주행 꺼짐(`idle_snapshot()`)·음성 꺼짐이면 시작 거절 없음(모드 선택 화면 경로)
  - 감지기 꺼짐·주행 꺼짐이면 `mic_check_side_stops` 가 `[]`
  - `mic_check_command("/v", "", None)` 에 `"--collect"`·`"--name", "점검"` 이 있고 `"--destinations"` 는 없음
  - `derive_status(...)["mic_check"] == {"state": "inactive", "port": 8770}`(유닛이 없을 때)
- [ ] **Step 2:** `python3 -m pytest scripts/tests/test_mode_logic.py -q -k MicCheck` → FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4:** `python3 -m pytest scripts/tests -q` → PASS
- [ ] **Step 5: 커밋** `feat(autostart): 마이크 점검 거절 규칙·실행 명령 — 점검 중 주행 켜기·지도 바꾸기 거절(11-19 설계)`

### Task 8: 모드 노드 — `mic_check_start|stop` · 끄기 스크립트

**Files:**
- Modify: `~/wt-autostart/scripts/autostart/vica_mode_node.py`, `scripts/autostart/vica_shutdown.sh`(15행 앞)
- Test: `scripts/tests/test_mode_node_voice_mic.py`, `scripts/tests/test_autostart_shutdown.py`(26행 순서 시험)

**Interfaces:**
- Consumes: Task 7 의 함수 전부
- Produces: 서비스 `/vica/mode/mic_check_start`(받으면 `"마이크 점검을 시작합니다 · 준비 1~2분"`)·`/vica/mode/mic_check_stop`(받으면 `"점검을 끝냅니다"`)

- [ ] **Step 1: 실패하는 시험**
  - 노드 원문에 `'"/vica/mode/mic_check_start"'`·`'"/vica/mode/mic_check_stop"'`·`"ml.mic_check_command"`·`"ml.mic_check_side_stops"`·`"VICA_VOICE_ROOT"`.
  - 끄기 스크립트: `"stop vica-miccheck.service"` 가 `"stop vica-drive.target"` 보다 앞.
- [ ] **Step 2:** `python3 -m pytest scripts/tests/test_mode_node_voice_mic.py scripts/tests/test_autostart_shutdown.py -q` → FAIL
- [ ] **Step 3: 구현**
  - `VOICE_ROOT = Path(os.environ.get("VICA_VOICE_ROOT", str(Path.home() / "VICA-smarthandle/vica-voice-llm")))`,
    목적지 파일 `Path.home() / "vica_data/destinations" / <current_map> / "destinations.yaml"`(없으면 None).
  - 시작: 거절 → `side_stops` 가 있으면 `systemctl stop --no-block <그것>` → `subprocess.run(mic_check_command(...), timeout=SYSTEMCTL_TIMEOUT)`;
    실패 문구는 기존 `run()` 과 같은 꼴(`"systemctl 실패: …"`). 끝나면 `self.tick()`.
  - 중지: `self.run(response, ml.mic_check_stop_refusal(snap), ["stop", "--no-block", ml.MIC_CHECK_UNIT], "점검을 끝냅니다")`.
  - 끄기 스크립트: `systemctl --user stop vica-miccheck.service 2>/dev/null || true`.
- [ ] **Step 4:** `python3 -m pytest scripts/tests -q` → PASS · `bash -n scripts/autostart/vica_shutdown.sh` → 출력 없음
- [ ] **Step 5: 커밋** `feat(autostart): 모드 노드 mic_check_start|stop — 일회성 유닛 vica-miccheck, 시작 때 카메라 감지 끔`

## 3단계 — Flutter 점검 화면

### Task 9: 메시지 모델 (`lib/models/mic_check.dart`)

**Files:**
- Create: `~/wt-mic-check-app/lib/models/mic_check.dart`
- Test: Create `test/models/mic_check_test.dart`

**Interfaces:**
- Produces (모두 불변, `fromJson` 은 빠진 칸에 기본값):
  - `MicCheckStep(key, no, of, name, say, rep, reps, seconds, recording, robot)`
  - `MicCheckStatus(place, step, steps: List<({String key, String name, int need})>, progress: Map<String, List<String>>, values, envValues: Map<String, Object?>,
    ranges: Map<String, ({double min, double max, double step})>, names, envKeys: Map<String, String>, agcChoices, gainChoices: List<double>,
    echoChoices: Map<String, String>, gainFactory: double, chipOk: bool, chipProblems: List<String>)`
  - `MicCheckFrame(rms, wake, stop: double, vad: bool?, robot, muted: bool, state: String, noise: double?, recLeft: double?, busy: String?)`
  - `MicCheckEvent(grade, title, detail, time)` · `MicCheckRow(key, name, grade, detail, todo)`
  - `MicCheckResults(title, overall, rows: List<MicCheckRow>, first: List<MicCheckRow>?, stale: bool)` · `MicCheckSavePreview(lines, backupDir)`
  - `sealed class MicCheckMessage` + `static MicCheckMessage? parse(String raw)` — 하위형 `MicPrep(text)`·`MicStatusMessage(status)`·
    `MicFrameMessage(frame)`·`MicEventMessage(event)`·`MicToast(text)`·`MicBusy(text)`·`MicResultsMessage(results)`·`MicSavePreviewMessage(preview)`.
    모르는 `type`·JSON 아님·`type` 없음 → `null`.
  - 형식의 정본: `src/mic_check_session.py` `_publish_status`(769)·`_live_frame`(686)·`_publish_results`(797)·`_save_preview`(635)·`_event`(676).

- [ ] **Step 1: 실패하는 시험** — 8종 각각을 Python 이 내는 모양 그대로의 JSON 문자열로 파싱해 칸 값을 단언
  (`status.step.no == 3`, `ranges['listen_min_rms']!.max == 0.040`, `frame.recLeft == null` 일 때 등), 모르는 type·깨진 JSON → `null`.
- [ ] **Step 2:** `flutter test test/models/mic_check_test.dart` → FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4:** PASS
- [ ] **Step 5: 커밋** `feat(app): 마이크 점검기 메시지 8종 모델`

### Task 10: 연결 (`lib/services/mic_check_controller.dart`)

**Files:**
- Create: `lib/services/mic_check_controller.dart`
- Modify: `lib/core/mode_contract.dart`, `lib/models/mode_status.dart`, `lib/providers/supervisor_provider.dart`
- Test: Create `test/services/mic_check_controller_test.dart` (가짜 채널은 `test/ros/fake_ros_socket.dart` 의 `FakeRosSocket`), `test/models/mode_status_test.dart`

**Interfaces:**
- Consumes: Task 9 `MicCheckMessage.parse`, `RosChannelConnector`(`lib/ros/ros_bridge_client.dart:19`)
- Produces (앞의 셋은 Task 12·13 이 쓴다):
  - `kMicCheckStartService = '/vica/mode/mic_check_start'` · `kMicCheckStopService = '/vica/mode/mic_check_stop'` · `kMicCheckPort = 8770`
  - `ModeStatus.micCheck: String`(`json['mic_check']['state']`, 없으면 `'none'`) · `ModeStatus.micCheckActive`(`active`·`activating`·`deactivating`)
  - `SupervisorProvider.startMicCheck()`·`stopMicCheck() → Future<(bool, String)>`(`_callModeService(…, '마이크 점검 시작'|'마이크 점검 끝내기')`)
  - `class MicCheckController extends ChangeNotifier`
  - `MicCheckController({required Uri uri, RosChannelConnector? connector, Duration retry = const Duration(seconds: 2), Duration ping = const Duration(seconds: 10)})`
  - `static Uri uriFor(String rosBridgeUrl, {int port = kMicCheckPort})` — 같은 host, `ws://<host>:<port>/ws`
  - `void start()` · `dispose()`
  - 상태: `bool connected` · `String? prep` · `MicCheckStatus? status` · `MicCheckFrame? frame` · `List<double> rmsHistory` · `List<double> wakeHistory`(각 최대 `kMicHistory = 125`)
    · `List<MicCheckEvent> events`(새것 먼저, 최대 50) · `MicCheckResults? results` · `MicCheckSavePreview? savePreview` · `String? busy` · `Stream<String> toasts`
  - `bool send(String cmd, [Map<String, Object?> args = const {}])` — 끊겨 있으면 보내지 않고 `false`
  - `const kMicHistory = 125`(80 ms × 125 = 10 s, 페이지 `HIST`)

- [ ] **Step 1: 실패하는 시험**
  - `mode_status_test`: `mic_check` 없음 → `micCheck == 'none'`·`micCheckActive` 거짓, `{'state':'deactivating','port':8770}` → 참
  - `uriFor('ws://192.168.0.10:9090') == Uri.parse('ws://192.168.0.10:8770/ws')`
  - `status` 메시지를 받으면 `status` 가 채워지고 `prep` 이 지워짐 · `frame` 130개 → `rmsHistory.length == 125`
  - 채널이 닫히면 `connected` 거짓 → `fake_async` 로 2 s 뒤 다시 붙음(connector 호출 2회)
  - 끊긴 동안 `send('record')` → `false` · 붙어 있으면 `{"cmd":"record"}` 를 보냄 · `send('set', {'key': 'band_min', 'value': 0.004})` → `{"cmd":"set","key":"band_min","value":0.004}`
  - 10 s 마다 `{"cmd":"ping"}` · `toast` 메시지는 `toasts` 스트림으로
  - `frame.busy` 가 null 이 되면 `busy` 도 null
- [ ] **Step 2:** `flutter test test/services/mic_check_controller_test.dart test/models/mode_status_test.dart` → FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4:** PASS
- [ ] **Step 5: 커밋** `feat(app): 마이크 점검기 연결 — 다시 붙기 2 s·ping 10 s·10초 기록`

### Task 11: 점검 화면 ① — 위쪽·단계·실시간 (보드 107·108)

**Files:**
- Create: `lib/screens/mic_check_screen.dart`, `lib/widgets/mic_check/step_card.dart`, `lib/widgets/mic_check/live_card.dart`
  (막대 3개·잡음 바닥), `lib/widgets/mic_check/recent_graph.dart`(CustomPainter), `lib/widgets/mic_check/event_list.dart`, `lib/widgets/mic_check/progress_card.dart`
- Test: Create `test/screens/mic_check_screen_test.dart`

**Interfaces:**
- Consumes: Task 10 `MicCheckController`
- Produces: `MicCheckScreen({required MicCheckController controller})`, 위젯 키 `ValueKey('mic_check_record')`·`('mic_check_redo')`·`('mic_check_robot')`·`('mic_check_finish')`

페이지에서 옮길 계산(정본 `git -C ~/wt-mic-check-voice show 9e30ba6:tools/mic_check_page.html` 스크립트 5~10·108~170행):
`logPos(v) = clamp((log10(max(v, 1e-4)) + 3) / 3, 0, 1)`(소리 크기 막대), 점수 막대는 0~1 선형, 로봇 말하는 중 "멈춰" 점선 문턱 `0.35`,
문턱선 = `values.listen_min_rms`·`short_answer_min_rms`·`wake_threshold`·`emergency_single`.

- [ ] **Step 1: 실패하는 시험** (가짜 컨트롤러 상태를 넣고 pump)
  - `prep` 만 있으면 `'준비 중'`·`'모델 올리는 중 · 1~2분'`
  - 머리 칩: 장소 `'4층 복도'`·`'● 연결됨'`·`'● 칩 정상'`(chipOk 거짓이면 빨간 경고 `'칩에 값을 쓰지 못해 시작하지 않습니다 · 끝내기 뒤 다시 시작하세요'`)
  - 단계 `3 / 9 · 호출`·`'2/5회'`·할 말 문장 · `'녹음 시작'` 누르면 `send('record')`, `'다시'` 누르면 `send('redo', {'step': 'call'})`
  - `frame.recLeft = 4.2` → `'녹음 중 · 4.2초 남음'`, 버튼 비활성
  - 6단계: `'로봇 말하기'` → `send('robot')`; `step.robot` 참이면 `'로봇 말하는 중 · N초 남음'`(앱이 30 s 에서 1 s 씩 셈 — 점검기는 남은 초를 보내지 않는다)
  - 9단계: `'점검이 끝났습니다'`
  - 섹션 제목 `'실시간'`·`'최근 10초'`·`'실시간 판정'`·`'진행'`, 진행 범례 `'통과'`·`'주의'`·`'실패'`·`'아직'`
  - 끊긴 상태에서 `'녹음 시작'` → 스낵 `'연결이 끊겨 보내지 못했습니다 · 다시 붙은 뒤 누르세요'`
  - 폭 1200 에서는 두 칸(왼쪽 단계·실시간·최근 10초·판정 / 오른쪽 진행·결과표·값 고치기), 폭 390 에서는 한 줄(`VicaBreakpoints.isCompact`)
- [ ] **Step 2:** `flutter test test/screens/mic_check_screen_test.dart` → FAIL
- [ ] **Step 3: 구현** — 뒤로 가기는 `Navigator.pop` 만(점검은 계속). 앱바 제목 `'마이크 점검'`, 오른쪽 `'끝내기'`(Task 12 시트).
- [ ] **Step 4:** PASS · `flutter analyze` → 문제 없음
- [ ] **Step 5: 커밋** `feat(app): 마이크 점검 화면 — 단계·실시간·최근 10초·판정·진행(캔버스 107·108)`

### Task 12: 점검 화면 ② — 결과표·값 고치기·시트·끝 (보드 109·110)

**Files:**
- Create: `lib/widgets/mic_check/result_card.dart`, `lib/widgets/mic_check/values_card.dart`, `lib/widgets/mic_check/sheets.dart`
- Modify: `lib/screens/mic_check_screen.dart`
- Test: `test/screens/mic_check_screen_test.dart`

**Interfaces:**
- Consumes: Task 10 `send`·`savePreview`·`results`·`toasts`·`ModeStatus.micCheck`(화면이 `SupervisorProvider.modeStatus` 를 본다)·`stopMicCheck()`
- Produces: `Future<void> showMicFinishSheet(BuildContext, MicCheckController)` 등 시트 함수(이름은 구현자), 끝나면 `Navigator.pop(context, true)`

- [ ] **Step 1: 실패하는 시험**
  - 결과표: 머리 칩에 `results.title`(예 `'사용 가능 · 확인 필요'`), `stale` 참이면 `'값이 바뀌었습니다 · 다시 채점하세요'`, 줄마다 이름·detail·todo, `first` 가 있으면 `'처음 값 결과 보기'`
  - 값 고치기: 범위 6개(이름은 `status.names`, 키는 `status.envKeys`, `.env` 값과 다르면 `'바뀜'`), 슬라이더 놓을 때 `send('set', {'key': k, 'value': v})`,
    AGC 목표·AGC 최대 증폭(`×2`·`×4`·`×8`·`×31.6`)·에코 지우기(`센`·`중간`·`공장값`) 단추도 `set`
  - `'다시 채점'` → `send('rejudge')`, `'저장'` → `send('save_preview')` → `save_preview` 메시지를 받으면 시트 `'.env 에 이렇게 씁니다'`·줄들·
    `'백업 뒤 저장 · 음성 노드를 다시 켜야 적용'` → `'저장'` → `send('save')`
  - `'원래대로'` → 시트 `'가장 최근 백업으로 되돌릴까요?'`·`'마이크 값 10개만 · 음성 노드를 다시 켜야 적용'` → `'되돌리기'` → `send('restore')`
  - `'끝내기'` → 시트 `'점검을 끝낼까요?'`·`'칩을 .env 값으로 되돌리고 끝냅니다 · 기록은 젯슨에 남음'` → `'끝내기'` → `send('finish')` 후 화면이 `true` 로 닫힘.
    보내지 못하면(끊김) `stopMicCheck()` 를 대신 부름.
  - `busy` 가 있으면 `'다시 채점하는 중'` 덮개 · `toasts` → 스낵
  - 끊김 칩 `'● 끊김 · 다시 붙는 중'`
  - **Review Focus 1:** 화면이 열린 채 `modeStatus.micCheck` 가 `activating` → `inactive` 로 바뀌면(결과 없이) 스낵 `'점검기가 끝났습니다'` 뒤 화면이 닫힘
- [ ] **Step 2:** `flutter test test/screens/mic_check_screen_test.dart` → FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4:** PASS · `flutter analyze`
- [ ] **Step 5: 커밋** `feat(app): 마이크 점검 화면 — 결과표·값 고치기·저장/되돌리기/끝내기 시트(캔버스 109·110)`

### Task 13: 들어가는 곳 — 주행 설정 칸·모드 선택·시작 확인 (보드 104 ②④·105·106)

**Files:**
- Modify: `lib/widgets/drive_nodes_card.dart`, `lib/screens/mode_select_screen.dart`
- Create: `lib/widgets/mic_check_entry.dart` (두 화면이 같이 쓰는 줄·시작 확인·화면 열기)
- Test: `test/widgets/drive_nodes_card_test.dart`, `test/screens/mode_select_screen_test.dart`

**Interfaces:**
- Consumes: Task 10 의 상수·`ModeStatus.micCheck(Active)`·`startMicCheck()`·`MicCheckController.uriFor`, Task 11 `MicCheckScreen`
- Produces: `Future<void> openMicCheck(BuildContext context)` (컨트롤러를 만들고 화면을 띄우고, `true` 로 닫히면 끝 스낵)

- [ ] **Step 1: 실패하는 시험**
  - 주행 설정 칸: 음성 켜짐 → `'마이크 점검'` 줄 상태 `'음성을 끄면 시작'`·`'점검'` 비활성 · 음성 꺼짐 → `'약 12분 · 로봇 정지'`·`'점검'` 활성 ·
    점검 중 → `'● 점검 중'`·`'이어 하기'`, 음성 노드 스위치 비활성·상태 `'점검 중 · 켤 수 없음'`
  - `'점검'` → 확인 `'마이크 점검을 시작할까요?'`·`'약 12분 · 손잡이 자리에서 진행'`(+ 주행 켜짐·감지 켜짐일 때만 `'카메라 감지도 끕니다'`) ·
    `'시작'` → `startMicCheck()` → 받으면 문구 스낵 + 점검 화면, 거절이면 `'안 됐습니다: '` + 문구
  - 모드 선택(주행 꺼짐): 카드 두 장 아래 `'마이크 점검'` 줄, 점검 중이면 주행 카드 상태 `'마이크 점검 중'`·누름 막힘, 줄은 `'이어 하기'`
  - 점검 화면이 `true` 로 닫히면: 주행 켜짐 → `'점검을 끝냈습니다 · 음성 노드를 켜야 새 값이 적용됩니다'`, 꺼짐 → `'점검을 끝냈습니다'`
- [ ] **Step 2:** 두 시험 파일 → FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4:** `flutter test` (전체) → PASS · `flutter analyze` → 문제 없음
- [ ] **Step 5: 커밋** `feat(app): 마이크 점검 들어가는 곳 — 주행 설정 칸·모드 선택·시작 확인(캔버스 104~106)`

### Task 14: 2·3단계 운영 반영 · 실기 (사용자 승인 뒤)

- [ ] **Step 1:** 사용자 승인: 루트 `bin` 반영(`mode_logic.py`·`vica_mode_node.py`·`vica_shutdown.sh`) + `vica-mode` 재시작, 음성 `feat/app-mic-check` →
  운영 `after_feedback_waiting` 합치기, 앱 합치기 + 웹 5000 빌드. 주행 묶음 꺼짐 확인.
- [ ] **Step 2:** `bin` 백업(`~/vica_data/autostart_bin_backup/2026-11-19_before_mic_check`) 뒤 복사·재시작, 음성·앱 `git merge --no-ff`, `flutter build web --release`.
- [ ] **Step 3: 실기(주행 꺼짐)** — 모드 선택 → 마이크 점검 → 준비 1~2분 → 2~8단계 → 결과표 → 값 하나 바꿔 다시 채점 → 저장 → 원래대로 → 끝내기.
  확인: `systemctl --user is-active vica-miccheck` 가 끝나면 `inactive`, `~/vica_data/mic_checks/<시각>_<지도>/result.json` 생김, 머리 `'사용 가능…'`.
- [ ] **Step 4: 실기(주행 켜짐)** — 음성 노드 끔 → 점검 시작(감지 꺼지는지) → 점검 중 주행 켜기·음성 켜기·지도 바꾸기 거절 문구 → 끝내기 → 음성 켬.
- [ ] **Step 5:** 터미널에서 `.venv/bin/python -m tools.mic_venue_check` → `"마이크 점검은 앱에서만 합니다…"`, `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8770/`(점검 중) → `404`.
  결과를 `~/vica_data/field_test_20261119_mic_check/RESULT.md` 에.

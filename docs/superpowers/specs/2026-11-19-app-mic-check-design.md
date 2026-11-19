# 앱 마이크 점검 · 음성 노드 켜기/끄기 설계 (2026-11-19)

세 저장소에 걸친다: 루트(모드 노드) · `vica-voice-llm`(점검기) · `VICA_Supervisor`(앱).
원래 점검기 설계: `docs/superpowers/specs/2026-10-12-mic-venue-check-design.md`, 사용법: `docs/mic-venue-check.md`.

## 1. 목적

사용자 요청(11-19, 사용자 표현 그대로):

1. "앱/웹에 새로운 기능을 추가 … 음성에서 하는 마이크 테스트를 앱에서 할 수 있게. 그를 위해서는 음성 노드만 따로
   껐다켰다 하는 기능도 추가되어야" — "앱에서 토글스위치 같은 걸로 음성노드만 껐다켰다".
2. "마이크테스트도 당연히 앱에 있어야죠" · 화면은 3번(Flutter 로 새로 만들기) · "이제 점검은 오직 앱을 통해서만".

## 2. 목표 · 비목표 · 실패 조건

**목표**

- G1. 대시보드 '주행 설정' 카드에 음성 노드(⑫ `vica-llm.service`) 스위치.
- G2. 마이크 점검 시작 버튼 두 곳: 주행 켜짐 → 대시보드 '주행 설정' 카드(음성 꺼짐·정지일 때만), 주행 꺼짐 → 모드 선택 화면.
- G3. 점검 화면을 Flutter 로 새로 만든다(웹 5000·APK 같은 코드). 지금 휴대폰 페이지가 하는 일을 전부 한다(5.3).
- G4. 점검은 앱으로만: 모드 노드가 띄운 점검기만 점검을 한다. 휴대폰 페이지(`mic_check_page.html`)는 지운다.
- G5. 점검기와 음성 노드가 마이크·스피커·칩을 동시에 잡지 않게 모드 노드가 막는다.

**비목표**

- 채점 규칙·단계(2~9)·`.env` 마이크 키 10개·판정 기준은 바꾸지 않는다(`mic_check_rules.py`·`mic_check_session.py` 판정 로직 그대로).
- 점검기 ↔ 화면 메시지 형식은 바꾸지 않는다(앱이 지금 페이지 자리를 대신한다). 필요하면 덧붙이기만.
- rosbridge 를 거치지 않는다(실시간 값 80 ms 마다 — 9090 에 싣지 않는다).
- 점검 중 앱의 안내·배송·경비 시작은 막지 않는다(같은 관리자가 하는 일).

**실패 조건(돌아가도 실패)**

- 점검 중 `vica-llm` 이 떠서 마이크를 두 프로세스가 잡는다.
- 점검이 끝났는데 칩(AGC·에코)이 점검 중 값으로 남는다.
- 로봇이 움직이는 중에 음성이 꺼진다(호출어·'멈춰' 음성 입력이 사라진다).
- 화면을 옮기면서 결과표·판정이 지금 페이지와 달라진다.

## 3. 지금 (11-19 조사 사실)

- 점검기 `tools/mic_venue_check.py` + 중계 `tools/mic_check_web.py`(`LiveServer`, 포트 8770, `GET /` 페이지 + `/ws` WebSocket)
  + 페이지 `tools/mic_check_page.html`(551줄). 세션 `src/mic_check_session.py`(842줄).
- 서버 → 화면 8종: `prep`·`status`·`results`(keep — 새로 붙는 화면에 먼저 보냄), `frame`(오디오 조각 80 ms 마다),
  `event`·`toast`·`busy`·`save_preview`.
- 화면 → 서버: `record`·`robot`·`redo{step}`·`set{key,value}`·`rejudge`·`save_preview`·`save`·`restore`·`finish`·`ping`.
- 시작 검사 `voice_stack_running`: 명령줄에 `vica_voice.launch.py`·`-m src.ros_` 가 있으면 거절.
- 끝낼 때(`finish`·Ctrl+C·SIGTERM·SIGHUP) `finally` 에서 칩을 `.env` 값으로 되돌린다. 단 SIGTERM 처리기는 모델을 다 올린 뒤
  (`_install_signal_handlers`) 걸린다 — 그 전에는 칩이 이미 `.env` 값이라 되돌릴 것이 없다.
- 음성 유닛 `vica-llm.service`: `PartOf=vica-drive.target`, `Restart=no`. 주행 묶음을 켤 때만 뜬다.
- 모드 노드(배포 `~/.config/vica-terminator/bin` == 루트 `feat/autostart` `~/wt-autostart`):
  `detector_on|off`(감지기 유닛 start/stop — 같은 모양의 선례), `voice_online|offline`(음성 LLM 고르기, 떠 있을 때만 재시작),
  `map_request`(떠 있는 유닛만 재시작 — `map_restart_units`), `shutdown`(`systemd-run --user` 일회성 유닛 선례).
  상태 `/vica/mode_status` 에 `voice` = 음성 유닛 상태.
- 앱 `after_feedback_waiting`: `drive_nodes_card.dart`(주행 켜짐 때만 그림 — 카메라 감지·빔포밍 회전·음성 LLM),
  `mode_select_screen.dart`, `mode_contract.dart`, `SupervisorProvider._callModeService`, `ModeStatus.voice`.
  젯슨 주소는 `AppSettings.rosBridgeUrl`(`ws://<host>:9090`).

## 4. 방식 (사용자가 3번을 고름, 11-19)

| 안 | 내용 | 얻는 것 | 잃는 것 |
|---|---|---|---|
| 1 | 앱은 시작 버튼만, 페이지는 새 탭·브라우저 | 앱 작업 최소 | 점검 화면이 앱 밖, APK 는 크롬으로 넘어감 |
| 2 | 지금 페이지를 앱 안에 끼움(iframe·`webview_flutter`) | 검증된 화면 그대로, 점검기 한 곳만 고침 | 앱 스타일과 다름, 새 의존성 1개 |
| **3** | **Flutter 로 새로 만들고 `/ws` 에 직접 붙음** | **앱 디자인·문구 규칙 그대로** | **화면 전부 다시 만듦** |

3번에 "점검은 앱으로만"을 더해 페이지를 지운다 — 화면 두 벌을 같이 고칠 일이 없어진다.

## 5. 설계

### 5.1 모드 노드 (루트 `scripts/autostart/vica_mode_node.py` · `mode_logic.py`)

새 서비스(모두 `std_srvs/Trigger`, `systemctl --no-block` 으로 바로 답):

| 서비스 | 하는 일 | 거절 |
|---|---|---|
| `/vica/mode/voice_on` | `start vica-llm.service` | 주행 모드 아님 · 움직이는 중 · 마이크 점검 중 |
| `/vica/mode/voice_off` | `stop vica-llm.service` | 주행 모드 아님 · 움직이는 중 |
| `/vica/mode/mic_check_start` | `systemd-run --user --unit=vica-miccheck …` 로 점검기 실행 | 이미 점검 중 · 움직이는 중 · 지도 그리기 중 · 음성 유닛 active/activating · 칸(터미널)에서 띄운 음성 프로세스 있음 |
| `/vica/mode/mic_check_stop` | `stop vica-miccheck` (SIGTERM → 점검기가 칩 되돌리고 끝남) | 점검 중 아님 |

- '움직이는 중' = 미션 `is_moving` 참 **또는** 최근 1 s 안 `/cmd_vel_safe` 가 0 이 아님. 기존 `robot_moving`(미션
  `is_moving` 만)은 사람 접근(APPROACHING·TURNING·RETURNING)·앱 수동 주행·도킹을 정지로 본다(`mission_manager_node.py:2461`).
  이 기준은 새 서비스 4개에만 쓴다 — 기존 거절(음성 LLM 고르기·주행 끄기 등)은 이번에 바꾸지 않는다.
- 점검을 시작할 때 주행이 켜져 있고 카메라 감지(⑮ `vica-detector.service`)가 켜져 있으면 같이 끈다(기존 `detector_off` 와
  같은 `stop`). 미션의 사람 접근은 음성이 살아 있는지 보지 않아(`mission_logic.py:3868` 앞 관문) 감지가 켜져 있으면
  점검 중에 말없이 다가간다(사용자 11-19 ① 결정). 음성 스위치만 끌 때는 감지를 건드리지 않는다.
- 점검기 명령: `cd ~/VICA-smarthandle/vica-voice-llm && .venv/bin/python -m tools.mic_venue_check --name <CURRENT_MAP id>
  --destinations ~/vica_data/destinations/<id>/destinations.yaml`(파일이 없으면 빼고).
- 점검 중 추가로 거절: `drive_start`(주행 묶음이 `vica-llm` 을 띄운다), `voice_on`, 지도 바꾸기(`map_request`).
  음성 LLM 고르기(`voice_online|offline`)는 음성이 꺼져 있으면 파일만 바꾸므로 그대로 둔다.
- `/vica/mode_status` 에 `mic_check` 칸: 유닛 상태(`none`·`activating`·`active`·`deactivating`·`failed`)와 포트(8770).
  거절 문구는 지금처럼 앱이 그대로 보여 준다("마이크 점검 중입니다. 앱에서 끝낸 뒤 …").
- 로봇 끄기(`shutdown`)는 점검 중이어도 받는다 — 끄기 스크립트가 `vica-miccheck` 를 먼저 멈춘다(칩 되돌리기).

### 5.2 점검기 (`vica-voice-llm`)

- `LiveServer`: 페이지 응답을 지우고 `/ws` 만 연다. `GET /` 는 404 + 한 줄("마이크 점검은 앱에서 합니다").
- `tools/mic_check_page.html` 과 `tests/test_mic_check_page.py` 를 지운다.
- 실행 표지: 모드 노드가 띄운 점검기인지 확인한다(`/proc/self/cgroup` 에 `vica-miccheck` 유닛, 또는 모드 노드가 주는 환경 변수 —
  빌더 선택). 아니면 "마이크 점검은 앱에서만 합니다"를 찍고 끝낸다. `--demo`(가짜 소리, 화면 개발용)·`--restore`(`.env` 비상 되돌리기)는 통과.
- 메시지 형식은 그대로. 화면 문구에 남은 '⑫'·'휴대폰'은 '음성 노드'·'앱'으로(`_toast` 문구 등).
- `docs/mic-venue-check.md` 를 앱 기준으로 다시 쓰고, 10-12 설계서 머리에 "11-19 페이지 폐지, 이 문서 참조"를 단다.

### 5.3 앱 (`VICA_Supervisor`)

- `mode_contract.dart`: 서비스 4개 이름 · 점검 포트 8770.
- `SupervisorProvider`: `setVoiceNode(bool)`·`startMicCheck()`·`stopMicCheck()` — 기존 `_callModeService` 로.
- `ModeStatus`: `micCheck` 칸.
- `drive_nodes_card.dart`: '음성 노드' 줄(스위치, 꺼지면 "호출어·'멈춰' 안 됨" 한 줄) + '마이크 점검' 버튼(음성 꺼짐·정지일 때만).
  점검 중이면 버튼이 '이어 하기'.
- `mode_select_screen.dart`: 주행 꺼짐일 때 '마이크 점검' 칸.
- 새 화면 `mic_check_screen.dart` + 연결 `mic_check_client.dart`: `ws://<rosBridgeUrl 의 host>:8770/ws`.
  지금 페이지 구성을 그대로 옮긴다:
  1. 머리: 장소 · 연결 · 칩 상태 · 준비 중(`prep`) · 칩 쓰기 실패 경고
  2. 단계 카드: n/9 · 단계 이름 · 할 말 · 몇 번째 · 녹음/로봇 말하기 버튼 · 남은 초 · 다시
  3. 실시간 막대 3개: 소리 크기(문턱선 2개·잡음 바닥) · "비카야" 점수 · "멈춰" 점수(로봇 말하는 중 문턱 점선)
  4. 최근 10초 그래프
  5. 실시간 판정 목록(`event`)
  6. 진행(단계별 시도 통과·실패 점)
  7. 결과표(`results` — 처음 값 비교, 값이 바뀌었으면 '다시 채점' 안내)
  8. 값 고치기(범위 막대·단추·AGC/에코/최대 증폭 고르기) · 다시 채점 · 저장(미리보기 시트 → 확인) · 원래대로(확인 시트)
  9. 끝내기(확인 시트) · 바쁨 표시(`busy`) · 토스트(`toast`)
- 문구는 앱 규칙(제목 하나 · 상태는 점+단어 · 설명 한 줄 · 버튼 2~6자)으로 줄인다. 목업 승인 대상.
- 화면을 나가도 점검은 계속된다(keep 메시지로 다시 붙는다). 대시보드·모드 선택에 '점검 중'과 이어 하기.
- 점검 시작 확인에 "카메라 감지도 끕니다"를 함께 적는다(감지가 켜져 있을 때만).
- 끝내기 → 점검기 종료 → 화면 닫힘. 주행 켜짐이면 "음성 노드를 켜야 새 값이 적용됩니다" 스낵.
  음성·카메라 감지는 자동으로 켜지 않는다(주행 설정 카드 스위치로 켠다).
- 연결이 끊기면 다시 붙는다(`ping` 주기는 빌더).

### 5.4 순서 · 배포 · 시험

1. 음성 노드 스위치(모드 노드 + 앱) — 작고 따로 쓸 수 있어 먼저. **운영에 따로 먼저 반영한다.**
2. 점검기 실행 표지·페이지 폐지 + 모드 노드 점검 서비스·거절.
3. Flutter 점검 화면.
   **2·3 은 한 번에 운영에 반영한다** — 따로 내면 그 사이 점검할 방법이 없다.

- 브랜치(사용자 11-19 승인): 음성·앱은 `after_feedback_waiting` 에서 새 `feat/app-mic-check`(worktree),
  루트(모드 노드)는 기존 `feat/autostart`(`~/wt-autostart`)에 커밋. 운영 합치기는 단계 반영 때 승인받는다.

- 목업: "앱 UI 사전확인" 캔버스(`3gzpfMHhQZ2A9wkYG59ewU`)에 보드로 추가, 승인 뒤 구현.
- 배포: 모드 노드는 `bin` 반영 + `vica-mode` 재시작(백업), 웹 5000 빌드는 주행 묶음이 꺼졌을 때(11-17 사고), APK 는 노트북.
- 시험: `mode_logic` 단위 시험(거절 표), 점검기 실행 표지 시험, 앱 위젯 시험(가짜 WebSocket 메시지 8종),
  화면 실기는 `--demo` → 실제 마이크로 한 번 끝까지.

## 6. 사용자 결정 (11-19)

- 화면 방식 3번(Flutter 새로).
- 점검은 앱으로만.
- 시작 위치 두 곳(대시보드 '주행 설정' 카드 · 모드 선택 화면) — 추천안 승인.
- 음성 노드 스위치 위치·동작은 제안에 이견 없음.
- 점검 시작 때 카메라 감지도 같이 끈다(① 추천안).
- 움직이는 중 음성 켜기 거절(STT·LLM 모델 올리는 부하를 주행 중에 걸지 않는다).
- 브랜치·반영 순서 추천안 승인(5.4).

## 7. 내 판단 (확인 받을 것)

- 기록 이름(`--name`) = 현재 지도 id 자동 — 장소 이름 입력 칸을 만들지 않는다.
- '움직이는 중' = 미션 `is_moving` 또는 최근 1 s `/cmd_vel_safe` 0 아님(5.1) — 움직이는 중 음성 끄기도 거절.
- 점검이 끝나도 음성 노드·카메라 감지를 자동으로 켜지 않는다(다시 점검할 수 있게).
- `--demo`·`--restore` CLI 는 남긴다.
- 점검 중 주행 켜기·음성 켜기·지도 바꾸기 거절, 안내 시작은 막지 않는다.
- 지도 그리기 중 점검 거절.

## 8. 빌더에게 맡김

- `mode_status.mic_check` 정확한 모양, 실행 표지 방식(cgroup·환경 변수), 재연결·`ping` 주기, 위젯 나누기, 그래프 그리는 방식.

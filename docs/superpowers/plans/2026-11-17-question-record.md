# 질문 기록 발행·소비 (설계 7절의 2단계) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 미션이 대답을 기다리는 말을 낼 때마다 질문 기록(`/vica/question`)을 내고, 음성 의도 노드가 그것을 받아 로그로만 남긴다. 로봇의 말·동작은 바뀌지 않는다.

**Architecture:** 미션의 `Say` 에 질문 종류(`kind`)와 후보(`options`)를 붙이고, 순수 파이썬 `QuestionTracker` 가 열림·닫힘과 id 를 관리한다. 노드는 `Say.expects_reply` 를 내보내는 한 자리(`_run_actions`)에서 기록을 발행한다. 음성은 구독해서 발화마다 한 줄 로그를 남기고, `tools/answer_audit` 가 그 줄을 읽어 실기 뒤 표로 만든다.

**Tech Stack:** ROS 2 Humble(rclpy, rosidl), Python 3.10, pytest. 저장소 둘: `vica_ros2_ws`(`vica_interfaces`, `vica_mission_manager`), `vica-voice-llm`.

**Spec:** `docs/superpowers/specs/2026-11-14-question-aware-answers-design.md` — 4.1절(질문 기록), 7절 2번. 기준 점수는 `~/vica_data/voice_answer_audit_20261113/README.md` "확정 기준 점수(11-17)".

## Global Constraints

- **동작 변화 없음.** 로봇이 하는 말, 상태 전이, 기존 토픽의 내용은 그대로다. 기존 시험은 고치지 않고 통과해야 한다(작업 전부터 실패하던 것은 Task 1 첫 단계에서 목록으로 적어 둔다).
- 새 ROS 노드 없음. 새 멘트 없음. 긴급 경로(`emergency_filter`, `/vica/emergency`)는 손대지 않는다.
- 브랜치는 두 저장소 모두 `after_feedback_waiting`. 새 브랜치를 만들지 않는다. 푸시와 dev 머지는 하지 않는다.
- ROS 저장소에는 다른 작업의 미커밋 파일(`maps/` 등)이 있다. `git add` 는 이 계획의 파일만 이름으로 지정한다.
- 음성 노드는 `VicaQuestion` 이 없는 옛 `vica_interfaces` 설치본에서도 떠야 한다(구독을 건너뛰고 로그 한 줄).
- 로봇 스택은 systemd 사용자 유닛으로 돈다. `colcon build` 와 유닛 재시작은 사용자에게 확인받고, 주행 중에는 하지 않는다.
- 질문 종류는 여섯 개뿐이다: `open` · `confirm` · `choice` · `yesno` · `yesno_inverted` · `minutes`.
- 메시지의 `step` 과 `on_silence` 는 칸만 둔다. 이 단계에서는 항상 `0` 과 `""` 다(되묻기는 4단계).
- 주석은 한 줄 한 뜻, 한국어, 기존 문체.

## Review Focus

1. 음성 노드가 미션보다 먼저 뜨거나 미션만 재시작된 경우 — 기록이 없어도 음성은 평소대로 해석하고 `[질문] 없음` 을 남긴다.
2. 옛 `vica_interfaces` 설치본(`VicaQuestion` 없음) — 음성 노드가 죽지 않는다.
3. 되묻기로 같은 문장을 바로 다시 묻는 경우 — id 가 새로 나오고, 닫힘이 열림보다 먼저 발행된다.
4. `kind` 없이 `expects_reply=True` 인 `Say` 가 새로 생긴 경우 — 시험이 잡고, 실행 중에는 `open` 으로 내며 경고 로그를 남긴다.
5. 아무도 대답하지 않고 상태도 안 바뀌는 질문("네?" 뒤 침묵) — 음성이 듣기를 빈손으로 끝냈다는 신호(`/vica/listen_state` 의 `empty…`)로 닫힌다. 음성에서 아무 신호도 안 올 때만 마지막 신호 뒤 45초에 닫고 경고로 남긴다(사유 `no_signal`).

---

### Task 1: `VicaQuestion.msg`

**Files:**
- Create: `vica_ros2_ws/src/vica_interfaces/msg/VicaQuestion.msg`
- Modify: `vica_ros2_ws/src/vica_interfaces/CMakeLists.txt`(`"msg/PersonDetection.msg"` 다음 줄)

**Interfaces:**
- Produces: `vica_interfaces/msg/VicaQuestion` — 칸 `string id` · `string kind` · `string text` · `string[] option_ids` · `string[] option_names` · `uint8 step` · `string on_silence` · `builtin_interfaces/Time opened_at`. `id == ""` 는 "열린 질문 없음"(닫힘 알림)이다.

- [ ] **Step 1: 기준 실패 목록** — `vica_mission_manager` 와 음성 저장소의 전체 시험을 한 번 돌려 지금 실패하는 시험 이름을 진행 기록에 적는다(11-14 기준 미션 4개: `test_progress_narration` 3, `test_tip_over` 1).
- [ ] **Step 2: 메시지 파일 작성** — 위 칸 그대로, 칸마다 한 줄 주석. `CMakeLists.txt` 에 등록.
- [ ] **Step 3: 빌드(사용자 확인 뒤)** — `colcon build --packages-select vica_interfaces`. Expected: `Summary: 1 package finished`.
- [ ] **Step 4: 확인** — ROS 환경을 source 한 셸에서 `python3 -c "from vica_interfaces.msg import VicaQuestion as Q; print(Q.get_fields_and_field_types())"`. Expected: 위 8칸.
- [ ] **Step 5: Commit** — `feat(interfaces): 질문 기록 VicaQuestion(설계 11-14 4.1)`

### Task 2: 미션 — 질문마다 종류와 후보 표기

**Files:**
- Modify: `vica_ros2_ws/src/vica_mission_manager/vica_mission_manager/mission_logic.py`(`Say`, `_ask`, 질문을 내는 42곳)
- Test: `vica_ros2_ws/src/vica_mission_manager/test/test_question_kind.py`

**Interfaces:**
- Produces: `Say(text, priority="narration", expects_reply=False, kind: str = "", options: tuple[tuple[str, str], ...] = ())` — `options` 는 `(목적지 id, 이름)` 묶음. `MissionLogic._ask(text: str, kind: str, options: tuple = ()) -> Say`. 상수 `QUESTION_KINDS = ("open", "confirm", "choice", "yesno", "yesno_inverted", "minutes")`.

문구별 종류(설계 4.1 표):

| 종류 | 문구 |
|---|---|
| `open` | `MSG_WAKE_GREETING` · `MSG_WAIT_FINISH_ASK` · `MSG_APPROACH_ONBOARDING` · `MSG_DEST_RETRY` · `MSG_ARRIVAL_RETRY` · `MSG_SPK_REASK` |
| `confirm` | `_confirm_prompt`(목적지 확인) · `MSG_CONFIRM_SWITCH` — `options` = 묻는 목적지 하나 |
| `yesno` | `MSG_ASK_RESTROOM` · `MSG_ASK_GENERIC` · `MSG_ASK_ENTRANCE`(도착 안내 문장이 앞에 붙은 것 포함) · `MSG_APPROACH_QUESTION` · `MSG_APPROACH_REASK` · `MSG_CANCEL_CONFIRM` · `MSG_HELP_CONFIRM` · `MSG_HELP_CANCEL_CONFIRM` · `MSG_SPK_IDENTITY` |
| `yesno_inverted` | `MSG_WAIT_NEED_ASK` |
| `minutes` | `MSG_ASK_WAIT_TIME` |

`_asking_question` 을 다시 묻는 자리는 처음 물을 때의 종류를 같이 보관해 그대로 쓴다. `choice` 는 지금 미션이 말하는 문구가 없다(음성이 짓는다 — 3단계에서 옮긴다).

- [ ] **Step 1: 실패하는 시험 작성**

```python
def test_every_question_site_declares_a_kind():
    """expects_reply=True 인 Say(…)·_ask(…) 호출은 모두 kind 를 넘긴다(소스 AST 검사)."""
    missing = sites_without_kind(Path(mission_logic.__file__).read_text())
    assert missing == []          # [(줄 번호, 소스 한 줄)]

@pytest.mark.parametrize("text,kind", [
    (MSG_WAKE_GREETING, "open"), (MSG_WAIT_NEED_ASK, "yesno_inverted"),
    (MSG_ASK_WAIT_TIME, "minutes"), (MSG_CANCEL_CONFIRM, "yesno"),
])
def test_flows_emit_the_declared_kind(text, kind): ...   # 그 문구를 내는 흐름을 돌려 Say.kind 를 본다

def test_confirm_question_carries_the_destination_as_option():
    ...  # navigate 제안 → Say.kind == "confirm" and Say.options == ((dest.id, dest.name),)

def test_kind_is_one_of_the_six():
    ...  # 위 흐름들에서 나온 모든 질문 Say 의 kind 가 QUESTION_KINDS 안
```

- [ ] **Step 2: 실패 확인** — `python3 -m pytest test/test_question_kind.py -q`. Expected: FAIL(`Say` 에 `kind` 없음).
- [ ] **Step 3: 구현** — `Say` 에 두 칸, `_ask` 에 `kind`·`options`, 42곳에 표대로 표기. `sites_without_kind(source: str) -> list[tuple[int, str]]` 는 시험 파일 안 도우미(`ast` 로 `Say`/`_ask` 호출 검사).
- [ ] **Step 4: 통과 확인** — 새 시험 통과 + `python3 -m pytest test -q` 가 Step 1 기준과 같은 실패 목록.
- [ ] **Step 5: Commit** — `feat(mission): 질문 Say 에 종류·후보 표기(동작 변화 없음)`

### Task 3: 미션 — 질문 기록의 열림·닫힘

**Files:**
- Create: `vica_ros2_ws/src/vica_mission_manager/vica_mission_manager/question_record.py`
- Test: `vica_ros2_ws/src/vica_mission_manager/test/test_question_record.py`

**Interfaces:**
- Consumes: Task 2 의 `Say.kind`·`Say.options`.
- Produces:
  - `QuestionRecord(id: str, kind: str, text: str, options: tuple[tuple[str, str], ...], step: int, on_silence: str, opened_at: float, state: str)` (frozen dataclass)
  - `QuestionTracker(prefix: str, no_signal_close_sec: float = 45.0)` — `current: Optional[QuestionRecord]`
  - `open(say, state: str, now: float) -> tuple[Optional[QuestionRecord], QuestionRecord]` — (닫힌 옛 기록, 새 기록). id 는 `f"{prefix}-{n}"`, n 은 1부터.
  - `close(reason: str) -> Optional[QuestionRecord]` — 사유 `answer` · `silence` · `state` · `replaced` · `no_signal`
  - `on_listen_state(state: str, now: float) -> Optional[QuestionRecord]` — 음성의 듣기 결과(올 때마다 마지막 신호 시각을 갱신). `empty` 로 시작하면(말이 없었거나 버려짐) 닫고 사유 `silence`. `open`·`speech`·`closed` 는 그대로 둔다(말을 했으면 의도가 뒤따라 와서 `answer` 로 닫힌다).
  - `tick(state: str, now: float) -> Optional[tuple[QuestionRecord, str]]` — 상태가 열 때와 달라졌으면 `state`, 음성의 마지막 신호(없으면 연 때) 뒤 `no_signal_close_sec` 지났으면 `no_signal`.

닫는 기준에 시간을 쓰지 않는다(11-17 사용자: 40 s 는 근거 없음). 실기 1,660건에서 질문 재생 시작 → 대답은 중앙 7.2 s · 99 % 17.5 s · 최대 23.6 s 이고,
귀로 확인한 행에서 20 s 뒤에 온 3건은 모두 대답이 아니었다. `no_signal_close_sec` 45 s 는 음성에서 아무 신호도 안 올 때만 쓰는 값이다 — 음성의 마지막 신호부터 센다.
가장 긴 정상 간격이 질문 뒤 듣기 30 s(음성 `CONFIRM_WINDOW_SEC`)라 여유 15 s 를 더했다. 처음에는 연 때부터 30 s 로 셌는데
(6 + 6 + 15 + 3, "비카야" 뒤 듣기 값으로 잘못 계산) 평소 무응답에서 음성의 `empty` 보다 먼저 터졌다 — 11-17 최종 검토에서 고침.

- [ ] **Step 1: 실패하는 시험 작성**

```python
def test_open_gives_new_id_each_time():       # q-1, q-2 · 같은 문장을 다시 물어도 새 id
def test_open_replaces_and_returns_old():     # 두 번째 open 이 첫 기록을 돌려준다
def test_close_returns_record_once():         # 두 번째 close 는 None
def test_tick_closes_on_state_change():       # open(state="confirming") → tick("navigating") = (기록, "state")
def test_empty_listen_state_closes():         # on_listen_state("empty") · ("empty:reject '…'") → 기록, 사유 silence
def test_speech_listen_state_keeps_open():    # "open" · "speech" · "closed" → None, current 그대로
def test_no_signal_close_counts_from_the_question_when_voice_is_silent():   # 신호 없음 · 연 때 + 45.1 s → (기록, "no_signal")
def test_no_signal_close_counts_from_the_last_voice_signal():             # open 신호 뒤 30 s 듣기 중에는 안 닫힘 → empty 로 닫힘
def test_missing_kind_becomes_open():         # Say.kind == "" → 기록 kind "open"
```

- [ ] **Step 2: 실패 확인** — Expected: FAIL(모듈 없음).
- [ ] **Step 3: 구현** — 위 서명대로. rclpy 를 가져오지 않는다.
- [ ] **Step 4: 통과 확인**
- [ ] **Step 5: Commit** — `feat(mission): 질문 기록 열림·닫힘(QuestionTracker)`

### Task 4: 미션 노드 — `/vica/question` 발행과 로그

**Files:**
- Modify: `vica_ros2_ws/src/vica_mission_manager/vica_mission_manager/mission_manager_node.py`(`_run_actions`, `_on_intent`, `/vica/listen_state` 콜백, 주기 타이머, 발행자)
- Test: `vica_ros2_ws/src/vica_mission_manager/test/test_question_record.py`(노드 소스 검사 추가)

**Interfaces:**
- Consumes: `VicaQuestion`(Task 1), `QuestionTracker`(Task 3).
- Produces: 토픽 `/vica/question`(`VicaQuestion`, RELIABLE · TRANSIENT_LOCAL · depth 1). 로그 두 줄(Task 6 이 읽는다, 글자 그대로):
  - `[질문] 열림 id={id} kind={kind} 후보={n} 요청={y|n} 글자={text}` — `요청` 은 듣기 요청(`expects_reply`)을 같이 냈는지. 호출 응답 "네?"는 `n`.
  - `[질문] 닫힘 id={id} 사유={reason} 열린 시간={sec:.1f}s`

발행 순서: `_on_intent` 는 로직이 낸 동작을 실행하기 **전에** `close("answer")` 를 하고 빈 기록을 낸다. `_run_actions` 는 `expects_reply` 인 `Say` 마다 `open` 하고 새 기록을 낸다. `/vica/listen_state` 콜백(이미 구독 중)은 `on_listen_state(state)` 결과가 있으면 빈 기록을 낸다. 주기 타이머는 `tick(logic.state, now)` 결과가 있으면 빈 기록을 내고, 사유가 `no_signal` 이면 경고 수준으로 남긴다. `kind` 가 빈 `Say` 를 만나면 경고 `[질문] kind 없음 — open 으로 냄: {text}`.

- [ ] **Step 1: 실패하는 시험 작성** — 기존 노드 소스 검사 방식(`NODE` 문자열)으로: `"/vica/question"` 발행자 있음 · `TRANSIENT_LOCAL` · `_run_actions` 안에 `open(` · `_on_intent` 안에 `close("answer")` · 듣기 결과 콜백 안에 `on_listen_state(` · 두 로그 문구.
- [ ] **Step 2: 실패 확인**
- [ ] **Step 3: 구현** — 메시지 채우기 함수 `_question_msg(record: Optional[QuestionRecord]) -> VicaQuestion`(None 이면 빈 기록).
- [ ] **Step 4: 통과 확인** — 새 시험 + 전체 시험이 기준과 같음.
- [ ] **Step 5: 빌드(사용자 확인 뒤)** — `colcon build --packages-select vica_mission_manager`.
- [ ] **Step 6: Commit** — `feat(mission): 질문 기록 발행 /vica/question(기록만)`

### Task 5: 음성 — 질문 기록 받아 로그로

**Files:**
- Create: `vica-voice-llm/src/question_record.py`
- Modify: `vica-voice-llm/src/ros_node.py`(가져오기 보호, 구독, `_publish_intent` 의 `입력=` 로그 다음 줄)
- Test: `vica-voice-llm/tests/test_question_record.py`

**Interfaces:**
- Consumes: `/vica/question`(Task 4).
- Produces:
  - `OpenQuestion(id: str, kind: str, text: str, option_ids: tuple[str, ...], opened_at: float)` (frozen dataclass)
  - `from_fields(id: str, kind: str, text: str, option_ids, opened_at: float) -> Optional[OpenQuestion]` — `id == ""` 면 None
  - `log_line(q: Optional[OpenQuestion], now: float) -> str` — `[질문] 없음` 또는 `[질문] id={id} kind={kind} 나이={sec:.1f}s`

- [ ] **Step 1: 실패하는 시험 작성**

```python
def test_empty_id_means_no_question():        # from_fields("", …) is None
def test_log_line_without_question():         # "[질문] 없음"
def test_log_line_with_question():            # "[질문] id=q-3 kind=yesno 나이=4.2s"
def test_node_guards_missing_interface():     # ros_node 소스: VicaQuestion 가져오기가 try/except ImportError 안
def test_node_logs_question_after_input():    # ros_node 소스: "입력=" 로그 뒤에 log_line 호출
```

- [ ] **Step 2: 실패 확인**
- [ ] **Step 3: 구현** — 구독 QoS 는 발행과 같게(RELIABLE · TRANSIENT_LOCAL · depth 1). 가져오기 실패 시 `질문 기록 없음(옛 vica_interfaces) — 구독 안 함` 한 줄. 해석 결과는 건드리지 않는다.
- [ ] **Step 4: 통과 확인** — `.venv/bin/python -m pytest -q` 가 기준과 같음.
- [ ] **Step 5: Commit** — `feat(voice): 질문 기록 구독 — 발화마다 [질문] 로그(기록만)`

### Task 6: 도구 — 실기 로그에서 질문 기록 읽기

**Files:**
- Modify: `vica-voice-llm/tools/answer_audit.py`(`COLUMNS` 끝에 3칸, `extract_rows`)
- Test: `vica-voice-llm/tests/test_answer_audit.py`

**Interfaces:**
- Consumes: Task 4·5 의 로그 문구.
- Produces: `COLUMNS` 끝에 `q_id` · `q_kind` · `q_age_s`. `q_id` 는 그 발화 뒤 1 s 안의 음성 `[질문]` 줄에서, 없으면 `""`. 요약 출력에 `질문 기록 있음 N · 종류 같음 N · 종류 다름 N · 기록 없음 N`(같음 = `q_kind == kind`, `kind` 는 지금처럼 글자로 본 값).

- [ ] **Step 1: 실패하는 시험 작성**

```python
def test_row_takes_question_id_from_voice_line():   # 입력= 줄 뒤 "[질문] id=q-2 kind=confirm 나이=3.0s" → q_id "q-2", q_kind "confirm", q_age_s "3.0"
def test_row_without_record_has_empty_fields():     # "[질문] 없음" → 세 칸 ""
def test_old_logs_without_question_lines_still_parse():   # 11-14 이전 로그 모양 → 행 수 그대로, 세 칸 ""
```

- [ ] **Step 2: 실패 확인**
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — 전체 시험 + 옛 로그로 `--since 2026-10-10` 을 돌려 행 수가 전과 같음(1,660행 근처, 새 칸만 빈칸).
- [ ] **Step 5: Commit** — `feat(tools): answer_audit 가 질문 기록 로그를 읽음`

---

## 단계 끝 실기 확인 (사용자와 함께)

빌드와 유닛 재시작 뒤, 짧게 한 번 돈다: "비카야" → 목적지 → "그래" → 도착 질문 → "30분" → 대기 중 "비카야".

| 볼 것 | 기준 |
|---|---|
| 미션 로그 `질문 발화 — 재청취 요청 발행` 줄 수 | `[질문] 열림 … 요청=y` 줄 수와 같다(호출 응답 "네?"는 `요청=n` 이라 뺀다 — 최종 검토 반영) |
| 열린 질문마다 닫힘 | `[질문] 닫힘` 이 짝으로 있고 사유가 찍힌다 |
| 음성 신호 없이 닫힌 질문(`사유=no_signal`) | 0건(나오면 그 장면을 본다) |
| 음성 로그 | `입력=` 줄마다 `[질문]` 줄이 따른다 |
| `tools.answer_audit` 요약 | 미션이 물은 질문의 대답은 `질문 기록 있음`, 종류 다름 0 |
| 로봇의 말과 동작 | 전과 같다(확정 기준 점수의 재생 결과가 변하지 않는다) |

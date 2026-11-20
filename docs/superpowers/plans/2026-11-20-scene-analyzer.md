# 대화 장면 분석기(A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 지난 시험·운영 로그에서 "설계한 대답이 안 나온 장면"을 골라 원인(확정·추정·미분류)과 함께 브라우저로 보여 주고, 음성 노드가 다음부터 판단 이유를 JSONL 로 남기게 한다.

**Architecture:** 1부는 저장소 안 오프라인 도구 `tools/scene/`(로그 읽기 → 사건 → 장면 → 원인·집계·설계상 대답 → 웹 서버 8772). 운영 노드는 건드리지 않는다. 2부는 음성 노드 3개에 `src/voice_trace.py` 기록기를 붙이고(동작 영향 0, `VICA_TRACE`), 1부가 그 기록을 우선해서 읽는다. 미션 저장소는 고치지 않는다.

**Tech Stack:** Python 3.10(음성 `.venv`), 표준 `http.server`·`json`·`subprocess`(journalctl), pytest, 기존 모듈 `tools/answer_audit.py`·`tools/answer_replay.py`·`src/mission_phrases.py`·`src/destination_loader.py`.

**Spec:** `docs/superpowers/specs/2026-11-20-scene-analyzer-design.md` (11-20 사용자 승인)

## Global Constraints

- 미션 저장소(`vica_ros2_ws`) 수정 금지. 토픽·메시지 형식(`/vica/user_text`·`/vica/listen_state`·`/vica/tts_request` 등) 변경 금지.
- 기록 실패가 듣기·판정·말하기를 멈추면 안 된다 — 기록 경로는 전부 `try/except`, 실패는 경고 한 번.
- `VICA_TRACE` = `on`|`off`, 기본 `on`. 위치 `~/vica_data/voice_trace/<YYYYMMDD>/<node>.jsonl`, 한 줄 `{"t","node","stage","decision","reason","values"}`.
- 빈손 창 문구(그대로): `청취 창 말소리 없음(자유 창|질문 창) — 최대 rms {x:.4f} · 배경 rms {y:.4f}`. 빈손 창 소리는 저장하지 않는다.
- 서버 포트 8772, 기본 주소 127.0.0.1(`--host` 로 바꿈). 캐시 `~/vica_data/scene_viewer/`. 기록 파일을 지우거나 덮어쓰지 않는다.
- LLM 다시 묻기: 사람이 누를 때만 · 그때와 같은 경로(midm·gpt·realtime) · 결과 캐시 · `systemctl --user is-active vica-drive.target` 이 `active` 면 midm 거부.
- 분석기는 `.env` 를 직접 열지 않는다. 해석기 import 전 환경값은 `tools.answer_replay.path_env()` 로 넣는다.
- 판정 숫자: LLM 시간 제한 15 s ±0.2 s · 질문 겹침 2.0 s · 한 동작의 연속 0.1 s 미만 · 확정 직후 같은 말 2.0 s · 배경 rms = 창 열린 뒤 0.5 s.
- 문체: 주석·로그는 기존 한국어 문체, 커밋은 `feat(scene): …`/`feat(voice): …` + `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. 쓰는 중인 로그(마지막 줄이 잘림) — 잘린 줄은 건너뛰고 나머지는 읽는다 → Task 1 `test_truncated_last_line_skipped`.
2. 노드 재시작·같은 줄이 launch 출력과 노드 로그 양쪽에 있음 — 같은 (시각, 노드, 글) 은 한 번만 → Task 1 `test_duplicate_lines_merged`.
3. 옛 형식 줄(`speech=-0.00s`) — 음수 말 길이도 ghost 로 센다(62 의 8건) → Task 1 `test_ghost_negative_speech`.
4. 소리 요청 경로 조작(`../`, 절대 경로) — 거부(403) → Task 6 `test_clip_path_outside_dirs_rejected`.
5. 나흘치(노드 로그 100여 개) 첫 열기 — 캐시로 두 번째부터 즉시. 첫 분석 시간은 Task 4 수용 단계에서 재서 적는다.

---

### Task 1: 로그 읽기와 사건 분류

**Files:**
- Create: `tools/scene/__init__.py`, `tools/scene/events.py`
- Create: `tests/test_scene_events.py`, `tests/fixtures/scene/1104_sample.log`(실제 11-04 노드 로그 줄을 옮긴 것: 11:09:21·11:18:50~11:19:07·11:20:21~11:20:51·15:06:45~15:07:33·10:21~10:23 구간, 미션·TTS·LLM·호출 노드 줄 포함)

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Event(t: float, node: str, kind: str, data: dict, raw: str)` — node ∈ `wake|llm|tts|mis`.
  - `classify(t: float, node_full: str, msg: str) -> Event | None` — kind ∈ `wake_fire, wake_near, wake_self, wake_open, wake_ignored, listen_ghost, listen_reject, listen_empty_free, user_text, listen_quality, listen_timing, spk, llm_input, llm_rt, relisten_drop, forward_answer, q_voice, q_open, q_close, mis_say, mis_req, mis_wake, state, tts_play, usb, llm_fail, startup`.
  - `read_events(since: float, until: float, log_dir: Path = ~/.ros/log, relaunch_dir: Path = ~/vica_data/relaunch, journal: bool = True) -> list[Event]` — 시각순, 중복 제거.
- Consumes: `tools.answer_audit.parse_log_line`, `_INPUT`·`_RT`·`_MSAY`·`_MREQ`·`_QOPEN`·`_QCLOSE`·`_QVOICE`·`_PLAY`.

- [ ] **Step 1: 실패하는 시험 쓰기**

```python
def test_wake_near_and_fire():
    e = classify(1.0, "vica_wakeword_node", "🎯 호출 점수 놓침: 0.83 (문턱 0.70)")
    assert e.kind == "wake_near" and e.data == {"peak": 0.83, "threshold": 0.70}

def test_ghost_negative_speech():
    e = classify(1.0, "vica_wakeword_node", "청취 기각: empty:ghost speech=-0.00s rms=0.0051")
    assert e.kind == "listen_ghost" and e.data["rms"] == 0.0051 and e.data["speech"] <= 0

def test_mission_say_is_mission_source():
    e = classify(1.0, "vica_mission_manager", "TTS[response]: 안내를 취소할까요?")
    assert e.kind == "mis_say" and e.data == {"priority": "response", "text": "안내를 취소할까요?"}

def test_llm_fail_from_stderr():
    e = classify(1.0, "vica_llm", "[ERROR] [LLM] 호출 실패(모델 midm2-mini): timed out")
    assert e.kind == "llm_fail"

def test_truncated_last_line_skipped(tmp_path): ...   # 마지막 줄이 '[INFO] [17921' 로 끝나는 파일 → 예외 없이 앞줄만
def test_duplicate_lines_merged(tmp_path): ...        # 같은 줄이 노드 로그·relaunch 출력 양쪽 → 1번
def test_fixture_counts():                             # 1104_sample.log → wake_fire·relisten_drop·forward_answer·mis_say 개수 고정값
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/python -m pytest tests/test_scene_events.py -q` · Expected: ImportError/FAIL
- [ ] **Step 3: `events.py` 구현** — 노드 로그는 파일 mtime 으로 기간 거르기(field_report.read_logs 방식). stderr 는 `journalctl --user -u vica-llm --since … --until … -o short-iso --no-pager`(11-16~) 와 relaunch `voice*_*.log` 에서 `[vica_llm-N] [ERROR] [LLM] …` 줄만. 정규식이 answer_audit 에 있으면 그것을 import.
- [ ] **Step 4: 통과 확인** — 같은 명령 · Expected: PASS
- [ ] **Step 5: 커밋** — `git add tools/scene tests/test_scene_events.py tests/fixtures/scene && git commit -m "feat(scene): 로그 읽기·사건 분류"`

### Task 2: 장면 묶기와 대화 단계 시계열

**Files:** Create `tools/scene/scenes.py`, `tests/test_scene_scenes.py`

**Interfaces:**
- Consumes: `Event`, `read_events`.
- Produces:
  - `state_at(events: list[Event], t: float) -> str` — 미션 상태 전이 줄(`a -> b`)로 만든 그 시각의 dialog_state, 모르면 `""`.
  - `@dataclass Scene(id: str, start: float, end: float, trigger: str, events: list[Event], linked_by: str, question_id: str | None)` — trigger ∈ `wake|question`, linked_by ∈ `question_id|time`.
  - `build_scenes(events: list[Event]) -> list[Scene]` — 시작 = `wake_open`(호출 확정) 또는 `q_open`/질문 예약, 끝 = 다음 시작 직전. id = `YYYYMMDD-HHMMSS-<trigger>`.

- [ ] **Step 1: 실패하는 시험** — `test_scene_1507_contains_both_forwards`(15:07:19~ 장면에 user_text 2개·tts_play "여기까지 안내를 마칠까요?" 2개), `test_scene_1118_has_two_mission_questions`, `test_old_logs_linked_by_time`(11-04 → linked_by == "time"), `test_question_id_link`(합성 11-18 줄 → linked_by == "question_id"), `test_state_at_asking_next`.
- [ ] **Step 2: 실패 확인** — `.venv/bin/python -m pytest tests/test_scene_scenes.py -q` · FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `feat(scene): 장면 묶기·대화 단계 시계열`

### Task 3: 원인 판정

**Files:** Create `tools/scene/causes.py`, `tests/test_scene_causes.py`

**Interfaces:**
- Consumes: `Scene`, `Event`, `state_at`.
- Produces:
  - `@dataclass Cause(name: str, status: str, evidence: list[str], note: str = "")` — status ∈ `확정|추정|정상|미분류|확인 필요`.
  - `judge(scene: Scene, ctx: dict) -> list[Cause]` — ctx: `{"wake_threshold": float, "listen_min_rms": float, "approach_min_rms": float|None, "llm_timeout": 15.0}`(기동 줄 `startup` 사건에서 채움).
  - `flagged(causes: list[Cause]) -> bool` — `정상`만이면 False.
  - 이름(그대로): `호출 놓침`, `정상: 확정 직후 같은 말`, `정상: 대기 중 위치 알림`, `목소리 작음`, `말이 너무 짧아 버림`, `말소리 안 잡힘`, `LLM 시간 초과`, `재청취 기각`, `같은 말 두 번 전달`, `화자 판정으로 버림`, `대기 중 본인 확인 미달`, `질문 겹침`, `느린 응답`, `USB 끊김`, `재생 죽음`, `원인 미분류`.

- [ ] **Step 1: 실패하는 시험** (합격 장면 그대로)

```python
def test_1507_double_forward():     assert names(scene("15:07:19")) >= {"같은 말 두 번 전달"}
def test_1120_relisten():           assert ("재청취 기각", "확정") in pairs(scene("11:20:38"))
def test_1118_question_overlap():   assert ("질문 겹침", "확정") in pairs(scene("11:18:54"))
def test_1022_timeout_then_drop():  c = pairs(scene("10:22:43")); assert ("LLM 시간 초과", "추정") in c and ("재청취 기각", "확정") in c
def test_1109_wake_near():          assert ("호출 놓침", "추정") in pairs(scene_near("11:09:21"))
def test_near_within_2s_after_fire_is_normal(): ...
def test_usage_guide_two_sentences_not_overlap(): ...   # 요청 간격 < 0.1 s
def test_ghost_quiet_vs_short(): ...                     # rms<문턱 → 목소리 작음, rms≥문턱·짧음 → 말이 너무 짧아 버림
```

- [ ] **Step 2: 실패 확인** — FAIL
- [ ] **Step 3: 구현** — 규칙은 설계서 5.3 표 그대로. "시연자 말"은 `spk` 사건이나 marks.tsv 가 없으면 `확인 필요`.
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `feat(scene): 원인 판정 규칙`

### Task 4: 집계와 실제 로그 수용 확인

**Files:** Create `tools/scene/tally.py`, `tools/scene/__main__.py`(하위 명령 `report`), `tests/test_scene_tally.py`

**Interfaces:**
- Consumes: Task 1~3.
- Produces: `tally(events: list[Event], scenes: list[Scene], causes: dict[str, list[Cause]]) -> dict` — 키: `wake_fire, wake_near_over, wake_near_over_after_fire, ghost, ghost_followup, ghost_after_wake, relisten_drop, after_wake{heard,locate,quiet,silent,none}, overlap, flagged`; `python -m tools.scene report --since … --until …` 가 tally 와 어긋난 장면 목록을 출력.

- [ ] **Step 1: 실패하는 시험** — 고정 사건 목록으로 각 키 개수 확인(`test_after_wake_buckets`, `test_ghost_split_followup_after_wake`).
- [ ] **Step 2: 실패 확인** — FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 실제 로그 수용 확인(주행 묶음 꺼졌을 때)** — Run: `.venv/bin/python -m tools.scene report --since "2026-11-04 00:00" --until "2026-11-08 00:00"` · Expected: wake_fire 108 · wake_near_over 68(after_fire 14) · ghost 62(followup 54 · after_wake 8) · relisten_drop 11 · after_wake heard 72 / locate 15 / quiet 8 / silent 12 / none 1 · overlap 1(11-04 11:18:54) · 설계서 6절 장면 7개가 목록에 있음. 첫 분석 시간을 진행 기록에 적는다.
- [ ] **Step 6: 커밋** — `feat(scene): 집계·report 명령(11-04~07 기준선 일치)`

### Task 5: 설계상 대답 — 규칙 결과·예상 문장·LLM 다시 묻기

**Files:** Create `tools/scene/expected.py`, `tests/test_scene_expected.py` · Modify `src/mission_phrases.py`(표에 없는 미션 문장 사본 추가, 예 "안내를 취소할까요?") · `tests/test_mission_phrases.py`(미션 파일 글자 대조에 새 사본 포함)

**Interfaces:**
- Consumes: `Scene`, `state_at`, `tools.answer_replay.path_env/parse 경로`, `src.destination_loader`, `src.mission_phrases`.
- Produces:
  - `@dataclass Expected(intent: str|None, by: str, sentence: str|None, reply: str|None, note: str)` — by ∈ `규칙|LLM 추정|없음`.
  - `rule_expected(scene: Scene, destinations: list) -> Expected | None` — 지름길·로컬 규칙·질문 대답 규칙이 정하면 그 결과, 아니면 None.
  - `sentence_for(intent: str, dest_id: str|None, need_confirm: bool, state: str, destinations: list) -> str | None` — navigate+확인 → 그 목적지 `confirm_prompt`, 미션 고정 문장은 mission_phrases, 표에 없으면 None.
  - `reask(scene: Scene, path: str, interpret: Callable | None = None) -> Expected` — path ∈ `midm|gpt|realtime`; 캐시 `~/vica_data/scene_viewer/reask/<scene.id>_<path>.json`; midm 이고 `vica-drive.target` active 면 `RuntimeError("주행 중이라 로컬 다시 묻기 거부")`.

- [ ] **Step 1: 실패하는 시험** — `test_sentence_navigate_confirm`(목적지 "현장 안전 지원 휴머노이드" → "현장 안전 지원 휴머노이드로 안내해드릴까요?"), `test_rule_deny_short_answer`, `test_reask_uses_cache`(가짜 interpret 1번만 불림), `test_reask_midm_refused_when_driving`(drive 판정 주입), `test_expected_marks_estimate`(LLM 추정이면 note 에 "다시 물은 결과 — 그때와 다를 수 있음").
- [ ] **Step 2: 실패 확인** — FAIL
- [ ] **Step 3: 구현** — 다시 묻기는 answer_replay 와 같은 방식으로 import 전 환경값을 넣고, realtime 은 `clip_for` 로 저장 클립을 찾는다(없으면 "소리 없음"으로 Expected(by="없음")).
- [ ] **Step 4: 통과 확인** — PASS (`tests/test_mission_phrases.py` 포함)
- [ ] **Step 5: 커밋** — `feat(scene): 설계상 대답 — 규칙·예상 문장·LLM 다시 묻기`

### Task 6: 웹 서버와 화면

**Files:** Create `tools/scene/server.py`, `tools/scene/page.html` · Modify `tools/scene/__main__.py`(하위 명령 `serve --port 8772 --host 127.0.0.1`) · Create `tests/test_scene_server.py`

**Interfaces:**
- Consumes: Task 1~5.
- Produces: 경로 `GET /`(page) · `GET /api/scenes?since=&until=`(집계 + 목록, 캐시) · `GET /api/scene/<id>`(사건·원인·설계상 대답) · `GET /api/clip?file=`(listen_clips·wake_clips 안 파일만) · `POST /api/reask/<id>?path=`. 순수 함수 `resolve_clip(name: str) -> Path | None`.

- [ ] **Step 1: 실패하는 시험** — `test_clip_path_outside_dirs_rejected`(`../x.wav`·`/etc/passwd` → None), `test_api_scene_json_shape`(가짜 장면 → `causes`·`expected`·`events` 키), `test_reask_endpoint_refuses_when_driving`.
- [ ] **Step 2: 실패 확인** — FAIL
- [ ] **Step 3: 구현** — 화면은 목업(https://claude.ai/artifact/UY1gkfDvfCnG9bcbKyML6k) 구성 그대로. 단 11:18:54 는 "질문 겹침(미션 두 질문)", 빈손 창은 문구 그대로 보인다. 버튼 옆에 다시 묻기 경로 표시.
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 손으로 확인(주행 묶음 꺼졌을 때)** — `.venv/bin/python -m tools.scene serve` → 노트북 `ssh -L 8772:127.0.0.1:8772 <젯슨>` → 11-04 장면 7개를 열어 원인·시간축·소리 재생 확인.
- [ ] **Step 6: 커밋** — `feat(scene): 장면 분석기 화면(포트 8772)`

### Task 7: 기록기 `voice_trace`

**Files:** Create `src/voice_trace.py`, `tests/test_voice_trace.py`

**Interfaces:**
- Produces: `class TraceWriter(node: str, root: Path = ~/vica_data/voice_trace, enabled: bool | None = None)` — `enabled` None 이면 `VICA_TRACE`(기본 on); `write(stage: str, decision: str, reason: str = "", **values) -> None`(스레드 안전, 실패해도 예외 없음, 첫 실패만 경고 콜백); 날짜가 바뀌면 새 파일.

- [ ] **Step 1: 실패하는 시험** — `test_writes_one_json_line`, `test_off_writes_nothing`, `test_write_never_raises`(읽기 전용 폴더), `test_date_rollover`(시계 주입).
- [ ] **Step 2: 실패 확인** — FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — PASS
- [ ] **Step 5: 커밋** — `feat(voice): 판단 기록기 voice_trace(VICA_TRACE, 기본 켬, 동작 영향 없음)`

### Task 8: 호출 감시기 기록 — 넘은 조각 수·빈손 창 숫자

**Files:** Modify `src/wakeword_monitor.py`(`__init__` 인자 `on_trace`, `_feed_wake`·`_report_wake_score`·`_reset_wake_episode`, `_listen_step` 닫힘 지점 1373~1380 과 ghost·reject 지점), `src/ros_wakeword_node.py`(TraceWriter("wakeword") 연결, 빈손 창 로그 줄) · Create `tests/test_wakeword_trace.py`

**Interfaces:**
- Consumes: `TraceWriter`.
- Produces:
  - `WakewordMonitor(..., on_trace: Callable[[str, str, str, dict], None] | None = None)`.
  - 호출 기록: `on_trace("wake", kind, "", {"peak", "threshold", "frames_over", "max_streak", "cooldown_blocked", "self_guard"})` — kind ∈ `fire|near|self`.
  - 창 기록: `on_trace("listen", result, reason, {"window": "free|followup", "max_rms", "bg_rms", "speech_sec", "rms"})` — result ∈ `text|ghost|reject|silent|...`; `silent` 일 때 노드가 문구 그대로 info 로그.
  - 창 안 최대 rms·배경 rms 는 `_listen_step` 에서 프레임마다 이미 구하는 `_frame_rms(frame)` 로 누적(새 버퍼 없음). 배경 = 창 열린 뒤 0.5 s 안 프레임 평균.

- [ ] **Step 1: 실패하는 시험** — `test_near_reports_frames_over_one`(0.83 한 조각 → frames_over 1, max_streak 1), `test_fire_reports_streak_two`, `test_silent_followup_window_traced`(질문 창 무음 → result "silent", window "followup", max_rms·bg_rms 숫자), `test_trace_exception_does_not_stop_listening`(on_trace 가 예외 → 판정 결과 그대로), `test_no_trace_same_behavior`(on_trace None 과 있음의 반환값·콜백 순서 같음).
- [ ] **Step 2: 실패 확인** — `.venv/bin/python -m pytest tests/test_wakeword_trace.py -q` · FAIL
- [ ] **Step 3: 구현** — `/vica/listen_state` 의 "empty" 문자열은 그대로 둔다(미션이 읽음).
- [ ] **Step 4: 통과 + 전체 시험** — `.venv/bin/python -m pytest -q` · Expected: 기존 수 + 새 시험 모두 PASS
- [ ] **Step 5: 커밋** — `feat(voice): 호출 감시기 판단 기록 — 넘은 조각 수·빈손 창 '말소리 안 잡힘'`

### Task 9: 의도·TTS 노드 기록

**Files:** Modify `src/ros_node.py`(`_on_user_text` 329·`_on_user_audio` 667·재청취 기각 508 지점), `src/ros_tts_node.py`(`_playback_loop` 271~306) · Create `tests/test_node_trace_records.py`

**Interfaces:**
- Consumes: `TraceWriter`.
- Produces: 의도 노드 `TraceWriter("llm").write("input", "text|audio", …)`·`write("intent", intent, reason, need_confirm=…, src=…, dt=…, question_id=…)`·`write("relisten_drop", intent, "is_relisten_noise")`; TTS 노드 `write("play", "start|end|interrupted", priority=…, text=…, queue_wait=…)`. 기록 줄을 만드는 부분은 `src/voice_trace.py` 의 순수 함수(`intent_record(...) -> dict`, `play_record(...) -> dict`)로 두고 시험한다.

- [ ] **Step 1: 실패하는 시험** — `test_intent_record_fields`, `test_play_record_interrupted`.
- [ ] **Step 2: 실패 확인** — FAIL
- [ ] **Step 3: 구현** — 노드에는 한 줄 호출만 넣는다(기존 로그 줄은 그대로).
- [ ] **Step 4: 통과 + 전체 시험** — `.venv/bin/python -m pytest -q` · PASS
- [ ] **Step 5: 커밋** — `feat(voice): 의도·TTS 노드 판단 기록`

### Task 10: 분석기가 기록을 우선해서 읽기 + 실기 확인

**Files:** Modify `tools/scene/events.py`(trace 읽기), `tools/scene/causes.py`(trace 있으면 `호출 놓침` 확정·`말소리 안 잡힘` 숫자) · Create `tests/test_scene_trace_read.py`

**Interfaces:**
- Consumes: Task 7~9 기록 형식.
- Produces: `read_events(..., trace_root: Path = ~/vica_data/voice_trace)` — 같은 사건이 로그와 기록에 있으면 기록의 값으로 채운다.

- [ ] **Step 1: 실패하는 시험** — `test_near_confirmed_with_frames_over`(기록 frames_over 1 → `호출 놓침` 확정, note "넘은 조각 1개(2개 연속 필요)"), `test_silent_window_shows_numbers`.
- [ ] **Step 2: 실패 확인** — FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 + 전체 시험** — PASS
- [ ] **Step 5: 커밋** — `feat(scene): 판단 기록 우선 읽기`
- [ ] **Step 6: 실기 확인(사용자와, 음성 재기동 뒤)** — 호출을 일부러 작게·짧게 몇 번, 질문에 대답하지 않기 한 번 → 화면에서 호출 놓침 조각 수·"청취 창 말소리 없음(질문 창)"·소리 재생 확인, `VICA_TRACE=off` 로 한 번 더 돌려 동작 같음 확인.

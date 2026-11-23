# 세션 주인 목소리 따라가기 · 대기 중 본인 확인 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 세션 주인 1명의 목소리를 등록해 안내 중 다른 사람의 긴 말을 거르고, 대기 중 주행·종료·취소·다시 가자·대기 시간 바꾸기를 손잡이 + 목소리(되묻기 → 본인 질문)로 받으며, reSpeaker 소리가 끊기면 터미널에 남긴다.

**Architecture:** 판정은 호출 노드(`ros_wakeword_node` → `WakewordMonitor`) 안의 새 순수 모듈 `speaker_follow` 가 하고(sherpa-onnx ERes2Net, CPU), 결과를 `/vica/user_speaker`(JSON) 로 낸다. 의도 노드가 같은 문장의 판정을 `VicaIntent` 새 필드 3개에 싣고, 미션은 새 순수 모듈 `speaker_gate.WaitSpeakerGate` 로 대기 중 흐름을 정한다. 안내 중 거르기는 호출 노드에서 끝난다(미션 변경 없음).

**Tech Stack:** Python 3.10, ROS 2 Humble(rclpy, colcon), sherpa-onnx 2.0.1(aarch64 CPU 휠), numpy, pytest.

**Spec:** `vica-voice-llm/docs/superpowers/specs/2026-11-11-speaker-follow-design.md`(커밋 6d338e7). 9절 B1~B5 와 3절 D13~D15 가 4절보다 우선한다.

## Global Constraints

- 두 저장소 모두 브랜치 `after_feedback_waiting`. 새 브랜치 만들지 않음. 작업마다 커밋, 푸시·머지 없음(D15).
- 새 ROS 노드 추가 금지. 판정은 호출 노드 프로세스 안.
- "멈춰" 등 긴급 경로(`emergency_monitor`, `detect_emergency`, `on_emergency`)는 손대지 않는다(D12).
- 멘트 문구(그대로): 되묻기 `"확인을 위해 한 번 더 말씀해 주세요"` · 본인 질문 `"아까 {목적지}에 가자고 하신 사용자님이신가요?"` · 거절 `"죄송합니다. 다른 분을 안내 중이라 지금은 이용하실 수 없어요."` · 대기 첫 호출 `"지금은 대기중입니다. 사용자님을 기다리고 있어요"` · 손잡이 안내는 기존 `MSG_HANDLE_HINT`.
- 문턱 시작값: "비카야"+발화 0.55, 발화만 0.50, 되묻기 이어 붙인 점수 0.50, 짧은 발화 기준 1.0 s, 손잡이 부채꼴 `VICA_USER_DOA_CENTER`·`VICA_USER_DOA_WIDTH`(지금 180·60).
- 기본값은 모두 꺼짐: 음성 `VICA_SPK_FOLLOW=off`, 미션 파라미터 `speaker_gate=false`. 꺼져 있으면 지금 동작과 한 글자도 달라지지 않는다.
- 등록 벡터는 메모리에만. 파일로 쓰지 않는다. 판정 기록(점수·판정·사유)은 노드 로그와 `listen_clips/clips.jsonl` 에 남긴다.
- 안내 중 짧은 발화(말소리 < 1.0 s)는 거르지 않는다(D13). 대기 중 위험 명령은 목소리만으로 판정(B1).
- 세션 시작 = `dialog_state` 가 세션 밖에서 처음 `confirming`·`navigating` 이 됨. 끝 = `idle`·`returning`·`estopped`. `grip_wait` 는 끝 아님. 거르기는 `navigating` 을 한 번 본 뒤부터(B3).
- 등록 = 세션 시작 직전(10 s 안) 마지막 발화 하나. 호출 창이었으면 그 창의 "비카야" 소리를 앞에 붙인 것(D14). 누적 없음.
- 시험 명령: 음성 `cd ~/VICA-smarthandle/vica-voice-llm && .venv/bin/python -m pytest -q`, ROS `cd ~/VICA-smarthandle/vica_ros2_ws/src/vica_mission_manager && python3 -m pytest test -q`.

## Review Focus

- 대기 중 되묻기·본인 질문을 기다리는 동안 "멈춰"·비상정지가 오면 즉시 멈추고 문 상태가 지워져야 한다 → Task 8 `test_estop_clears_gate`.
- 의도에 판정 필드가 비어 오는 경우(소리 의도 경로, 판정 메시지가 늦거나 빠짐)는 통과가 아니라 본인 질문으로 가야 한다 → Task 8 `test_missing_verdict_goes_to_identity`.
- 임베딩이 None(말소리 너무 짧음·빈 소리)일 때 안내 중에는 버리지 않고, 대기 중에는 실패로 본다 → Task 3 `test_embed_none`.
- 손잡이 보드 끊김(`fresh` 거짓)이면 손잡이 앞 조건을 건너뛰고 목소리 판정으로 간다 → Task 8 `test_sensor_stale_skips_grip`.
- 앱으로 목적지를 정해 첫 발화 없이 `confirming` 이 되면 등록 없음(`no_target`) → Task 3 `test_start_without_recent_utterance`.

---

### Task 1: reSpeaker 소리 끊김 기록 (G3)

**Files:**
- Create: `vica-voice-llm/src/frame_watch.py`
- Modify: `vica-voice-llm/src/wakeword_monitor.py:1535-1593`(`run`)
- Test: `vica-voice-llm/tests/test_frame_watch.py`

**Interfaces:**
- Produces: `class FrameWatch(gap_sec: float = 1.0, repeat_sec: float = 10.0)` — `on_frame(now: float) -> Optional[str]`, `on_idle(now: float) -> Optional[str]`, `on_status(text: str, now: float) -> Optional[str]`. 돌려준 문자열이 있으면 로그로 낸다.

- [ ] **Step 1: 실패하는 시험 작성** — 손 계산 값으로:

```python
def test_gap_reported_once_then_every_10s():
    w = FrameWatch()
    assert w.on_frame(0.0) is None
    assert w.on_idle(0.5) is None
    assert w.on_idle(1.2) == "[USB] reSpeaker 소리가 1 s 째 안 들어옴 — 끊김 의심"
    assert w.on_idle(5.0) is None
    assert w.on_idle(11.3).startswith("[USB] reSpeaker 소리가 11 s 째")

def test_recovery_reported_with_gap_length():
    w = FrameWatch(); w.on_frame(0.0); w.on_idle(3.0)
    assert w.on_frame(4.5) == "[USB] 소리 다시 들어옴(끊김 4.5 s)"
    assert w.on_frame(4.6) is None

def test_no_recovery_line_without_reported_gap():
    w = FrameWatch(); w.on_frame(0.0)
    assert w.on_frame(0.08) is None

def test_status_same_text_rate_limited():
    w = FrameWatch()
    assert w.on_status("input overflow", 0.0) == "[USB] 입력 상태: input overflow"
    assert w.on_status("input overflow", 5.0) is None
    assert w.on_status("input overflow", 10.1) is not None
```

- [ ] **Step 2: 실행해 실패 확인** — `.venv/bin/python -m pytest tests/test_frame_watch.py -q` → ImportError.
- [ ] **Step 3: `FrameWatch` 구현** (초는 `on_idle` 은 내림 정수, 회복 줄은 소수 한 자리).
- [ ] **Step 4: `run()` 에 연결** — 콜백은 `status` 가 참이면 `str(status)` 를 별도 `queue.SimpleQueue`(상태 전용)에 넣기만 한다(콜백에서 로그를 쓰지 않는다). 감시 루프는 프레임을 `q.get(timeout=0.5)` 로 받고, `queue.Empty` 면 `watch.on_idle(now)`, 프레임이면 `watch.on_frame(now)`, 그리고 상태 큐에 쌓인 것을 꺼내 `watch.on_status(...)`. 나온 문자열은 `self._log(...)`(ROS 로거; `log` 가 None 이면 `print`). 복구·재시작은 하지 않는다.
- [ ] **Step 5: 시험 통과 + 전체 시험** — `.venv/bin/python -m pytest -q` 전부 통과.
- [ ] **Step 6: 커밋** — `feat(voice): reSpeaker 소리가 1 s 넘게 안 들어오면 터미널에 끊김 기록(11-11 사용자)`.

### Task 2: 말소리 자르기·임베딩을 src 로 옮기고 sherpa-onnx 를 운영 venv 에 설치

**Files:**
- Create: `vica-voice-llm/src/speech_trim.py`(`trim_speech` 와 상수 `FRAME_SEC`·`REL_DB`·`FLOOR`·`PAD_SEC`·`MERGE_GAP_SEC`·`KEEP_REL` 를 `tools/spk_verify_eval.py` 에서 그대로 옮김), `vica-voice-llm/src/spk_embed.py`
- Modify: `vica-voice-llm/tools/spk_verify_eval.py`(두 모듈에서 가져와 다시 내보냄 — 기존 시험 그대로 통과), `vica-voice-llm/requirements.txt`(`sherpa-onnx==2.0.1`)
- Test: 기존 `tests/test_spk_verify_eval.py`·`tests/test_spk_verify_record.py`

**Interfaces:**
- Produces: `src.speech_trim.trim_speech(x: np.ndarray, sr: int = 16000) -> np.ndarray` · `src.spk_embed.load_embedder(model_path: str, threads: int = 2) -> Callable[[np.ndarray], Optional[np.ndarray]]`(float32 −1~1 모노 16 kHz 를 받아 벡터, 너무 짧으면 None; sherpa 는 함수 안에서 import) · `src.spk_embed.cosine(a, b) -> float` · `src.spk_embed.mean_embedding(vectors) -> np.ndarray`.

- [ ] **Step 1: 옮기기** — 코드 이동만, 동작 변경 없음. `tools/spk_verify_eval.Embedder` 는 `load_embedder` 를 감싸는 얇은 클래스로 남긴다(시간 기록 유지).
- [ ] **Step 2: 기존 시험 실행** — `.venv/bin/python -m pytest tests/test_spk_verify_eval.py tests/test_spk_verify_record.py -q` → 31 passed.
- [ ] **Step 3: 설치** — `.venv/bin/pip install sherpa-onnx==2.0.1` 후 `.venv/bin/python -c "import sherpa_onnx, numpy; print(sherpa_onnx.__version__, numpy.__version__)"` → `2.0.1 2.2.6`(numpy 가 바뀌면 멈추고 보고).
- [ ] **Step 4: 실모델 확인** — `.venv/bin/python -c "from src.spk_embed import load_embedder; import numpy as np; e=load_embedder('$HOME/vica_data/spk_eval/models/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx'); print(e(np.zeros(16000,np.float32)) is None or e(np.random.randn(16000).astype(np.float32)*0.1).shape)"` → 벡터 모양 출력.
- [ ] **Step 5: 전체 시험 + 커밋** — `refactor(spk): 말소리 자르기·임베딩을 src 로, sherpa-onnx 2.0.1 운영 venv`.

### Task 3: 판정 핵심 `speaker_follow` (순수 모듈)

**Files:**
- Create: `vica-voice-llm/src/speaker_follow.py`
- Test: `vica-voice-llm/tests/test_speaker_follow.py`

**Interfaces:**
- Consumes: Task 2 의 `trim_speech`, `cosine`.
- Produces:
  - `@dataclass SpkConfig(mode: str = "off", model_path: str = "", th_wake: float = 0.55, th: float = 0.50, short_sec: float = 1.0, doa_center: Optional[float] = None, doa_width: float = 60.0)` · `spk_config_from_env(env: Mapping[str, str]) -> SpkConfig`(키 `VICA_SPK_FOLLOW`·`VICA_SPK_MODEL`·`VICA_SPK_TH_WAKE`·`VICA_SPK_TH`·`VICA_SPK_SHORT_SEC`·`VICA_USER_DOA_CENTER`·`VICA_USER_DOA_WIDTH`; 모르는 mode 는 `off`).
  - `@dataclass Verdict(verdict: str, score: float, joined: float, unit: str, speech_sec: float, doa: Optional[float], drop: bool, reason: str, text: str = "")` — `verdict` ∈ `pass`·`fail`·`short`·`no_target`; 점수 없음은 −1; `unit` ∈ `wake+utt`·`utt`; `to_json() -> str`(키 `verdict`·`score`·`joined`·`unit`·`speech_sec`·`doa`·`reason`·`text`).
  - `judge(score: Optional[float], has_wake: bool, speech_sec: float, doa: Optional[float], cfg: SpkConfig, *, has_target: bool, guiding: bool, waiting: bool) -> tuple[str, bool, str]` → (verdict, drop, reason).
  - `class SessionTracker` — `on_dialog_state(state: str) -> Optional[str]`(`"start"`·`"end"`·None), 속성 `active`, `guiding`, `waiting`.
  - `class SpeakerFollow(cfg: SpkConfig, embed: Callable[[np.ndarray], Optional[np.ndarray]], log: Callable[[str], None] = print)` — `note_wake_audio(audio_i16: np.ndarray, now: float) -> None`, `on_dialog_state(state: str, now: float) -> None`, `check(audio_i16: np.ndarray, is_wake_window: bool, doa: Optional[float], speech_sec: float, now: float, text: str = "") -> Verdict`, 속성 `has_target`.

`judge` 규칙(이 순서, 스펙 4.2·B1·B3·D13):
1. `not has_target` → (`no_target`, False, "등록 없음").
2. `waiting` → 목소리만: 점수 None 이면 `fail`; 문턱은 `has_wake` 면 `th_wake` 아니면 `th`. drop False.
3. `speech_sec < short_sec` → (`short`, False, "짧은 말").
4. 점수 ≥ 문턱 → `pass`; 아니면 DOA 가 부채꼴(최단 각 차 ≤ `doa_width`) → `pass`(reason "방향"); 아니면 `fail`. 점수 None 이면 방향만 본다.
5. `drop = cfg.mode == "on" and guiding and verdict == "fail"`.

`SpeakerFollow` 규칙: `check` 는 이번 발화 소리(호출 창이면 30 s 안에 받은 "비카야" 소리 + 발화, 각각 `trim_speech`)로 점수를 내고, 직전 발화(20 s 안)와 이번 발화(비카야 없이)를 이어 붙인 점수를 `joined` 로 낸 뒤 이번 소리를 "마지막 발화"(시각 포함)로 보관한다. `on_dialog_state` 가 `"start"` 를 받으면 마지막 발화가 10 s 안이면 그 소리로 등록, 아니면 등록 없음. `"end"` 면 등록·마지막 발화·"비카야" 소리를 모두 지운다. `mode == "off"` 면 `check` 는 임베딩을 하지 않고 `Verdict("no_target", -1, -1, ..., drop=False, reason="꺼짐")`.

- [ ] **Step 1: 실패하는 시험 작성** — 가짜 임베딩(소리 첫 샘플 부호로 [1,0] 또는 [0,1] 벡터, 길이 < 1600 이면 None)으로:

```python
def test_judge_table():
    cfg = SpkConfig(mode="on", doa_center=180, doa_width=60)
    kw = dict(has_target=True, guiding=True, waiting=False)
    assert judge(0.60, True, 2.0, 0.0, cfg, **kw)[:2] == ("pass", False)
    assert judge(0.52, True, 2.0, 0.0, cfg, **kw)[:2] == ("fail", True)       # 비카야 문턱 0.55
    assert judge(0.52, False, 2.0, 0.0, cfg, **kw)[:2] == ("pass", False)     # 발화만 문턱 0.50
    assert judge(0.10, False, 2.0, 200.0, cfg, **kw)[:2] == ("pass", False)   # 손잡이 방향
    assert judge(0.10, False, 0.8, 0.0, cfg, **kw)[:2] == ("short", False)
    assert judge(0.10, False, 2.0, 0.0, cfg, has_target=True, guiding=False, waiting=False)[:2] == ("fail", False)

def test_waiting_is_voice_only():
    cfg = SpkConfig(mode="on", doa_center=180, doa_width=60)
    assert judge(0.10, False, 0.6, 180.0, cfg, has_target=True, guiding=True, waiting=True)[:2] == ("fail", False)

def test_no_target():
    assert judge(0.9, True, 2.0, None, SpkConfig(mode="on"), has_target=False, guiding=True, waiting=False)[0] == "no_target"

def test_log_mode_never_drops():
    assert judge(0.1, False, 2.0, 0.0, SpkConfig(mode="log", doa_center=180, doa_width=60),
                 has_target=True, guiding=True, waiting=False)[1] is False

def test_session_tracker():
    t = SessionTracker()
    assert [t.on_dialog_state(s) for s in ("idle", "grip_wait", "confirming", "navigating", "waiting", "idle")] \
        == [None, None, "start", None, None, "end"]

def test_enroll_from_last_utterance_on_start(): ...      # check(A) 후 start → has_target, check(A) pass, check(B) fail
def test_start_without_recent_utterance(): ...           # 11 s 지난 발화뿐이면 no_target
def test_end_clears_target(): ...                        # estopped → has_target False
def test_wake_audio_prepended_in_wake_window(): ...      # unit == "wake+utt"
def test_joined_uses_previous_within_20s(): ...          # 25 s 지나면 joined == -1
def test_embed_none(): ...                               # 안내 중 → short 또는 방향 판정, drop False / 대기 중 → fail
def test_off_mode_does_not_embed(): ...                  # 가짜 임베딩 호출 0 번
```

- [ ] **Step 2: 실패 확인** — `.venv/bin/python -m pytest tests/test_speaker_follow.py -q`.
- [ ] **Step 3: 구현** — 위 시그니처·규칙대로.
- [ ] **Step 4: 통과 + 전체 시험.**
- [ ] **Step 5: 커밋** — `feat(spk): 세션 주인 목소리 판정 핵심(speaker_follow) — 등록·판정·세션 경계`.

### Task 4: 호출 감시기(`WakewordMonitor`)에 연결

**Files:**
- Modify: `vica-voice-llm/src/wakeword_monitor.py`(`__init__` 445-480, `run` 1535-1593, `_open_listen` 1103, `_listen_step` 1254-1425, `_report_clip` 1427)
- Test: `vica-voice-llm/tests/test_wakeword_monitor.py`(기존 `make` 도우미 34행 재사용)

**Interfaces:**
- Consumes: Task 3 `SpeakerFollow`, `Verdict`.
- Produces: 생성자 인자 `speaker: Optional[SpeakerFollow] = None`, `on_speaker_verdict: Optional[Callable[[Verdict], None]] = None`; 메서드 `set_dialog_state(state: str) -> None`(`speaker.on_dialog_state` 로 넘김, `speaker` 없으면 아무것도 안 함).

- [ ] **Step 1: 실패하는 시험** — 가짜 `speaker`(정해진 `Verdict` 를 돌려주고 호출 인자를 기록):
  - `test_wake_audio_copied_on_wake` — 호출 확정 때 `note_wake_audio` 가 링 전체(31 조각) 길이 소리로 1 번 불린다.
  - `test_drop_verdict_suppresses_text` — `drop=True` 판정이면 `texts` 가 비고, `_report_clip` 결과가 `"spk-drop"`, 청취 상태 `"empty:speaker"`.
  - `test_verdict_callback_before_text` — 통과 판정이면 `on_speaker_verdict` 가 글자 콜백보다 먼저 불린다.
  - `test_doa_median_passed` — 청취 창에서 VAD 참인 조각의 DOA [170, 182, 190] → `check(doa=182.0)`.
  - `test_no_speaker_unchanged` — `speaker=None` 이면 기존 시험과 같은 결과(콜백 0 번).
- [ ] **Step 2: 실패 확인.**
- [ ] **Step 3: 구현** — (a) 호출 확정 지점(`run` 1590 근처 `r == "wake"`)에서 `np.concatenate(list(self._ring))` 을 `note_wake_audio` 로. (b) `run` 에서 `speaker` 가 있고 청취 상태이며 VAD 참이면 `dsp.doa_angle()` 을 읽어 `process_frame(doa=...)` 로 넘기고, `_listen_step` 은 `_open_listen` 에서 비운 `self._listen_doas` 에 쌓는다. (c) 받아쓰기·환각 기각을 통과한 뒤(1416·1421 직전) `speaker.check(audio, is_wake_window=not self._listen_is_followup, doa=중앙값 또는 None, speech_sec, now, text)` → `on_speaker_verdict(v)` → `v.drop` 이면 기록·버림, 아니면 기존대로 글자를 넘긴다. (d) `_report_clip` 에 `spk_verdict`·`spk_score`·`spk_joined`·`spk_reason` 필드 추가(판정이 있을 때만).
- [ ] **Step 4: 통과 + 전체 시험.**
- [ ] **Step 5: 커밋** — `feat(spk): 호출 감시기 — 비카야 소리 보관·청취 창 DOA·판정 뒤 거르기`.

### Task 5: `VicaIntent` 필드 추가 (ROS 저장소)

**Files:**
- Modify: `vica_ros2_ws/src/vica_interfaces/msg/VicaIntent.msg`(끝에 3 줄), `vica_ros2_ws/src/vica_mission_manager/vica_mission_manager/mission_logic.py:180-193`(`IntentData`), `vica_ros2_ws/src/vica_mission_manager/vica_mission_manager/mission_manager_node.py:791-801`(`_on_intent`)
- Test: `vica_ros2_ws/src/vica_mission_manager/test/test_mission_logic.py`(기본값 시험 1 개)

**Interfaces:**
- Produces: 메시지 `string speaker_verdict`(`""`·`pass`·`fail`·`short`·`no_target`), `float32 speaker_score`, `float32 speaker_score_joined`(없으면 −1) · `IntentData.speaker_verdict: str = ""`, `IntentData.speaker_score: float = -1.0`, `IntentData.speaker_joined: float = -1.0`. 이름 대응: 메시지 `speaker_score_joined` ↔ 파이썬 `speaker_joined`(미션 `IntentData`, 음성 `VicaIntent` 둘 다).

- [ ] **Step 1: 시험** — `IntentData(intent="cancel", matched_destination_id="", need_confirm=False, safety_flag="normal")` 의 세 필드 기본값이 `""`·−1.0·−1.0.
- [ ] **Step 2: 메시지·자료형·노드 변환 추가** — 노드는 `getattr(msg, "speaker_verdict", "")` 처럼 옛 메시지에도 견디게.
- [ ] **Step 3: 빌드** — `cd ~/VICA-smarthandle/vica_ros2_ws && colcon build --packages-select vica_interfaces vica_mission_manager` → 성공. `source install/setup.bash && ros2 interface show vica_interfaces/msg/VicaIntent | tail -3` → 새 필드 3 줄.
- [ ] **Step 4: 미션 전체 시험 + 커밋** — `feat(interfaces): VicaIntent 에 화자 판정 필드 3개(11-11 설계)`.

### Task 6: 호출 노드·의도 노드 연결 (음성 저장소)

**Files:**
- Modify: `vica-voice-llm/src/ros_wakeword_node.py`(생성 192, `_on_user_text` 315-341, `_on_robot_state` 400-407), `vica-voice-llm/src/ros_node.py`(구독 151 근처, `_publish_intent` 357), `vica-voice-llm/src/schema.py:70`(`VicaIntent`), `vica-voice-llm/src/ros_convert.py:26-37`(`intent_to_msg`), `vica-voice-llm/src/mission_question.py:62`(`quiet_for_mission`), `vica-voice-llm/.env.example`
- Create: `vica-voice-llm/src/speaker_link.py`(의도 노드 쪽 판정 보관함, 순수)
- Test: `vica-voice-llm/tests/test_speaker_link.py`, `tests/test_mission_question*.py`(있는 파일에 추가), `tests/test_ros_convert*.py`(있으면 추가, 없으면 `test_speaker_link.py` 에)

**Interfaces:**
- Consumes: Task 3 `spk_config_from_env`, `SpeakerFollow`, `Verdict.to_json`; Task 2 `load_embedder`; Task 5 메시지 필드.
- Produces: `class SpeakerLink(ttl_sec: float = 15.0)` — `put(json_text: str, now: float) -> None`, `take(text: str, now: float) -> Optional[dict]`(같은 글자의 가장 최근 판정을 꺼내고 지움; 없으면 None) · `VicaIntent.speaker_verdict: str = ""`, `speaker_score: float = -1.0`, `speaker_joined: float = -1.0` · `quiet_for_mission(..., spk_on: bool = False)`.

- [ ] **Step 1: 실패하는 시험**
  - `test_take_matches_text_and_consumes`, `test_take_expires_after_ttl`, `test_take_none_when_absent`.
  - `test_intent_to_msg_carries_speaker_fields` — 값 `"fail"`·0.42·0.61 이 메시지에 그대로.
  - `test_quiet_for_mission_waiting_gated_kinds` — `spk_on=True`, `dialog_state="waiting"`, `navigate`·`cancel`·`finish`·`resume`·`wait` 는 reply `""`; `question` 은 그대로; `spk_on=False` 면 지금과 같음.
- [ ] **Step 2: 실패 확인.**
- [ ] **Step 3: 구현**
  - 호출 노드: `spk_config_from_env(os.environ)` 이 `off` 가 아니면 `load_embedder(cfg.model_path)` 로 `SpeakerFollow` 를 만들어 감시기에 넘긴다(모델을 못 올리면 FATAL 로그 후 `off` 로 계속 — 듣기는 멈추지 않는다). `on_speaker_verdict` 는 판정을 `self._spk_pending` 에 두고 로그 `[화자] {verdict} 점수 {score:.2f}/{joined:.2f} {unit} {speech_sec:.1f}s {reason}`. `_on_user_text` 가 글자를 내기 **직전**, 보관된 판정(15 s 안)의 `text` 를 에코 제거 뒤 글자로 바꿔 `/vica/user_speaker`(String) 로 내고 비운다. `_on_robot_state` 는 기존 일에 더해 `self._monitor.set_dialog_state(msg.dialog_state)`.
  - 의도 노드: `/vica/user_speaker` 구독 → `SpeakerLink.put`. `_publish_intent(intent, text)` 에서 `take(text)` 결과를 의도의 세 필드에 싣는다(없으면 기본값). `quiet_for_mission` 호출에 `spk_on=(VICA_SPK_FOLLOW == "on")` 전달; 규칙: `spk_on and dialog_state in ("waiting", "waiting_release") and intent.intent in ("navigate", "cancel", "finish", "resume", "wait")` → reply `""`.
  - `.env.example` 에 7절 키 5 개(주석에 뜻·시작값).
- [ ] **Step 4: 통과 + 전체 시험.**
- [ ] **Step 5: 손 확인(ROS 켜진 젯슨, 음성 칸만)** — `VICA_SPK_FOLLOW=log` 로 띄우고 "비카야 … 화장실 가자" 한 번 → 호출 노드 로그에 `[화자] no_target` 1 줄, `ros2 topic echo /vica/user_speaker --once` 에 JSON 1 개.
- [ ] **Step 6: 커밋** — `feat(spk): 판정을 /vica/user_speaker 로 내고 의도에 실음 · 대기 중 위험 의도는 미션이 말함`.

### Task 7: 미션 대기 문 `WaitSpeakerGate` (순수) + 손잡이 '최근 접촉'

**Files:**
- Create: `vica_ros2_ws/src/vica_mission_manager/vica_mission_manager/speaker_gate.py`
- Modify: `vica_ros2_ws/src/vica_mission_manager/vica_mission_manager/grip_meter.py`
- Test: `vica_ros2_ws/src/vica_mission_manager/test/test_speaker_gate.py`, `test/test_grip_meter.py`

**Interfaces:**
- Produces:
  - `GripMeter.touched_within(now: float, sec: float) -> bool` — 지난 `sec` 초 안에 접촉 참 구간이 하나라도 있었나(`history_sec` 10 s 이상 보관).
  - `GATED_KINDS = ("navigate", "cancel", "finish", "resume", "wait")`, `GATE_WINDOW_SEC = 15.0`, `GRIP_RECENT_SEC = 10.0`.
  - `@dataclass(frozen=True) GateOut(action: str, use: str = "current", skip_confirm: bool = False, note: str = "")` — `action` ∈ `pass`·`hint`·`reask`·`identity`·`reject`·`drop`; `use` ∈ `current`·`stored`.
  - `class WaitSpeakerGate(joined_threshold: float = 0.50)` — `decide(intent: IntentData, grip_recent: bool, sensor_ok: bool, followup_open: bool, now: float) -> GateOut`, `stored -> Optional[IntentData]`, `reset() -> None`, `tick(now: float) -> None`(15 s 지나면 조용히 `reset`).

`decide` 규칙(스펙 4.5·B1~B4, 손 계산 시험으로 고정):
- 열린 단계 없음:
  - `kind` 가 `GATED_KINDS` 밖이거나 `followup_open`(미션 자신의 대기 질문에 답하는 중) → `pass`.
  - `sensor_ok and not grip_recent` → `hint`.
  - `speaker_verdict == "pass"` → `pass`.
  - `speaker_verdict in ("", "no_target")` → 명령 보관, 단계 = 본인 질문, `identity`.
  - 그 밖 → 명령 보관, 단계 = 되묻기, `reask`.
- 되묻기 단계(15 s 안): `deny` → `reset`, `drop`. `speaker_joined >= joined_threshold`: 같은 `kind` → `pass`, `use="current"`, `skip_confirm=True`; `GATED_KINDS` 밖(질문·불평·unknown·affirm) → `pass`, `use="stored"`; 다른 위험 의도 → `pass`, `use="current"`. 미달 → 단계 = 본인 질문(위험 의도면 그것을 보관, 아니면 첫 명령 유지; 같은 명령 두 번이었는지 기억), `identity`.
- 본인 질문 단계(15 s 안): `affirm` → `pass`, `use="stored"`, `skip_confirm`=같은 명령 두 번 들었나, `note="[본인질문] 출발"`. `deny`·`cancel` → `reject`. 그 밖 → `reset`, `drop`.

- [ ] **Step 1: 실패하는 시험** — `test_touched_within_after_auto_release`(0~8.2 s 접촉 후 놓음 → 9 s 에 `touched_within(9, 10)` 참, 19 s 에 거짓), 그리고 `test_speaker_gate.py` 에 위 규칙의 각 갈래 하나씩(이름 예: `test_not_gated_kind_passes`, `test_no_grip_hint`, `test_pass_verdict_passes`, `test_fail_then_same_command_skips_confirm`, `test_fail_then_question_replays_stored`, `test_fail_then_other_command_keeps_confirm`, `test_fail_twice_goes_identity`, `test_identity_affirm_passes_with_note`, `test_identity_deny_rejects`, `test_identity_silence_resets_on_tick`, `test_reask_deny_drops`, `test_followup_open_passes`, `test_missing_verdict_goes_to_identity`).
- [ ] **Step 2: 실패 확인** — `python3 -m pytest test/test_speaker_gate.py test/test_grip_meter.py -q`.
- [ ] **Step 3: 구현.**
- [ ] **Step 4: 통과 + 미션 전체 시험.**
- [ ] **Step 5: 커밋** — `feat(mission): 대기 중 본인 확인 문(WaitSpeakerGate)·손잡이 최근 접촉`.

### Task 8: 미션 로직에 문 연결 · 멘트 · 파라미터

**Files:**
- Modify: `vica_ros2_ws/src/vica_mission_manager/vica_mission_manager/mission_logic.py`(멘트 상수 묶음 근처 496·683, `on_voice_intent` 1690, 대기 중 `navigate` 제안 2230-2240, `on_wake_call` 3782-3792, `on_tick`, 세션 끝 `_to_idle`·`_enter_returning`·`_enter_estopped`), `mission_manager_node.py`(파라미터 선언 293 근처), `launch/mission_manager.launch.py`
- Test: `vica_ros2_ws/src/vica_mission_manager/test/test_wait_speaker_flow.py`

**Interfaces:**
- Consumes: Task 7 `WaitSpeakerGate`, `GateOut`, `GripMeter.touched_within`; Task 5 `IntentData` 새 필드.
- Produces: 생성자 인자 `speaker_gate: bool = False`, `spk_joined_threshold: float = 0.50`; 노드 파라미터 `speaker_gate`(기본 false)·`spk_joined_threshold`(0.50), launch 인자 같은 이름. 상수 `MSG_SPK_REASK`·`MSG_SPK_IDENTITY`(`{dest}` 자리)·`MSG_SPK_REJECT`·`MSG_WAIT_IN_USE` (Global Constraints 문구 그대로).

`speaker_gate` 가 참이고 상태가 `WAITING`·`WAITING_RELEASE` 일 때만 `on_voice_intent` 맨 앞(`_answer_predates_question` 다음)에서 `decide` 를 부른다. `grip_recent = self._grip.touched_within(now, GRIP_RECENT_SEC)`, `sensor_ok = self._grip.fresh(now)`, `followup_open` = `_wait_need_asked_at`·`_wait_finish_asked_at` 이 열려 있음(`WAIT_FINISH_REPEAT_SEC` 안). 결과 → 동작:
- `pass` → `use` 대로 고른 의도로 기존 흐름. `skip_confirm` 이면: `navigate` → `dataclasses.replace(intent, need_confirm=False)`; `cancel`·`finish` → `_reset_arrival_dialog()` 후 `[Say(MSG_FINISH, priority="response"), *self._go_home(now)]`; `resume`·`wait` → 기존 흐름. `note` 가 있으면 `voice_note` 에 붙인다.
- `hint` → `[Haptic(HAPTIC_PATTERN_HANDLE_HINT), Say(MSG_HANDLE_HINT, priority="response")]`.
- `reask` → `[self._ask(MSG_SPK_REASK)]` · `identity` → `[self._ask(MSG_SPK_IDENTITY.format(dest=self._arrived_destination.name))]`(없으면 `"목적지"`) · `reject` → `[Say(MSG_SPK_REJECT, priority="response")]` · `drop` → `[]`.
- 대기 상태의 `navigate` 제안(`need_confirm`)은 B5 에 따라 미션이 확인 질문도 말한다: `speaker_gate` 이면 CONFIRMING 으로 갈 때 `self._ask(self._confirm_prompt)` 를 덧붙인다.
- `on_wake_call` WAITING 첫 호출: `speaker_gate` 면 `MSG_WAIT_IN_USE`, 아니면 `MSG_WAIT_LOCATE`(지금 그대로).
- `on_tick` 에서 `gate.tick(now)`; `_to_idle`·`_enter_returning`·`_enter_estopped` 에서 `gate.reset()`.

- [ ] **Step 1: 실패하는 시험**(기존 대기 시험 `test/test_wait_spot.py` 의 대기 진입 도우미 재사용):
  - `test_gate_off_keeps_current_behavior` — `speaker_gate=False` 면 대기 중 `cancel` 이 지금처럼 `MSG_WAIT_NEED_ASK`.
  - `test_wait_first_wake_in_use_ment` — 첫 호출 멘트가 `"지금은 대기중입니다. 사용자님을 기다리고 있어요"`.
  - `test_no_grip_handle_hint` · `test_sensor_stale_skips_grip`(보드 끊김이면 손잡이 앞 조건 건너뜀 → 판정으로).
  - `test_pass_navigate_asks_confirm_by_mission` — 통과한 `navigate` 제안 → CONFIRMING + 미션이 확인 질문을 말함.
  - `test_reask_then_same_cancel_finishes_without_need_ask` — `MSG_FINISH` + RETURNING, `MSG_WAIT_NEED_ASK` 없음.
  - `test_identity_question_uses_destination_name` — `"아까 화장실에 가자고 하신 사용자님이신가요?"`.
  - `test_identity_affirm_departs_with_note` · `test_identity_deny_rejects_and_keeps_waiting`.
  - `test_missing_verdict_goes_to_identity` · `test_estop_clears_gate`(본인 질문 중 비상정지 → ESTOPPED, 해제 뒤 새 세션에서 `affirm` 이 출발시키지 않음).
- [ ] **Step 2: 실패 확인.**
- [ ] **Step 3: 구현.**
- [ ] **Step 4: 통과 + 미션 전체 시험 + 빌드** — `colcon build --packages-select vica_mission_manager`.
- [ ] **Step 5: 커밋** — `feat(mission): 대기 중 본인 확인 흐름(손잡이 안내·되묻기·본인 질문·거절) — speaker_gate 파라미터, 기본 꺼짐`.

### Task 9: 마무리 — 문서·전체 시험·실기 준비 (사용자와 함께)

**Files:**
- Modify: `vica-voice-llm/docs/superpowers/specs/2026-11-11-speaker-follow-design.md`(7절 설정에 미션 파라미터 `speaker_gate`·`spk_joined_threshold` 추가, "미션이 값을 어떻게 아는지" 문장을 이것으로 바꿈)

- [ ] **Step 1: 두 저장소 전체 시험** — 음성 `.venv/bin/python -m pytest -q`, 미션 `python3 -m pytest test -q` → 실패 0. 실패가 있으면 이름과 함께 보고.
- [ ] **Step 2: 꺼진 상태 회귀 확인** — `.env` 를 건드리지 않은 채 음성·미션을 띄워 호출·대기 첫 호출이 지금과 같은지(대기 첫 호출 멘트 "비카가 여기 있어요").
- [ ] **Step 3: 사용자 확인 후 `log` 실기** — `.env` 에 `VICA_SPK_FOLLOW=log`, `VICA_SPK_MODEL=~/vica_data/spk_eval/models/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx` 를 넣는 것은 사용자에게 묻고 한다(운영 설정). 장면 1~5 를 돌려 `clips.jsonl` 의 `spk_*` 로 점수 분포를 본다.
- [ ] **Step 4: 설계 문서 갱신 커밋** — `docs(spec): 화자 따라가기 — 미션 파라미터 speaker_gate 반영`.

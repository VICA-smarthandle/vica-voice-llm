# 대답 재생 도구 · 기준 점수 Implementation Plan (설계 7절 1단계)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 실기 로그의 "로봇 질문 → 사용자 대답"을 정답 묶음으로 만들고, 지금 구조(글자 경로 Mi:dm·gpt, 소리 경로 Realtime)에 다시 넣어 "잘못된 행동" 기준 점수를 잰다.

**Architecture:** 음성 저장소 `tools/` 에 순수 함수 모듈 셋(로그 뽑기·채점·정답 초안)과 실행 도구 하나(재생). 지금 `src/` 동작은 바꾸지 않는다 — 지금 구조를 그대로 부르는 측정 단계다. 로그·정답·결과는 저장소 밖 `~/vica_data/voice_answer_audit_20261113/`.

**Tech Stack:** Python 3.10 (`.venv`), pytest, `src.langchain_intent_parser.parse_intent`·`parse_intent_audio`, Ollama(midm2-mini), OpenAI(gpt-5.4-mini·Realtime).

**Spec:** `docs/superpowers/specs/2026-11-14-question-aware-answers-design.md` (5절 측정 · 7절 순서 1).

## Global Constraints

- `src/` 와 ROS 저장소는 이 계획에서 고치지 않는다. 새 ROS 노드 없음.
- 로그·정답·결과 파일은 저장소에 넣지 않는다: `~/vica_data/voice_answer_audit_20261113/`. 있는 파일(`qa_all.csv` 등)은 지우거나 덮지 않는다 — 새 이름으로 쓴다.
- 정답 칸 `answer` 값은 설계 4.2 그대로: `yes` `no` `destination` `choice` `minutes` `repeat` `other` `not_for_robot` `unclear`.
- 질문 kind 값은 설계 4.1 그대로: `open` `confirm` `choice` `yesno` `yesno_inverted` `minutes` (+ 이 도구 전용 `other` = 분류 안 되는 LLM 자유 질문).
- 잘못된 행동 = 정반대 동작·엉뚱한 출발·조용히 버림(설계 5절). `unclear` 로 되묻는 것은 잘못이 아니다.
- 주석·문서는 한국어, 짧게. 시험: `.venv/bin/python -m pytest` (ROS 환경 source 하지 않은 셸).
- Ollama 는 midm2-mini 를 `keep_alive -1` 로 둔다(10-24 결정). 다른 모델을 올렸으면 그것만 내린다.
- 주행 중에는 재생을 돌리지 않는다(무거운 작업 금지 규칙).
- 커밋 브랜치는 실행 전에 사용자에게 묻는다(새 브랜치 선승인 규칙).

## Review Focus

1. 행의 목적지 파일이 그 뒤 바뀌었거나 없음(10-29 이름 바꿈) — 재생은 그 행을 `skip` 으로 세고 멈추지 않는다. (Task 4 시험)
2. Ollama 꺼짐·모델 미적재 — `parse_intent` 가 `LLM_UNAVAILABLE` 문장으로 `unknown` 을 내므로 `unclear`(되묻기)로 채점돼 실패가 숨는다. 재생은 그 행을 `error` 로 세고, 연속 5건이면 멈춘다. (Task 4 시험)
3. 겹친 로그의 같은 말 두 줄(10-10 21:24:24 처럼) — 뽑기에서 (시각 0.5 s, 글자) 로 하나만 남긴다. (Task 1 시험)
4. 빈 대답 글자(로그 `A=''`) — 초안 정답 `unclear`, 재생은 LLM 을 부르지 않고 `unclear` 로 낸다. (Task 3·4 시험)
5. 같은 질문이 미션·음성 두 곳에서 같은 글자로 나옴("몇 분쯤 걸리실까요?") — kind 는 글자로만 정하고 `q_by` 는 기록만. (Task 1 시험)

---

### Task 1: 로그에서 질문·대답 행 뽑기

**Files:**
- Create: `tools/answer_audit.py`
- Test: `tests/test_answer_audit.py`

**Interfaces:**
- Produces:
  - `kind_from_text(question: str) -> str` — 설계 4.1 kind 또는 `other`
  - `parse_log_line(line: str) -> tuple[float, str, str] | None` — (epoch 초, 노드 이름, 본문)
  - `extract_rows(lines: Iterable[tuple[float, str, str]]) -> list[dict]` — 열: `ts` `date` `time` `question` `q_by` `kind` `answer_text` `stt` `path` `intent` `dest_id` `nc` `llm_reply` `dialog_state` `mission` `dest_yaml`
  - CLI `python -m tools.answer_audit --since 2026-10-10 --out ~/vica_data/voice_answer_audit_20261113/qa_rows.csv`

- [ ] **Step 1: 실패하는 시험 쓰기**

```python
@pytest.mark.parametrize("q,kind", [
    ("네?", "open"), ("어디로 모실까요?", "open"), ("잘 듣지 못했습니다. 어디로 가고 싶으신가요?", "open"),
    ("확인을 위해 한 번 더 말씀해 주세요", "open"),
    ("사회복지로 안내해드릴까요?", "confirm"),
    ("남자 화장실과 여자 화장실 중 어디로 갈까요?", "choice"),
    ("사회복지는 왼쪽에 있습니다. 여기서 대기할까요?", "yesno"), ("여기까지 안내를 마칠까요?", "yesno"),
    ("안녕하세요? 시각장애인 안내로봇 비카입니다. 안내를 받으시겠어요?", "yesno"),
    ("아까 화장실에 가자고 하신 사용자님이신가요?", "yesno"), ("직원의 도움이 필요하신가요?", "yesno"),
    ("안내가 필요 없으신가요?", "yesno_inverted"), ("몇 분쯤 걸리실까요?", "minutes"),
    ("어떤 서류를 내러 가시는지요?", "other"),
])
def test_kind_from_text(q, kind):
    assert kind_from_text(q) == kind

def test_rows_pair_question_answer_reaction_and_map():
    lines = [  # (ts, node, body) — 실제 로그 꼴
        (100.0, "vica_mission_manager", "목적지 9개 로드: /x/vica_map_0630/destinations.yaml"),
        (110.0, "vica_tts_node", "재생[response]: 여기서 대기할까요?"),
        (115.0, "vica_llm_intent_node", "입력='아니' -> intent=deny matched= safety=normal"),
        (115.1, "vica_mission_manager", "음성 요청 intent=deny dest=- confirm=False: asking_next -> asking_next 말=['여기까지 안내를 마칠까요?']"),
    ]
    [row] = extract_rows(lines)
    assert (row["kind"], row["answer_text"], row["intent"], row["dialog_state"]) == ("yesno", "아니", "deny", "asking_next")
    assert row["dest_yaml"] == "/x/vica_map_0630/destinations.yaml"

def test_duplicate_overlapping_lines_kept_once():
    q = (10.0, "vica_tts_node", "재생[response]: 여기서 대기할까요?")
    a = (14.0, "vica_llm_intent_node", "입력='네' -> intent=affirm matched= safety=normal")
    assert len(extract_rows([q, a, (14.3, *a[1:])])) == 1

def test_ambient_lines_are_not_questions():
    lines = [(10.0, "vica_tts_node", "재생[ambient]: 비카가 대기 중입니다."),
             (12.0, "vica_llm_intent_node", "입력='네' -> intent=affirm matched= safety=normal")]
    assert extract_rows(lines) == []
```

- [ ] **Step 2: 실행해 실패 확인** — `.venv/bin/python -m pytest tests/test_answer_audit.py -v` → FAIL(ModuleNotFoundError `tools.answer_audit`)

- [ ] **Step 3: `tools/answer_audit.py` 구현**

`~/vica_data/voice_answer_audit_20261113/extract_qa.py`·`flag_qa.py` 의 규칙을 옮긴다(질문 = 40 s 안 마지막 비-ambient 재생 줄이 "?"·"말씀해 주세요"로 끝남, 미션 반응 = 같은 intent 의 "음성 요청" 줄 −0.5~+3 s, `[RT]`·`[A/B]` 줄 붙이기). 더할 것: `dest_yaml` = 그 행 앞 마지막 "목적지 N개 로드:" 경로, `dialog_state` = 미션 반응 줄의 앞 상태, 중복 = (ts 0.5 s 반올림, 글자). kind 규칙은 `flag_qa.qtype` 을 위 표에 맞게 고친다("확인을 위해 한 번 더" → `open`, 본인 질문·직원 도움·취소 확인 → `yesno`).

- [ ] **Step 4: 시험 통과 확인** — 같은 명령 → PASS

- [ ] **Step 5: 실제 로그로 뽑기** — `.venv/bin/python -m tools.answer_audit --since 2026-10-10 --out ~/vica_data/voice_answer_audit_20261113/qa_rows.csv` → 화면에 행 수(조사 때 1,660 안팎)와 kind 별 개수. `dest_yaml` 이 빈 행 수도 찍는다.

- [ ] **Step 6: 커밋** — `git add tools/answer_audit.py tests/test_answer_audit.py && git commit -m "feat(tools): answer_audit — 실기 로그에서 로봇 질문·대답 행 뽑기(설계 11-14 5절)"`

### Task 2: 채점 — 지금 의도를 대답으로 옮기기 · 잘못된 행동 판정

**Files:**
- Create: `tools/answer_grade.py`
- Test: `tests/test_answer_grade.py`

**Interfaces:**
- Consumes: `kind_from_text` (Task 1)
- Produces:
  - `answer_of(intent: VicaIntent, kind: str, question: str, destinations: Sequence[DestinationData]) -> tuple[str, str]` — (answer, value). 지금 구조의 의도를 설계 4.2 대답으로 옮기는 기준선 전용 어댑터
  - `grade(label: str, label_value: str, pred: str, pred_value: str) -> str` — `ok` · `reask` · `wrong`

- [ ] **Step 1: 실패하는 시험 쓰기**

`answer_of` 표(질문 속 목적지 = 질문 글자에 든 가장 긴 목적지 이름):

```python
@pytest.mark.parametrize("intent,kind,question,expect", [
    (VicaIntent(intent="affirm"), "yesno", "여기서 대기할까요?", ("yes", "")),
    (VicaIntent(intent="deny"), "yesno", "여기까지 안내를 마칠까요?", ("no", "")),
    (VicaIntent(intent="navigate", matched_destination_id="d1", need_confirm=False), "confirm", "사회복지로 안내해드릴까요?", ("yes", "")),
    (VicaIntent(intent="navigate", matched_destination_id="d2", need_confirm=True), "confirm", "사회복지로 안내해드릴까요?", ("destination", "d2")),
    (VicaIntent(intent="navigate", matched_destination_id="d1", need_confirm=False), "yesno", "아까 사회복지에 가자고 하신 사용자님이신가요?", ("destination", "d1")),
    (VicaIntent(intent="wait", wait_minutes=10), "minutes", "몇 분쯤 걸리실까요?", ("minutes", "10")),
    (VicaIntent(intent="cancel"), "yesno_inverted", "안내가 필요 없으신가요?", ("yes", "")),
    (VicaIntent(intent="cancel"), "yesno", "여기까지 안내를 마칠까요?", ("yes", "")),
    (VicaIntent(intent="cancel"), "yesno", "여기서 대기할까요?", ("other", "")),
    (VicaIntent(intent="unknown"), "yesno", "여기서 대기할까요?", ("unclear", "")),
    (VicaIntent(intent="clarify"), "open", "네?", ("unclear", "")),
    (VicaIntent(intent="question"), "yesno", "안내를 받으시겠어요?", ("other", "")),
    (VicaIntent(intent="pause"), "minutes", "몇 분쯤 걸리실까요?", ("other", "")),
])
def test_answer_of(intent, kind, question, expect, dests):   # dests: d1=사회복지, d2=화장실
    assert answer_of(intent, kind, question, dests) == expect

@pytest.mark.parametrize("label,lv,pred,pv,out", [
    ("yes", "", "yes", "", "ok"), ("minutes", "10", "minutes", "10", "ok"),
    ("minutes", "10", "minutes", "15", "wrong"),
    ("no", "", "yes", "", "wrong"),                 # 정반대 (11-06 11:32 "아니" → 종료)
    ("yes", "", "destination", "d1", "wrong"),      # 엉뚱한 출발
    ("yes", "", "unclear", "", "reask"),
    ("no", "", "not_for_robot", "", "wrong"),       # 조용히 버림
    ("not_for_robot", "", "other", "", "wrong"),    # 딴말에 대꾸
    ("not_for_robot", "", "unclear", "", "reask"),
    ("repeat", "", "unclear", "", "reask"),
    ("unclear", "", "unclear", "", "ok"),
])
def test_grade(label, lv, pred, pv, out):
    assert grade(label, lv, pred, pv) == out
```

- [ ] **Step 2: 실행해 실패 확인** — `.venv/bin/python -m pytest tests/test_answer_grade.py -v` → FAIL(모듈 없음)
- [ ] **Step 3: `tools/answer_grade.py` 구현** — `answer_of` 는 위 표의 순서대로 규칙을 적용한다(`cancel`·`finish` 는 `yesno_inverted` 이거나 질문에 "마칠까요"가 있으면 `yes`, 아니면 `other`; `wait` 은 `minutes` kind 에서 분이 있으면 `minutes`, `yesno` 에서는 `yes`). `grade` 는 같으면 `ok`(값 있는 대답은 값까지 같아야), 예측이 `unclear` 면 `reask`(정답이 `unclear` 면 `ok`), 그 밖 모두 `wrong`.
- [ ] **Step 4: 시험 통과 확인** — 같은 명령 → PASS
- [ ] **Step 5: 커밋** — `git add tools/answer_grade.py tests/test_answer_grade.py && git commit -m "feat(tools): answer_grade — 지금 의도→대답 옮기기·잘못된 행동 채점"`

### Task 3: 정답 묶음 초안

**Files:**
- Create: `tools/answer_label.py`
- Test: `tests/test_answer_label.py`

**Interfaces:**
- Consumes: Task 1 행 dict, `src.handle_mode.AFFIRMATIVES`·`NEGATIVES`, `src.langchain_intent_parser.parse_wait_minutes`, `src.destination_loader.load_voice_destinations`
- Produces:
  - `select_rows(rows: list[dict], flagged_keys: set[tuple[str, str]], per_kind: int = 30, seed: int = 20261114) -> list[dict]` — 의심 행 전부 + 나머지에서 kind 별 `per_kind` 개
  - `draft_label(row: dict, destinations: Sequence[DestinationData]) -> tuple[str, str, str]` — (answer, value, source) · source = `rule` 또는 `todo`
  - CLI `python -m tools.answer_label --rows .../qa_rows.csv --flagged .../qa_flagged.csv --out .../labeled_draft.csv`

- [ ] **Step 1: 실패하는 시험 쓰기**

```python
@pytest.mark.parametrize("text,kind,expect", [
    ("비카야", "confirm", ("repeat", "", "rule")), ("피카야.", "yesno", ("repeat", "", "rule")),
    ("네. 네.", "yesno", ("yes", "", "rule")), ("그래 맞아", "yesno", ("yes", "", "rule")),
    ("아뇨.", "yesno", ("no", "", "rule")), ("아니 사회복지로 가자", "yesno", ("destination", "d1", "rule")),
    ("십 분.", "minutes", ("minutes", "10", "rule")), ("한 오 분.", "minutes", ("minutes", "5", "rule")),
    ("", "yesno", ("unclear", "", "rule")),
    ("급정지 되면은 그 사용감이 안 좋아서요.", "yesno", ("", "", "todo")),
])
def test_draft_label(text, kind, expect, dests):
    assert draft_label({"answer_text": text, "kind": kind}, dests) == expect

def test_select_rows_is_stable_and_keeps_all_flagged(rows_200):
    flagged = {(r["date"], r["time"]) for r in rows_200[:20]}
    a, b = select_rows(rows_200, flagged, per_kind=5), select_rows(rows_200, flagged, per_kind=5)
    assert a == b and flagged <= {(r["date"], r["time"]) for r in a}
```

- [ ] **Step 2: 실행해 실패 확인** — `.venv/bin/python -m pytest tests/test_answer_label.py -v` → FAIL
- [ ] **Step 3: `tools/answer_label.py` 구현** — 규칙 순서: 빈 글자 → `unclear`; 문장부호 뺀 글자가 비카야·피카야·지카야 → `repeat`; 첫 낱말 부정 + 목적지 이름·별칭 하나 → `destination`; 첫 낱말 부정 → `no`; 낱말 전부 긍정 → `yes`; kind `minutes` 이고 `parse_wait_minutes(text, 1.0)` 값 → `minutes`; 목적지 하나만 맞음 → `destination`; 그 밖 `todo`. 목적지는 행의 `dest_yaml` 로 읽고, 파일이 없으면 목적지 규칙만 건너뛴다. 출력 열 = Task 1 열 + `label_answer` `label_value` `label_source` `note`(빈칸).
- [ ] **Step 4: 시험 통과 확인** — 같은 명령 → PASS
- [ ] **Step 5: 초안 만들기** — `.venv/bin/python -m tools.answer_label --rows ~/vica_data/voice_answer_audit_20261113/qa_rows.csv --flagged ~/vica_data/voice_answer_audit_20261113/qa_flagged.csv --out ~/vica_data/voice_answer_audit_20261113/labeled_draft.csv` → 행 수와 `todo` 수 출력
- [ ] **Step 6: 커밋** — `git add tools/answer_label.py tests/test_answer_label.py && git commit -m "feat(tools): answer_label — 정답 묶음 초안(의심 행 전부 + kind 별 표본)"`

### Task 4: 재생 도구 — 경로별 기준 점수

**Files:**
- Create: `tools/answer_replay.py`
- Test: `tests/test_answer_replay.py`

**Interfaces:**
- Consumes: `answer_of`·`grade` (Task 2), 정답 CSV(Task 3 열, `label_answer` 가 채워진 행만)
- Produces:
  - `replay(rows: list[dict], interpret: Callable[[dict], tuple[VicaIntent | None, float]], destinations_for: Callable[[str], Sequence[DestinationData] | None]) -> dict` — 행마다 `status`(`ok`·`reask`·`wrong`·`skip`·`error`)·`pred`·`latency`, 요약 `counts`·`by_kind`·`p50`·`p90`
  - `clip_for(row: dict, clips: list[dict], clips_dir: Path, window_s: float = 3.0) -> Path | None` — `~/vica_data/listen_clips/clips.jsonl` 에서 같은 말의 wav. clips 줄의 실제 칸: `time`(현지 ISO 문자열 "2026-11-11T21:47:44.346") · `file`(clips_dir 안 wav 이름) · `result`("text" 만 본다) · `text`
  - CLI `python -m tools.answer_replay --path midm|gpt|realtime --labels .../labeled.csv [--rounds 3]` → `.../replay_<path>_<YYYYMMDD-HHMM>.json`·`.md`

- [ ] **Step 1: 실패하는 시험 쓰기** (LLM 없이 가짜 `interpret` 를 넣는다)

```python
def test_replay_counts_and_kinds(rows_labeled, dests):
    fake = lambda row: (VicaIntent(intent="affirm"), 1.2)
    out = replay(rows_labeled, fake, lambda _p: dests)
    assert set(out["counts"]) <= {"ok", "reask", "wrong", "skip", "error"} and out["p90"] >= 1.2

def test_missing_destination_file_is_skip(rows_labeled):
    out = replay(rows_labeled[:1], lambda r: (VicaIntent(intent="affirm"), 1.0), lambda _p: None)
    assert out["rows"][0]["status"] == "skip"

def test_llm_unavailable_is_error_and_stops_after_five(rows_labeled, dests):
    down = lambda r: (VicaIntent(intent="unknown", reply=LLM_UNAVAILABLE), 0.1)
    out = replay(rows_labeled * 3, down, lambda _p: dests)
    assert out["counts"]["error"] == 5 and out["stopped"] is True

def test_empty_answer_never_calls_llm(dests):
    called = []
    row = {"answer_text": "", "kind": "yesno", "question": "여기서 대기할까요?", "label_answer": "unclear",
           "label_value": "", "dest_yaml": "x"}
    out = replay([row], lambda r: called.append(r) or (None, 0.0), lambda _p: dests)
    assert called == [] and out["rows"][0]["status"] == "ok"

def test_clip_for_matches_time_and_text(tmp_path):
    clips = [{"time": "2026-11-11T21:47:44.346", "file": "a.wav", "result": "text", "text": "그래."}]
    ts = datetime(2026, 11, 11, 21, 47, 45, 800000).timestamp()
    assert clip_for({"ts": ts, "stt": "그래."}, clips, tmp_path) == tmp_path / "a.wav"
    assert clip_for({"ts": ts + 10, "stt": "그래."}, clips, tmp_path) is None
```

시험 파일 안 고정값: `dests` = `DestinationData(id="d1", name="사회복지")`·`(id="d2", name="화장실")`, `rows_labeled` = Task 3 열을 가진 3행(yesno·minutes·confirm, 정답 채움), `rows_200` = kind 를 돌려 가며 만든 200행. 각 시험 파일에 둔다.

- [ ] **Step 2: 실행해 실패 확인** — `.venv/bin/python -m pytest tests/test_answer_replay.py -v` → FAIL
- [ ] **Step 3: `tools/answer_replay.py` 구현**
  - `--path` 를 먼저 읽고 **`src` 를 import 하기 전에** 환경을 덮는다(`scripts/local_llm_bench/bench.py`·`~/vica_data/qwen_field/model_test.py` 와 같은 이유): midm = `VICA_LLM_PROVIDER=ollama`·`OLLAMA_HOST=http://localhost:11434`·모델 `midm2-mini`; gpt = `VICA_LLM_PROVIDER=openai`·모델 `gpt-5.4-mini`; 모두 `VICA_LLM_FALLBACK_MODEL=""`(다른 경로가 섞이지 않게), `VICA_INTENT_INPUT` 은 midm·gpt = `text`, realtime = `audio`.
  - 글자 경로 `interpret`: `parse_intent(row["answer_text"], dests, history=[AIMessage(row["question"])], robot_state=RobotState(dialog_state=row["dialog_state"]), model=...)`, 벽시계 초. 목적지는 `load_voice_destinations(row["dest_yaml"])`(노드와 같은 함수), 파일 없으면 `None`.
  - realtime `interpret`: `clip_for` 로 wav 를 찾아 16 kHz pcm16 → `parse_intent_audio(pcm, dests, history, robot_state)`; wav 없으면 `skip`.
  - `reply == LLM_UNAVAILABLE` → `error`; `error` 가 연속 5건이면 멈추고 `stopped=True`.
  - `--rounds N`: 같은 묶음을 N 번 돌려 행마다 결과를 모으고, 요약은 회차별 `wrong` 수와 "한 번이라도 wrong" 행 수(설계 C9).
  - `.md` 에는 kind 별 표(ok·reask·wrong)와 wrong 행 목록(질문·대답·예측·정답)을 쓴다.
- [ ] **Step 4: 시험 통과 확인** — 같은 명령 → PASS, 이어서 전체 `.venv/bin/python -m pytest -q` → 기존 시험 포함 통과
- [ ] **Step 5: 커밋** — `git add tools/answer_replay.py tests/test_answer_replay.py && git commit -m "feat(tools): answer_replay — 정답 묶음을 경로별로 다시 넣어 기준 점수(midm·gpt·realtime)"`

### Task 5: 정답 확인 · 기준 점수 재기 (실행 작업, 코드 없음)

- [ ] **Step 1:** `labeled_draft.csv` 의 `todo` 행과 의심 행 전부에 Claude 가 정답 초안을 채워 `labeled.csv` 로 저장(초안 파일은 그대로 둔다). 판단이 갈리는 행은 `note` 에 이유.
- [ ] **Step 2:** 사용자에게 `labeled.csv` 확인을 받는다 — 확인 전엔 Step 3 결과를 기준 점수로 부르지 않는다.
- [ ] **Step 3:** 로봇이 주행 중이 아님을 확인하고 Ollama 를 launch 와 같은 환경(`OLLAMA_GO_TEMPLATE=1`)으로 띄워 midm2-mini 를 `keep_alive -1` 로 올린 뒤:
  `.venv/bin/python -m tools.answer_replay --path midm --labels ~/vica_data/voice_answer_audit_20261113/labeled.csv --rounds 3`
  `.venv/bin/python -m tools.answer_replay --path gpt --labels ... --rounds 3`
  `.venv/bin/python -m tools.answer_replay --path realtime --labels ... --rounds 1` (wav 있는 10-21 이후 행만, 비용 때문에 1회)
- [ ] **Step 4:** `~/vica_data/voice_answer_audit_20261113/README.md` 에 "기준 점수(11-xx)" 절: 경로별 wrong·reask 수, kind 별 표, 대표 wrong 5건, 지연 p50·p90. 2~6단계 계획은 이 숫자를 보고 쓴다.

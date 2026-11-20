# 음성 원칙에 비춘 의도 분류 재단 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 설계서 6절의 여덟 군데를 네 단계로 고쳐, 원칙 31개와 어긋나는 말의 종류 11가지 중 10가지를 원칙대로 처리하게 한다.

**Architecture:** 지금 구조(LLM 이 `VicaIntent` 를 내고 미션 상태기계가 반응)는 그대로 둔다. LLM 이 고르는 값에 "나에게 한 말이 아님"·"대화 끝내기"를 더하고, 되묻는 문장은 LLM 이 짓지 않고 코드의 틀에서 만든다. 대답 처리·호출 뒤 처리·되묻기 사다리는 미션에 둔다. 단계마다 스위치를 두고 꺼져 있으면 지금과 같다.

**Tech Stack:** Python 3.10 · pydantic v2 · LangChain(OpenAI `gpt-5.4-mini` · Ollama `midm2-mini`) · OpenAI Realtime · ROS 2 Humble(`vica_interfaces`, `vica_mission_manager`) · pytest

**Spec:** `docs/superpowers/specs/2026-11-20-voice-principles-intent-recut-design.md` (원칙의 정본은 사용자 문서 '비카 음성 원칙'). 6.6 은 `docs/superpowers/specs/2026-11-18-answer-interpretation-design.md` 4.4·4.5, 6.8 은 `docs/superpowers/specs/2026-11-14-question-aware-answers-design.md` D5·4.5 를 함께 읽는다.

**이 계획이 정한 이름** (설계 9절이 빌더에게 맡긴 것)

| 무엇 | 이름 | 단계 |
|---|---|---|
| 음성 스위치 | `.env` `VICA_RECUT=on` | 2 |
| 음성 스위치 | `.env` `VICA_ANSWER=on`(이미 있음) · `VICA_ANSWER_LOCAL=on`(로컬도 대답 칸을 채움, 기본 꺼짐) | 3 |
| 미션 파라미터 | `answer_field` · `reask_ladder` · `wake_revive` (bool, 기본 false) | 3 |
| 음성 스위치 | `.env` `VICA_TALK=on` | 4 |
| LLM 이 고르는 새 값 | `not_for_robot` · `dismiss` | 2 |
| LLM 이 고르는 새 값 | `repeat` · `places` | 4 |
| 미션이 받는 새 값 | `dismiss`(2) · `repeat`(4). `not_for_robot` 과 `places` 는 음성에서 끝난다 | |
| LLM 출력 새 칸 | `clarify_kind`(`several`·`not_available`·`no_destination`) · `clarify_options: list[str]` · `clarify_wanted: str` | 2 |
| 새 문장(사용자 확정) | "제가 아는 목적지 중에서 ○○은(는) 없는데, 어디로 안내해드릴까요?" · "네, 필요하시면 '비카야'라고 불러 주세요." · "죄송합니다. 더 크게, 또박또박 말씀해 주세요." · "○○, ○○, ○○(…등)으로 안내할 수 있습니다." | 2·3·4 |

## Global Constraints

- 저장소 둘, 브랜치는 둘 다 `after_feedback_waiting`: 음성 `~/VICA-smarthandle/vica-voice-llm`, 미션 `~/VICA-smarthandle/vica_ros2_ws`. 새 브랜치를 만들지 않는다. **푸시하지 않는다**(사용자가 다시 말할 때까지).
- 운영 장비에서 소스로 바로 돈다. 스위치와 파라미터의 기본값은 꺼짐이고, 꺼진 상태의 동작과 지시문 글자는 지금과 같아야 한다. `.env` 와 launch 값은 사용자 승인 뒤에만 바꾼다.
- 과제를 시작할 때 `git status --short` 로 다른 세션의 변경을 확인하고 건드리지 않는다. 커밋은 `git commit -m "…" -- <내 파일>` 로 한다. 끝줄 `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.
- 커밋 문체는 기존과 같다: `feat(voice): …` · `feat(mission): …` · `test(...)` · `docs(...)`, 한국어.
- 주행 묶음(`vica-drive.target`)이 켜져 있으면 Ollama 재생·전체 시험·`colcon build` 를 돌리지 않는다. 상태 확인과 실행을 한 명령 줄에 넣지 않는다. 조건이 풀리면 스스로 시작하는 대기·감시를 걸지 않는다. 묶음을 직접 껐으면 끝난 뒤 다시 켠다.
- 자동 기동 유닛을 직접 stop/restart 하지 않는다. 노드 재기동이 필요하면 사용자에게 요청한다.
- 외부 API(gpt·Realtime) 측정은 돌릴 때마다 사용자 승인을 받는다. 사용자·프로젝트 정보는 외부로 보내지 않는다.
- 사람 말의 뜻을 정하는 글자 규칙(낱말 목록·정규식)을 새로 만들지 않는다. 코드에 두는 것은 로봇이 말할 문장의 틀과 상태 확인뿐이다.
- 새 멘트는 위 표의 네 문장만 넣는다. 그 밖의 문장이 필요해지면 멈추고 사용자에게 묻는다.
- 고치는 동안 안 본 말(`answer_key.csv` 의 `set=heldout`)의 행 내용을 읽지 않는다. 숫자와 깨진 원칙 번호만 본다.
- 원칙 번호를 이 계획이 손대는 줄의 주석(`# 원칙 7`)과 시험 이름(`test_p7_…`)에 적는다(설계 G3).
- 시험 실행
  - 음성: `cd ~/VICA-smarthandle/vica-voice-llm && nice -n 19 .venv/bin/python -m pytest <파일> -q -p no:cacheprovider`
  - 미션: `source /opt/ros/humble/setup.bash && source ~/VICA-smarthandle/vica_ros2_ws/install/setup.bash && cd ~/VICA-smarthandle/vica_ros2_ws/src/vica_mission_manager && python3 -m pytest <파일> -q`
  - 각 저장소의 첫 과제 전에 전체 시험을 한 번 돌려 통과·실패 수를 기준으로 적는다(미션은 11-17 에 1611 통과·4 실패였다). 과제마다 그 수가 줄지 않아야 한다.
- 줄 번호는 음성 `3c9e11d`, 미션 `7c41522` 기준이다. 어긋나면 이름으로 찾는다.

## Review Focus

- 외부 API 가 새 칸을 null·문자열·목록 밖 값으로 낸다 → 발화 하나가 통째로 실패하지 않고 빈 값으로 읽힌다 → Task 2-2 `test_recut_draft_tolerates_bad_values`
- `clarify_wanted` 에 긴급어나 긴 문장이 들어온다 → 로봇이 그 말을 소리 내면 긴급어 감시가 자기 목소리에 걸린다. "말씀하신 곳"으로 바꾼다 → Task 2-1 `test_p11_not_available_never_speaks_emergency_words`
- 되돌리기 어려운 상태(확인 질문 중·주행 중)에 `dismiss` 가 온다 → 확인 없이 접거나 취소하지 않는다 → Task 2-6 반응표 `bye` 열
- 늦게 온 대답(메시지의 `question_id` 가 지금 열린 질문과 다름) → 대답으로 처리하지 않고 지금처럼 의도로 처리한다 → Task 3-2 `test_stale_question_id_is_not_an_answer`
- "비카야"만 들림·나에게 한 말이 아님·"다시 말해줘" → 되묻기 횟수에 세지 않고, 알아들은 말이 오면 횟수가 0 으로 돌아간다 → Task 3-6 `test_p8_understood_utterance_resets`, Task 3-9 `test_p13_wake_only_is_not_counted_in_arrival_question`

---

## 단계 닫기 (2·3·4단계의 마지막 과제가 이 절차를 따른다)

1. **본 말로 고친다.** 스위치를 켠 재생을 `set=seen` 행에 돌린다(gpt·Realtime 은 승인 뒤, Mi:dm 은 주행 묶음이 꺼졌을 때). 맞지 않은 행을 읽고 지시문 문장만 고친다. 코드 틀과 스키마를 고쳐야 하면 해당 과제로 돌아간다.
2. **고정한다.** 커밋한 뒤에는 3~5번이 끝날 때까지 지시문을 고치지 않는다.
3. **안 본 말로 잰다.** `set=heldout` 행에 같은 재생을 돌린다. 결과 파일의 행 내용은 열지 않는다.
4. **실패 기준과 견준다**(설계 3절). 기준값은 Task 1-4 의 `key_baseline.md` 다.

   | 항목 | 실패 |
   |---|---|
   | 안 본 말의 잘못된 행동(`wrong`) | 경로별 기준보다 많음 |
   | 진짜 요청을 버림(`wrong` 중 로봇 반응이 `silent`) | 경로별 기준보다 많음 |
   | 확인 없이 출발(`go`) | 1건 이상 |
   | 해석 지연 p50 | 기준보다 0.1 s 넘게 늘어남 |
   | 음성·미션 전체 시험 | 통과 수가 줄어듦 |

5. **실기와 판정용 말 보충**(설계 7절 P6). 사용자가 장면표대로 실기한다. 그 기록에서 로봇 질문 뒤 말 30행 안팎을 `extract_qa.py` 방식으로 뽑아 `review2` 사이트에 올리고, 사용자가 귀로 확인한다. 그 행을 `answer_key.csv` 에 `set=heldout` 으로 보태 3번을 한 번 더 돌린다.
6. **보고하고 멈춘다.** 숫자·깨진 원칙 번호·실기에서 본 것을 사용자에게 보고한다. 실패 기준에 걸리면 다음 단계로 가지 않는다. 스위치를 운영 `.env`·launch 에 켤지는 사용자가 정한다. 단계가 닫히면 그 단계의 `heldout` 행을 `seen` 으로 바꾼다.

---

# 1단계 — 답지를 새 형식으로, 지금 구조 다시 재기 (제품 코드 0)

### Task 1-1: 새 답지 파일 만들기 (설계 8절 답지)

**Files:**
- Create: `~/vica_data/voice_answer_audit_20261113/review2/build_key.py`
- Output: `~/vica_data/voice_answer_audit_20261113/review2/answer_key.csv` · `people_talk_labels.csv`

**Interfaces:**
- Consumes: `review2/draft_rows.json`(행 `no`·`draft`·`old_label`·`dup`·`flags`) · `review2/answer_key_results.json`(`{"<no>": {verdict, action, target, minutes, say, drop_reason, memo}}`) · `labeled.csv`(행 번호 `no` → `ts`) · `~/vica_data/agent_spike_20261120/spike.py` 의 `load_rows()`·`tag_of()`
- Produces: `answer_key.csv` 열 `ts,date,time,set,action,target,minutes,say,source,drop,memo`. `people_talk_labels.csv` 는 `tools/answer_replay --labels` 가 읽는 열(`ts,date,time,question,kind,answer_text,stt,dialog_state,dest_yaml,label_answer,label_value`)을 가진다.

- [ ] **Step 1: `build_key.py` 를 쓴다.** 정하는 것:
  - `set` = 날짜가 `11-04`~`11-07` 이면 `seen`, 아니면 `heldout`.
  - 사용자 판정이 있는 행은 그 값을 쓰고 `source=user`. `drop_reason` 이 있으면 `drop` 에 적는다.
  - 판정이 없고 옛 정답이 `yes`·`no`·`destination`·`choice`·`minutes`·`repeat`·`not_for_robot` 이며 같은 이름 알림이 없는 행은 `draft` 를 그대로 쓰고 `source=rule`. 대응: yes→`yes` · no→`no` · destination→`confirm` · choice→`pick` · minutes→`wait` · repeat→`wake` · not_for_robot→`silent`.
  - 그 밖(옛 정답 `unclear`·`other`, 같은 이름이 걸린 행)은 `source=pending`.
  - `spike.load_rows()` 가운데 `tag_of(rec) == "사람끼리"` 인 21행을 `people_talk_labels.csv`(`label_answer=not_for_robot`)로 쓰고, `answer_key.csv` 에 `action=silent, set=seen, source=rule` 로 보탠다(`ts` 가 겹치면 한 번만).
  - 끝에 `set`·`action`·`source` 별 건수와 `pending` 행 번호를 찍는다.
- [ ] **Step 2: 실행한다.** `cd ~/VICA-smarthandle/vica-voice-llm && .venv/bin/python ~/vica_data/voice_answer_audit_20261113/review2/build_key.py`
  Expected: `rule` 이 182행 안팎, `pending` 이 26행 이하, 사람끼리 21행.

### Task 1-2: 확인 사이트가 남은 행만 보이게 하고 사용자 확인을 받는다

**Files:**
- Modify: `~/vica_data/voice_answer_audit_20261113/review2/review_site2.py`

**Interfaces:**
- Consumes: Task 1-1 의 `answer_key.csv`(`source=pending` 행)
- Produces: `answer_key_results.json` 에 남은 행의 판정. 행동 목록에 `bye`("대화를 끝내는 말로 받기") 추가.

- [ ] **Step 1:** 첫 화면의 목록을 `source=pending` 행만으로 바꾸고 맨 위에 "남은 N행"을 적는다. 전체 보기는 링크로 남긴다. `ACTIONS` 에 `"bye": "대화를 끝내는 말로 받기"` 를 더한다.
- [ ] **Step 2:** 서버를 다시 띄운다(`pgrep -f review_site2.py` 로 pid 를 본 뒤 `kill <pid>`, `python3 review_site2.py`). 주소는 그대로 `http://100.110.180.95:8787/<token>/`.
- [ ] **Step 3: 사용자에게 남은 행 확인을 요청한다.** 끝나면 Task 1-1 Step 2 를 다시 돌린다.
  Expected: `pending` 0행. 0 이 아니면 그 행 번호를 사용자에게 알린다.

### Task 1-3: 새 채점기 (설계 G2 · 8절 측정 규칙)

**Files:**
- Create: `tools/key_grade.py`
- Test: `tests/test_key_grade.py`

**Interfaces:**
- Consumes: `src.schema.VicaIntent` · `src.question_record.OpenQuestion` · `src.one_voice._CHOICE_TAIL` · `tools.answer_grade.question_destination_id`
- Produces:
  - `Reaction(action: str, target: str = "", minutes: str = "", say: str = "")` (frozen dataclass)
  - `reaction_of(intent, question: Optional[OpenQuestion], destinations: Sequence) -> Reaction`
  - `grade_key(key: dict, reaction: Reaction, kind: str = "") -> tuple[str, int]` — (`ok`·`reask`·`wrong`·`mismatch`·`drop`·`pending`, 깨진 원칙 번호 또는 0)
  - `load_key(path) -> dict[str, dict]` — `ts` → 답지 행

`reaction_of` 가 따르는 표(미션이 받는 의도 → 로봇이 하는 일):

| 의도 | 조건 | 행동 |
|---|---|---|
| `not_for_robot` · `dismiss` · `repeat` | | `silent` · `bye` · `repeat` |
| `affirm` · `deny` | | `yes` · `no` |
| `navigate` + 목적지 id | 열린 질문이 묻는 곳과 같음(kind `confirm`, 또는 `open` 인데 `question_destination_id` 가 그 id) | `yes` |
| | kind `choice` 이고 후보에 있음 | `pick`(target) |
| | 갈 수 없는 곳 | `notavail`(target) |
| | `need_confirm=True` / `False` | `confirm`(target) / `go`(target) |
| `wait` | kind `minutes` 이고 분이 있음 → `wait`(minutes) · kind `yesno` 이고 질문에 "마칠까요" → `no` · kind `yesno` → `yes` · 그 밖 `wait` | |
| `cancel` · `finish` | kind `yesno_inverted` 또는 질문에 "마칠까요" → `yes` · 그 밖 `finish` | |
| `clarify` | 문장(`ask_text` 없으면 `reply`)이 `_CHOICE_TAIL` 에 맞음 → `which` · "제가 아는 목적지 중에서"로 시작 → `notavail` · 그 밖 → `reask` | |
| `unknown` | 문장이 "네?" → `wake` · 문장 있음 → `reask` · 문장 없고 열린 질문이 있으며 그 글이 "네?"가 아님 → `reask`(미션이 같은 질문을 다시 함) · 그 밖 → `silent` | |
| `question` | | `answer`(say=reply) |
| `help` · `pause` · `resume` | | `etc` |

`grade_key` 규칙: `drop`·`pending` 행은 그대로 돌려준다 → 행동이 같고(`confirm`·`pick` 은 target, `wait` 는 답지에 분이 있으면 minutes 까지) → `ok` → 로봇 반응이 `reask` → `reask` → {답지, 반응} = {`yes`, `no`} → `wrong` → 반응이 `go`·`confirm`·`pick`·`wait`·`finish`·`bye`, 또는 `yes`(kind 가 `open` 이 아닐 때) → `wrong` → 반응이 `silent` 이고 답지가 `silent`·`reask`·`wake` 가 아님 → `wrong` → 그 밖 `mismatch`. 원칙 번호는 반응이 `go` 면 3, 아니면 답지 행동으로 정한다: `silent` 7 · `wake` 13 · `yes`·`no`·`pick`·`wait` 6 · `confirm` 5 · `which` 4 · `notavail` 11 · `reask` 8 · `finish` 3 · `answer` 25 · `bye` 14 · `repeat` 28 · 그 밖 0.

- [ ] **Step 1: 실패하는 시험 작성**

```python
def test_confirmed_departure_is_go_and_breaks_principle_3():
    r = reaction_of(nav(TOILET, need_confirm=False), None, DESTS)
    assert r == Reaction("go", TOILET.id)
    assert grade_key({"action": "confirm", "target": TOILET.id}, r) == ("wrong", 3)

def test_pick_from_choice_question():
    q = OpenQuestion("q", "choice", "남자 화장실과 여자 화장실 중 어디로 갈까요?", (MEN.id, WOMEN.id), 0.0)
    assert reaction_of(nav(MEN, need_confirm=True), q, DESTS) == Reaction("pick", MEN.id)

def test_clarify_sentences_split_into_which_notavail_reask():
    assert reaction_of(clarify("남자 화장실과 여자 화장실 중 어디로 갈까요?"), None, DESTS).action == "which"
    assert reaction_of(clarify("제가 아는 목적지 중에서 스무디는 없는데, 어디로 안내해드릴까요?"), None, DESTS).action == "notavail"
    assert reaction_of(clarify("어디로 안내해드릴까요?"), None, DESTS).action == "reask"

def test_silent_unknown_while_mission_question_open_is_reask():
    q = OpenQuestion("q", "confirm", "화장실로 안내해드릴까요?", (TOILET.id,), 0.0)
    assert reaction_of(VicaIntent(intent="unknown"), q, DESTS).action == "reask"
    assert reaction_of(VicaIntent(intent="unknown"), None, DESTS).action == "silent"

def test_dropping_a_real_request_is_wrong_and_talking_to_bystanders_breaks_7():
    assert grade_key({"action": "confirm", "target": "a"}, Reaction("silent")) == ("wrong", 5)
    assert grade_key({"action": "silent"}, Reaction("confirm", "a")) == ("wrong", 7)
    assert grade_key({"action": "silent"}, Reaction("reask")) == ("reask", 7)

def test_drop_and_pending_rows_are_not_graded():
    assert grade_key({"action": "yes", "drop": "mapping"}, Reaction("no"))[0] == "drop"
    assert grade_key({"action": "etc", "source": "pending"}, Reaction("no"))[0] == "pending"
```

- [ ] **Step 2:** Run `nice -n 19 .venv/bin/python -m pytest tests/test_key_grade.py -q -p no:cacheprovider` → FAIL(`tools.key_grade` 없음)
- [ ] **Step 3:** `tools/key_grade.py` 를 위 표와 규칙대로 쓴다.
- [ ] **Step 4:** 같은 명령 → PASS
- [ ] **Step 5:** `git commit -m "feat(tools): 새 답지 채점기 — 로봇이 한 일(행동·대상)을 답지와 견주고 깨진 원칙 번호를 낸다" -- tools/key_grade.py tests/test_key_grade.py`

### Task 1-4: 재생 도구에 새 답지 채점을 붙이고 지금 구조를 다시 잰다

**Files:**
- Modify: `tools/answer_replay.py`(`replay`·`_report_md`·`main`)
- Test: `tests/test_answer_replay.py`
- Output: `~/vica_data/voice_answer_audit_20261113/key_baseline.md`

**Interfaces:**
- Consumes: Task 1-3 의 `reaction_of`·`grade_key`·`load_key` · `src.one_voice.demote_unasked_confirm`·`hand_over_question`
- Produces: `--key <answer_key.csv>` 와 `--set seen|heldout|all`(기본 `all`). 결과 행에 `intent`·`dest`·`nc`·`reply`·`ask_text`·`reaction`(행동·대상·분·문장)·`key_status`·`principle`·`set` 이 더해진다. 요약에 `set` 별 건수(`ok`·`reask`·`wrong`·`mismatch`), `wrong` 중 반응 `silent` 건수, `go` 건수, 원칙 번호별 건수.

- [ ] **Step 1: 실패하는 시험 작성** — `replay(rows, interpret, destinations_for, key=...)` 에 가짜 `interpret` 를 넣는다.

```python
def test_key_grading_uses_what_the_mission_receives():
    # 확인 질문 없이 나온 확정 navigate 는 음성 노드가 제안으로 낮춘다(demote_unasked_confirm) — 반응은 go 가 아니라 confirm
    run = replay([ROW_NAV], lambda row: (nav(TOILET, need_confirm=False), 0.5), dests_for, key={ROW_NAV["ts"]: KEY_CONFIRM})
    assert run["rows"][0]["reaction"]["action"] == "confirm" and run["rows"][0]["key_status"] == "ok"

def test_heldout_rows_are_counted_but_not_listed_in_the_report():
    md = _report_md("gpt", [RUN_WITH_ONE_SEEN_WRONG_AND_ONE_HELDOUT_WRONG], setup="")
    assert SEEN_TEXT in md and HELDOUT_TEXT not in md and "heldout" in md
```

- [ ] **Step 2:** Run `… tests/test_answer_replay.py -q -p no:cacheprovider` → FAIL
- [ ] **Step 3:** 구현한다. 채점 전에 해석 결과에 `demote_unasked_confirm(intent, question, dests)` 와 `hand_over_question(intent, dests)` 를 차례로 적용해 미션이 받는 꼴로 만든다(음성 노드 `_publish_intent` 535·542 와 같은 순서). 보고서는 `seen` 행만 내용을 싣고 `heldout` 은 숫자만 싣는다.
- [ ] **Step 4:** 같은 명령 → PASS. 커밋: `feat(tools): 재생 도구에 새 답지 채점(--key·--set) — 미션이 받는 꼴로 채점, 안 본 말은 숫자만`
- [ ] **Step 5: 지금 구조를 다시 잰다.** 주행 묶음 상태를 먼저 확인한다. 입력은 `--labels ~/vica_data/voice_answer_audit_20261113/labels_208_confirmed.csv --key …/review2/answer_key.csv`.
  - Mi:dm: `--path midm --rounds 3`(외부 API 없음)
  - gpt: `--path gpt --rounds 3`(208행 × 3회, **승인 뒤**) — 11-18 기준 파일에는 로봇이 한 문장이 없어 되묻기 종류를 가를 수 없으므로 다시 돌린다.
  - Realtime: `--path realtime --rounds 1 --pause 1.0`(**승인 뒤**)
  - 사람끼리 21행: 세 경로 모두 `--labels …/review2/people_talk_labels.csv --rounds 1`
- [ ] **Step 6: 본 말에서 `ok` 가 아닌 행을 사람 눈으로 읽는다**(설계 8절: 채점 변환을 새로 만들면 숫자를 적기 전에 읽는다). 변환이 틀렸으면 Task 1-3 으로 돌아가 고치고 다시 채점한다.
- [ ] **Step 7:** `key_baseline.md` 에 경로별·`set` 별 숫자, `wrong` 중 `silent` 건수, p50 을 적고 사용자에게 보고한다. 옛 채점(안 본 말 `wrong` 7건)과 다른 까닭을 한 줄로 적는다.

---

# 2단계 — 말을 고르는 쪽 (설계 6.1 · 6.2 · 6.3)

### Task 2-1: 문장 틀과 스위치 (순수 함수)

**Files:**
- Create: `src/intent_recut.py`
- Modify: `src/replies.py`, `tests/conftest.py`(`_DEFAULTS` 에 `"VICA_RECUT": "off"`)
- Test: `tests/test_intent_recut.py`

**Interfaces:**
- Produces (`src/replies.py`):
  - `NOT_AVAILABLE = "제가 아는 목적지 중에서 {what}{josa} 없는데, 어디로 안내해드릴까요?"`
  - `UNNAMED_PLACE = "말씀하신 곳"`
  - `not_available(wanted: str) -> str`
- Produces (`src/intent_recut.py`):
  - `NOT_FOR_ROBOT = "not_for_robot"` · `DISMISS = "dismiss"` · `CLARIFY_KINDS = ("several", "not_available", "no_destination")`
  - `on() -> bool` — `.env` `VICA_RECUT=on`, 부를 때마다 읽는다(`answer_field.on` 과 같은 방식)
  - `clarify_reply(kind: str, wanted: str, names: Sequence[str]) -> str`
  - `clue_line(question: Optional[OpenQuestion]) -> str` · `with_clue(block: str, question: Optional[OpenQuestion]) -> str`
  - `speaker_line(verdict: str) -> str`
  - `PROMPT_INTENTS: str` · `PROMPT_ORDER: str`(Task 2-3 이 쓴다)

- [ ] **Step 1: 실패하는 시험 작성**

```python
def test_p11_not_available_sentence():
    assert not_available("스무디 파는 곳") == "제가 아는 목적지 중에서 스무디 파는 곳은 없는데, 어디로 안내해드릴까요?"
    assert not_available("카페") == "제가 아는 목적지 중에서 카페는 없는데, 어디로 안내해드릴까요?"
    assert not_available("") == "제가 아는 목적지 중에서 말씀하신 곳은 없는데, 어디로 안내해드릴까요?"

def test_p11_not_available_never_speaks_emergency_words():
    for wanted in ("멈춰 버튼", "정지하는 곳", "가" * 21, "스무디?\n파는 곳"):
        assert not_available(wanted).startswith("제가 아는 목적지 중에서 말씀하신 곳은")

def test_p4_clarify_reply_by_kind():
    assert clarify_reply("several", "", ["남자 화장실", "여자 화장실"]) == "남자 화장실과 여자 화장실 중 어디로 갈까요?"
    assert clarify_reply("several", "", ["남자 화장실"]) == ASK_DESTINATION        # 하나로는 고르기 질문이 안 된다 — 짐작하지 않는다
    assert clarify_reply("not_available", "스무디", []) == not_available("스무디")
    assert clarify_reply("no_destination", "", []) == ASK_DESTINATION
    assert clarify_reply("", "", []) == ASK_DESTINATION and clarify_reply("엉뚱", "", []) == ASK_DESTINATION

def test_p7_clue_line(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    wake = OpenQuestion("q1", "open", "네?", (), 0.0)
    asked = OpenQuestion("q2", "confirm", "화장실로 안내해드릴까요?", ("t",), 0.0)
    assert clue_line(None) == ""
    assert clue_line(wake) == '- 로봇이 방금 한 말: "네?" (사용자가 "비카야"라고 불러서 대답했다)'
    assert clue_line(asked) == '- 로봇이 방금 한 말: "화장실로 안내해드릴까요?" (확인 질문 — 대답을 기다리는 중)'
    assert with_clue("\n[지금 상황]\n- 층: 4층\n", asked).endswith(clue_line(asked) + "\n")
    assert with_clue("", asked).startswith("\n[지금 상황]\n")

def test_clue_is_off_by_default_and_yields_to_the_answer_line(monkeypatch):
    asked = OpenQuestion("q2", "confirm", "화장실로 안내해드릴까요?", ("t",), 0.0)
    assert with_clue("x", asked) == "x"                       # 꺼짐 — 지금과 같다
    monkeypatch.setenv("VICA_RECUT", "on"); monkeypatch.setenv("VICA_ANSWER", "on")
    assert with_clue("x", asked) == "x"                       # 대답 칸의 질문 줄이 같은 내용을 싣는다

def test_p19_speaker_line():
    assert speaker_line("pass") == "- 이 말의 목소리: 안내를 요청한 사용자와 같다"
    assert speaker_line("fail") == "- 이 말의 목소리: 안내를 요청한 사용자와 다르다"
    assert speaker_line("short") == "" and speaker_line("no_target") == "" and speaker_line("") == ""
```

- [ ] **Step 2:** Run `… tests/test_intent_recut.py -q -p no:cacheprovider` → FAIL
- [ ] **Step 3:** 구현한다. 정하는 것:
  - `not_available`: `wanted` 를 앞뒤 공백을 떼고 쓴다. 비었거나, 20자를 넘거나, 줄바꿈·물음표가 있거나, `emergency_filter.detect_emergency(wanted)` 가 값을 내면 `UNNAMED_PLACE` 를 쓴다. 조사는 `mission_phrases.josa_eun_neun`.
  - `clarify_reply`: `several` 은 이름이 둘 이상일 때만 `replies.choose_between(names)`.
  - `clue_line` 의 종류 이름은 `answer_field._KIND_KO` 를 쓴다(모르는 종류는 "질문").
  - `PROMPT_INTENTS`·`PROMPT_ORDER` 의 글은 Task 2-3 에 있다.
- [ ] **Step 4:** 같은 명령 → PASS. `… tests/test_spoken_text.py -q -p no:cacheprovider` 도 PASS(새 고정 문구에 긴급어 없음).
- [ ] **Step 5:** `git commit -m "feat(voice): 재단 스위치 VICA_RECUT·되묻는 문장 틀(여러 곳·없는 곳·목적지 안 말함)·[지금 상황] 단서 줄(기본 꺼짐)" -- src/intent_recut.py src/replies.py tests/conftest.py tests/test_intent_recut.py`

### Task 2-2: 출력 스키마에 새 값과 새 칸

**Files:**
- Modify: `src/schema.py:54-66`, `src/langchain_intent_parser.py:80-145`(`_IntentDraft`·`_draft_cls`)·`598-614`(`_ollama_structured`), `src/realtime_intent.py:30-31`·`96-124`
- Test: `tests/test_parser_recut.py`(새 파일), `tests/test_realtime_intent.py`

**Interfaces:**
- Consumes: `intent_recut.on`·`CLARIFY_KINDS`
- Produces:
  - `schema.BaseIntentType`(지금의 12개 값) · `schema.VicaIntentType`(12개 + `not_for_robot` + `dismiss`) — `VicaIntent.intent` 는 넓은 쪽, `_IntentDraft.intent` 는 `BaseIntentType`
  - `_RecutDraft(_IntentDraft)`: `intent: VicaIntentType` · `clarify_kind: str = ""`(enum `["", *CLARIFY_KINDS]`) · `clarify_options: list[str] = []` · `clarify_wanted: str = ""`
  - `_draft_cls()` 는 (`answer_field.on()`, `intent_recut.on()`) 네 조합의 클래스를 돌려준다. 로컬 `_ollama_structured` 와 `answer_field.narrowed` 의 바탕도 같은 선택을 따른다.
  - Realtime `build_intent_tool()`: 켜졌을 때 `intent` enum 에 두 값, 인자에 세 칸.

- [ ] **Step 1: 실패하는 시험 작성**

```python
def test_off_schema_is_unchanged():
    assert parser._draft_cls() is parser._IntentDraft
    assert "not_for_robot" not in json.dumps(parser._IntentDraft.model_json_schema())
    assert "clarify_kind" not in json.dumps(build_intent_tool())

def test_on_schema_offers_new_values_and_fields(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    props = parser._draft_cls().model_json_schema()["properties"]
    assert {"clarify_kind", "clarify_options", "clarify_wanted"} <= set(props)
    tool = build_intent_tool()["parameters"]["properties"]
    assert {"not_for_robot", "dismiss"} <= set(tool["intent"]["enum"])
    assert tool["clarify_kind"]["enum"] == ["", "several", "not_available", "no_destination"]

def test_recut_draft_tolerates_bad_values(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    d = parser._draft_cls()(intent="clarify", clarify_kind=None, clarify_options="남자 화장실", clarify_wanted=3)
    assert (d.clarify_kind, d.clarify_options, d.clarify_wanted) == ("", [], "")
    assert parser._draft_cls()(intent="clarify", clarify_kind="엉뚱").clarify_kind == "엉뚱"   # 값은 뒤처리가 no_destination 으로 읽는다

def test_both_switches_give_both_sets_of_fields(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on"); monkeypatch.setenv("VICA_ANSWER", "on")
    assert {"answer", "clarify_kind"} <= set(parser._draft_cls().model_json_schema()["properties"])
```

- [ ] **Step 2:** Run `… tests/test_parser_recut.py tests/test_realtime_intent.py -q -p no:cacheprovider` → FAIL
- [ ] **Step 3:** 구현한다. 새 칸은 `_AnswerDraft.answer` 처럼 `field_validator(mode="before")` 로 형이 다른 값을 빈 값으로 읽는다. `clarify_options` 의 설명은 "clarify_kind 가 several 일 때 목록의 name 을 모두", `clarify_wanted` 는 "clarify_kind 가 not_available 일 때 사용자가 찾는 것을 말한 낱말 그대로".
- [ ] **Step 4:** 같은 명령 → PASS. `… tests/test_parser_answer.py tests/test_parser_backend.py tests/test_help_request_voice.py -q -p no:cacheprovider` 도 PASS.
- [ ] **Step 5:** `git commit -m "feat(voice): 출력 스키마 — not_for_robot·dismiss 값과 clarify 상황 칸(VICA_RECUT 켰을 때만)" -- src/schema.py src/langchain_intent_parser.py src/realtime_intent.py tests/test_parser_recut.py tests/test_realtime_intent.py`

### Task 2-3: 지시문 두 벌을 같은 말로 고친다

**Files:**
- Modify: `src/intent_recut.py`(`PROMPT_INTENTS`·`PROMPT_ORDER`), `src/langchain_intent_parser.py:184-273`(`_build_system_prompt`)·`1114-1286`(`build_audio_prompt`)·`1306-1307`·`1327-1328`, `src/local_rules.py:33-40`
- Test: `tests/test_prompt_recut.py`(새 파일)

**Interfaces:**
- Consumes: Task 2-1 의 `on`·`with_clue`, Task 2-2 의 칸 이름
- Produces: 꺼졌을 때 두 지시문의 글자는 지금과 한 글자도 다르지 않다. 켜졌을 때 두 지시문이 `PROMPT_INTENTS`·`PROMPT_ORDER` 를 글자 그대로 싣는다.

두 지시문이 함께 싣는 글:

```text
PROMPT_INTENTS
- clarify: 로봇에게 한 말이고 가고 싶은 곳에 대한 말인데 한 곳으로 정할 수 없다. reply 는 빈 문자열로 두고(되묻는 문장은 시스템이 만든다) clarify_kind 에 까닭을 적는다.
  several = 목록의 여러 곳에 똑같이 걸린다. clarify_options 에 그 곳들의 name 을 모두 적는다(하나만 적지 않는다).
  not_available = 목록에 없는 곳을 찾는다. clarify_wanted 에 찾는 것을 사용자가 말한 낱말 그대로 적는다. 비슷한 다른 곳으로 바꾸지 않는다.
  no_destination = 어디로 갈지 말하지 않았다.
- unknown: 로봇에게 한 말인데 무슨 뜻인지 알아듣지 못했다. reply 는 빈 문자열 — 시스템이 다시 말해 달라고 한다.
- not_for_robot: 로봇에게 한 말이 아니다(옆 사람과 나눈 말, 로봇을 두고 설명하는 말, 혼잣말, 로봇 목소리의 메아리, 소음). reply 는 빈 문자열 — 로봇은 끼어들지 않는다.
- dismiss: 안내를 받고 있지 않은데 로봇과의 대화를 그만하려 한다("됐어", "아니야 괜찮아", "나중에 할게"). reply 는 빈 문자열.
  안내 중에 그만두려는 말은 cancel, 로봇이 방금 한 질문을 거절하는 말은 deny 다.

PROMPT_ORDER
[판단 순서] 들은 말을 이 순서로 판단하고, 앞에서 정해지면 뒤로 가지 않는다.
1. 로봇에게 한 말인가. 아니면 not_for_robot 이다. [지금 상황]의 '로봇이 방금 한 말' 줄을 함께 본다 — 로봇이 방금 부름에 대답했거나 질문했다면 이어진 말은 대개 로봇에게 한 말이다. 그래도 사람끼리 주고받는 말이거나 로봇을 가리켜 설명하는 말이면 not_for_robot 이다.
2. 로봇이 방금 한 질문의 대답인가. 대답이면 그 질문의 뜻으로 읽는다.
3. 새 요청인가. 무엇을 원하는지 읽는다. 가려는 곳이 목록에 없으면 clarify(not_available)다.
4. 로봇에게 한 말인데 알아듣지 못했으면 unknown 이다.
로봇에게 한 말인지 확실하지 않으면 not_for_robot 이 아니라 unknown 을 고른다 — 진짜 요청을 버리는 것이 되묻는 것보다 나쁘다.
```

켜졌을 때 바꾸는 줄:

| 지시문 | 줄 | 지금 | 켜졌을 때 |
|---|---|---|---|
| 글자 | 208 앞 | — | `PROMPT_ORDER` |
| 글자 | 211-212 | `clarify`·`unknown` 정의 | `PROMPT_INTENTS` |
| 글자 | 233 | "목록에 없으면 clarify" | "목록에 없으면 clarify(clarify_kind=not_available)" |
| 글자 | 238-240 | 여러 곳이면 clarify 로 name 을 모두 들어 묻는다(예문) | "…clarify(clarify_kind=several)로 내고 clarify_options 에 그 곳들의 name 을 모두 적는다" |
| 글자 | 241-243 | reply 를 비우는 의도 목록 / "그 외(question/clarify/unknown)의 reply 는…" | 목록에 clarify·unknown·not_for_robot·dismiss 를 더하고, 뒤 문장은 "question 의 reply 는…" |
| 글자 | 272 | "부정만 하고 목적지를 안 말하면 clarify." | "…clarify(clarify_kind=no_destination)." |
| 소리 | 1154-1158 | 사람 말이 아니면 intent=unknown, reply="" / "답이 꼭 필요한 자리면 clarify 로 되묻는다" | "…intent=not_for_robot" / "애매하면 confidence 를 낮추고 unknown 으로 둔다" |
| 소리 | 1158 뒤 | — | `PROMPT_ORDER` |
| 소리 | 1164-1167 | 여러 곳이면 clarify 로 name 을 모두 들어 묻는다(예문) | 글자 지시문과 같은 문장 |
| 소리 | 1189-1191 | 목록에 없는 곳이면 clarify 로 "…목록에 없는데, 학과사무실이 맞을까요?" | "clarify(clarify_kind=not_available)로 내고 clarify_wanted 에 그 말을 적는다. 비슷한 다른 곳을 권하지 않는다" |
| 소리 | 1231 | "약속 금지(question·clarify·unknown 공통…)" | "약속 금지(question 의 reply…)" |
| 소리 | 1240-1249 | `clarify` · `unknown + 침묵` 정의 | `PROMPT_INTENTS` + 한 줄 `호출어 "비카야"만 들리면 로봇을 부른 것이다 — intent=unknown, reply="네?".` |
| 소리 | 1251 | "목록에 없는 곳은 clarify" | "목록에 없는 곳은 clarify(clarify_kind=not_available)" |
| 소리 | 1270 | 숫자가 확실치 않으면 clarify 로 몇 번인지 되묻는다 | "…clarify(clarify_kind=several)로 내고 clarify_options 에 그 번호들의 name 을 적는다" |
| 소리 | 1284 | reply="" 로 정한 경우의 목록 | clarify·unknown·not_for_robot·dismiss 를 더한다 |
| 로컬 | `local_rules.py` 40 | 로봇에게 한 말이 아니면 intent 는 unknown | "…intent 는 not_for_robot" |

`answer_field.with_question` 을 부르는 세 곳(273 · 1307 · 1328)에서 그 앞에 `intent_recut.with_clue(<블록>, question)` 을 적용한다.

- [ ] **Step 1: 실패하는 시험 작성**

```python
def test_off_prompts_are_byte_identical(monkeypatch):
    before = (parser._build_system_prompt(DESTS), parser.build_audio_prompt(DESTS))
    monkeypatch.setenv("VICA_RECUT", "off")
    assert (parser._build_system_prompt(DESTS), parser.build_audio_prompt(DESTS)) == before
    assert "not_for_robot" not in before[0] and "not_for_robot" not in before[1]

def test_on_both_prompts_carry_the_same_principles(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    for prompt in (parser._build_system_prompt(DESTS), parser.build_audio_prompt(DESTS)):
        assert intent_recut.PROMPT_INTENTS in prompt and intent_recut.PROMPT_ORDER in prompt

def test_p12_on_prompts_stop_asking_the_model_to_write_reask_sentences(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    text, audio = parser._build_system_prompt(DESTS), parser.build_audio_prompt(DESTS)
    assert "reply 에 되묻는 질문을 담아라" not in text
    assert "reply 에 되묻는 질문(반드시" not in audio and "학과사무실이 맞을까요" not in audio
    assert "잘 못 들었는데, 다시 말씀해 주시겠어요" not in audio

def test_p7_clue_line_reaches_all_three_prompts(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    q = OpenQuestion("q", "confirm", "화장실로 안내해드릴까요?", ("t",), 0.0)
    assert intent_recut.clue_line(q) in parser._build_system_prompt(DESTS, question=q)
    # 소리 지시문과 재생용 글자 지시문은 parse_intent_audio · parse_intent_realtime_text 가 붙인다 — get_realtime_client 를
    # 가짜로 바꿔 넘어간 instructions 에 같은 줄이 있는지 본다(tests/test_parser_answer.py 의 FakeRT 방식)
```

- [ ] **Step 2:** Run `… tests/test_prompt_recut.py -q -p no:cacheprovider` → FAIL
- [ ] **Step 3:** 구현한다. 바뀌는 줄마다 f-string 안의 변수로 빼서 스위치로 고른다(지금 `first_line`·`reserve_rules` 를 고르는 방식).
- [ ] **Step 4:** 같은 명령 → PASS. 지시문 글자를 보는 기존 시험도 PASS: `… tests/test_parser_audio.py tests/test_prompt_rules_1011.py tests/test_mission_reactions_voice.py tests/test_local_rules.py tests/test_parser_answer.py tests/test_approach_question_voice.py -q -p no:cacheprovider`
- [ ] **Step 5:** `git commit -m "feat(voice): 지시문 두 벌 — 판단 순서와 clarify·unknown·not_for_robot·dismiss 를 같은 글로(VICA_RECUT 켰을 때만)" -- src/intent_recut.py src/langchain_intent_parser.py src/local_rules.py tests/test_prompt_recut.py`

### Task 2-4: 뒤처리 — 문장을 틀에서 만들고 새 값을 일찍 돌려보낸다

**Files:**
- Modify: `src/intent_recut.py`, `src/langchain_intent_parser.py:1385-1456`(`deliver_audio_draft`)·`1569-1710`(`_finalize`)
- Test: `tests/test_parser_recut.py`

**Interfaces:**
- Consumes: Task 2-1 의 `clarify_reply`, Task 2-2 의 `_RecutDraft`, `replies.RETRY_PROMPT`·`WAKE_GREETING`, 해석기의 `_WAKE_WORDS`·`_CHOICE_TAIL`
- Produces:
  - `intent_recut.early(draft) -> Optional[VicaIntent]` — `not_for_robot`·`dismiss` 면 `reply=""`·`need_confirm=False` 인 의도, 아니면 None
  - `intent_recut.settle(result: VicaIntent, draft, destinations: Sequence[DestinationData], heard: str) -> VicaIntent`

`settle` 규칙(켜졌을 때만, 긴급이면 그대로):
1. `result.intent == "clarify"`: reply 가 이미 고르기 문장(`_CHOICE_TAIL`)이면 그대로 둔다. 아니면 `clarify_options` 에서 목록의 name 과 글자가 같고 갈 수 있는 곳만 남긴다(겹침 제거). 둘 이상이면 `several`. 아니고 `draft.clarify_kind == "not_available"` 이거나 LLM 이 낸 의도가 `navigate` 였으면(목록에 없어 코드가 낮춘 것) `not_available` 이고 찾는 것은 `clarify_wanted or destination_candidate`. 그 밖은 `no_destination`. reply 는 `clarify_reply(...)`.
2. `result.intent == "unknown"`: `heard` 가 `_WAKE_WORDS` 에 있거나 reply 가 "네?"면 reply 를 `WAKE_GREETING` 으로, 아니면 `RETRY_PROMPT` 로 둔다.
3. 그 밖은 그대로.

- [ ] **Step 1: 실패하는 시험 작성** — `_finalize(_RecutDraft(...), …)` 와 `deliver_audio_draft` 를 직접 부른다(`tests/test_reply_fallback.py`·`tests/test_parser_audio.py` 방식).

```python
def test_p7_not_for_robot_and_dismiss_come_back_bare(monkeypatch):      # 두 경로 모두
    monkeypatch.setenv("VICA_RECUT", "on")
    for kind in ("not_for_robot", "dismiss"):
        out = finalize(draft(intent=kind, reply="아무 말", destination_candidate="화장실"))
        assert (out.intent, out.reply, out.need_confirm, out.matched_destination_id) == (kind, "", False, None)

def test_p4_several_uses_the_names_the_model_listed(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    out = finalize(draft(intent="clarify", clarify_kind="several", clarify_options=["남자 화장실", "여자 화장실", "없는 곳"], reply="화장실요?"))
    assert out.reply == "남자 화장실과 여자 화장실 중 어디로 갈까요?"

def test_p11_place_not_in_list(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    a = finalize(draft(intent="clarify", clarify_kind="not_available", clarify_wanted="스무디 파는 곳", reply="목록 중 어디로 갈까요?"))
    b = finalize(draft(intent="navigate", destination_candidate="스무디 가게"))          # 목록에 없어 코드가 clarify 로 낮춤
    assert a.reply == "제가 아는 목적지 중에서 스무디 파는 곳은 없는데, 어디로 안내해드릴까요?"
    assert b.intent == "clarify" and b.reply == "제가 아는 목적지 중에서 스무디 가게는 없는데, 어디로 안내해드릴까요?"

def test_p18_private_place_is_not_listed_as_an_option(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    out = finalize(draft(intent="clarify", clarify_kind="several", clarify_options=["407호", "비공개 408호"]))
    assert out.reply == ASK_DESTINATION

def test_p8_unknown_asks_again_on_both_paths_but_wake_stays(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    assert audio(draft(intent="unknown", reply=""), heard="으어").reply == RETRY_PROMPT
    assert audio(draft(intent="unknown", reply="네, 말씀하세요."), heard="비카야").reply == "네?"
    assert finalize(draft(intent="unknown", reply="제가 잘 모르겠어요")).reply == RETRY_PROMPT

def test_off_keeps_the_models_own_clarify_sentence():
    assert finalize(parser._IntentDraft(intent="clarify", reply="어느 방으로 갈까요?")).reply == "어느 방으로 갈까요?"
```

- [ ] **Step 2:** Run `… tests/test_parser_recut.py -q -p no:cacheprovider` → FAIL
- [ ] **Step 3:** 구현한다. `early` 는 두 함수의 맨 앞(목적지 맞추기 전)에서, `settle` 은 두 함수가 결과를 돌려주기 직전에 부른다. LLM 을 부르기 전의 지름길(`_shortcut_intent`·`_confirming_shortcut`)이 만든 `clarify` 는 이 두 함수를 거치지 않으므로 그대로다. 의도 값 일부만 고르는 기존 분기(해석기 342 · 1368 · 1436 · 1538 · 1596 · 1628 · 1684)는 새 두 값이 닿지 않으므로 고치지 않는다 — 닿는 곳이 있으면 `early` 의 자리를 앞으로 옮긴다.
- [ ] **Step 4:** 같은 명령 → PASS. `… tests/test_reply_fallback.py tests/test_parser_audio.py tests/test_destination_tie_1011.py tests/test_audio_shared_alias.py tests/test_text_shared_alias_1024.py tests/test_private_destinations.py -q -p no:cacheprovider` 도 PASS.
- [ ] **Step 5:** `git commit -m "feat(voice): 뒤처리 — 되묻는 문장을 틀에서 만들고 not_for_robot·dismiss 는 맨 앞에서 돌려보낸다(VICA_RECUT)" -- src/intent_recut.py src/langchain_intent_parser.py tests/test_parser_recut.py`

### Task 2-5: 음성 노드 — "나에게 한 말이 아님"은 미션으로 보내지 않는다

**Files:**
- Modify: `src/ros_node.py:462-510`(`_publish_intent`)·`667-724`(`_on_user_audio`)
- Test: `tests/test_recut_wiring.py`(새 파일, `tests/test_answer_wiring.py` 처럼 소스 글자와 순수 함수를 본다)

**Interfaces:**
- Consumes: `intent_recut.NOT_FOR_ROBOT`·`speaker_line`
- Produces: `_publish_intent` 가 `not_for_robot` 을 받으면 `[끼어들지 않음]` 로그 한 줄만 남기고 돌아간다 — 발행·TTS·대화 기록·`listen_request` 가 없다. 자리는 화자 판정을 붙인 뒤, `identity_answer` 앞이다(로그에 화자 판정이 남는다). 소리 경로는 `self._audio_spk` 가 `pass`·`fail` 일 때 `speaker_line` 을 `situation` 끝에 붙여 넘긴다.

- [ ] **Step 1: 실패하는 시험 작성**

```python
def test_p7_node_drops_not_for_robot_before_anything_is_published_or_said():
    body = method_source("_publish_intent")
    drop = body.index("NOT_FOR_ROBOT")
    assert drop < body.index("identity_answer") < body.index("_intent_pub.publish")
    assert "[끼어들지 않음]" in body

def test_p19_audio_path_passes_the_speaker_clue():
    body = method_source("_on_user_audio")
    assert body.index("speaker_line") < body.index("parse_intent_audio(")
```

- [ ] **Step 2:** Run `… tests/test_recut_wiring.py -q -p no:cacheprovider` → FAIL
- [ ] **Step 3:** 구현한다. 스위치 검사는 넣지 않는다 — 꺼져 있으면 `not_for_robot` 이 만들어지지 않고 `speaker_line` 을 붙이는 쪽만 `intent_recut.on()` 을 본다.
- [ ] **Step 4:** 같은 명령 → PASS. `… tests/test_one_voice_wiring.py tests/test_answer_wiring.py tests/test_relisten_drop_1024.py -q -p no:cacheprovider` 도 PASS.
- [ ] **Step 5:** `git commit -m "feat(voice): 의도 노드 — not_for_robot 은 로그만 남기고 미션으로 보내지 않음, 소리 경로에 화자 단서 줄" -- src/ros_node.py tests/test_recut_wiring.py`

### Task 2-6: 미션 — 대화 끝내기 (설계 6.3, 원칙 14)

**Files:**
- Modify: 미션 `vica_mission_manager/mission_logic.py`(`MSG_*` 694 부근 · `_react_voice_intent` 1977-2008), `test/reaction_states.py:295-306`, `test/test_reaction_table.py:54`·`56-345`
- Modify: 음성 `src/mission_phrases.py`, `tests/test_mission_phrases.py`
- Test: 미션 `test/test_reaction_table.py`, `test/test_spoken_text.py`

**Interfaces:**
- Produces:
  - 미션 `MSG_DISMISS_ACK = "네, 필요하시면 '비카야'라고 불러 주세요."`
  - 미션 `MissionLogic._dismiss_as(self, intent: IntentData) -> Optional[IntentData]` — 바꿔 읽을 의도, 또는 None(끝맺는 말을 한다)
  - 음성 `mission_phrases.DISMISS_ACK`(같은 글자, 미리 합성 목록에 포함)

`dismiss` 가 왔을 때(`_react_voice_intent` 의 순찰 검사 바로 뒤):

| 지금 상황 | 처리 |
|---|---|
| 미션 질문이 열려 있음(`CONFIRMING` · `ASKING_NEXT` · `ASKING_WAIT_TIME` · `AWAITING_USER` · 취소 확인 중 · 도움 확인 중 · 대기 중 질문을 한 뒤) | `deny` 로 바꿔 지금 길로 |
| `NAVIGATING` · `PAUSED` 이고 열린 질문 없음 | `cancel` 로 바꿔 지금 길로("안내를 취소할까요?") |
| `IDLE`, 질문 없는 `WAITING`·`WAITING_RELEASE` | `Say(MSG_DISMISS_ACK, priority="response")`, 상태 그대로. `expects_reply` 를 주지 않는다 — 듣기 창을 다시 열지 않는다. `IDLE` 은 `_forget_dest_prompt()` 도 부른다 |
| 그 밖 | `[]` |

- [ ] **Step 1: 실패하는 시험 작성** — `reaction_states.REQUESTS` 에 `"bye": intent("dismiss")`, `test_reaction_table.COLS` 끝에 `"bye"`. `EXPECT` 의 모든 행에 `bye` 칸을 위 표대로 채운다: 질문이 열린 행은 그 행의 `no` 칸과 같은 값, `NAVIGATING`·`PAUSED` 행은 그 행의 `cancel` 칸과 같은 값, `IDLE`·질문 없는 대기 행은 `((MSG_DISMISS_ACK,), <같은 상태>)`, 나머지는 `((), <같은 상태>)`.
- [ ] **Step 2:** Run `python3 -m pytest test/test_reaction_table.py -q` → FAIL(`bye` 칸)
- [ ] **Step 3:** 미션에 구현한다. 음성 `mission_phrases.py` 에 `DISMISS_ACK` 를 더하고 `baked_mission_ments`·미리 합성 목록에 넣는다.
- [ ] **Step 4:** 미션 `python3 -m pytest test/test_reaction_table.py test/test_reaction_rules.py test/test_spoken_text.py -q` → PASS. 음성 `… tests/test_mission_phrases.py tests/test_spoken_text.py -q -p no:cacheprovider` → PASS.
- [ ] **Step 5:** 저장소마다 커밋. 미션 `feat(mission): 대화 끝내기 dismiss — 안내 중이 아니면 끝맺는 말, 질문 중이면 거절, 주행 중이면 취소 확인(원칙 14)`, 음성 `feat(voice): 미션 문장 사본에 끝맺는 말`.

### Task 2-7: 2단계 측정과 닫기 (설계 7절 P3 포함)

**Files:**
- Modify: `tools/answer_replay.py`(`--recut`: `VICA_RECUT=on` 을 해석기 import 전에 넣는다 — `answer_env` 와 같은 방식), `tests/test_answer_replay.py`
- Create: `~/vica_data/field_test_<날짜>_recut/scenes.md`(실기 장면표)

- [ ] **Step 1:** `--recut` 을 더하고 시험한다(`recut_env(True) == {"VICA_RECUT": "on"}`). 커밋.
- [ ] **Step 2: "단계 닫기" 1~4번.** 재는 묶음은 208행과 사람끼리 21행이다. 사람끼리 한 말에서 `silent` 로 간 수(설계 6.1: 18행과 21건)를 따로 적는다.
- [ ] **Step 3: 로컬(Mi:dm) 판정**(P3). `--path midm --recut --rounds 3` 을 꺼진 기준(Task 1-4)과 견준다. 안 본 말의 `wrong` 이 늘거나 `ok` 가 3 넘게 줄면 로컬 경로는 옛 지시문과 스키마로 두기로 하고, Task 4-1(로컬 대피에 따로 만든 messages)을 이 단계로 앞당겨 구현한 뒤 다시 잰다.
- [ ] **Step 4: 장면표를 쓴다.** 장면: 옆 사람과 이야기하기(확인 질문 중 / "네?" 직후 / 안내 없음) · 없는 곳 말하기 · 화장실이 둘인 지도에서 "화장실" · 목적지 없이 "안내해 줘" · 안내 없을 때 "됐어" · 주행 중 "됐어 그만" · 확인 질문에 "됐어". 장면마다 기대 문장과 깨질 수 있는 원칙 번호를 적는다.
- [ ] **Step 5: "단계 닫기" 5~6번.** 운영 `.env` 에 `VICA_RECUT=on` 을 넣는 것과 노드 재기동은 사용자에게 요청한다. 글자 경로가 질문 뒤 창의 `unknown`·`clarify` 를 버리는 동작(`is_relisten_noise`, 설계 10절)이 실기에서 원칙 8 과 부딪히는지 본 대로 보고한다 — 고칠지는 사용자가 정한다.

---

# 3단계 — 미션과 맞닿은 쪽 (설계 6.6 · 6.7 · 6.8)

### Task 3-1: 고른 뒤 바로 출발 33건이 남았는지 확인 (설계 6.7, 원칙 3)

**Files:**
- Create: `~/vica_data/voice_answer_audit_20261113/recut/check_choice_confirm.py`
- Test: `tests/test_one_voice.py`

- [ ] **Step 1: 확인 스크립트.** `qa_rows.csv` 에서 `kind == "choice"` · `intent == "navigate"` · `dest_id` 있음 · `nc == "False"` 인 33행을 뽑는다. 행마다 그날 목적지(`dest_yaml`)로 `OpenQuestion(kind="choice", option_ids=_choice_option_ids(...))` 와 확정 navigate 를 만들어 `one_voice.demote_unasked_confirm` 에 넣고 `need_confirm` 을 센다.
  Expected: 33행 모두 `need_confirm=True`. 남는 행이 있으면 그 까닭(갈 수 없는 곳 · 목록에 없는 id)과 함께 보고한다.
- [ ] **Step 2: 못 박는 시험**

```python
def test_p3_pick_from_choice_is_demoted_to_a_confirm_question():
    q = OpenQuestion("q", "choice", "남자 화장실과 여자 화장실 중 어디로 갈까요?", (MEN.id, WOMEN.id), 0.0)
    out = demote_unasked_confirm(nav(MEN, need_confirm=False), q, DESTS)
    assert out.need_confirm is True and out.reply == ""
```

- [ ] **Step 3:** Run `… tests/test_one_voice.py -q -p no:cacheprovider` → PASS(코드 변경 없음). 남는 행이 없으면 6.7 의 이 항목은 닫는다. 커밋 `test(voice): 보기에서 고른 뒤에도 확인 질문(원칙 3) 못 박음`.

### Task 3-2: 대답 칸을 메시지에 싣는다 (11-18 설계 4.4)

**Files:**
- Modify: 미션 `src/vica_interfaces/msg/VicaIntent.msg`, `vica_mission_manager/question_record.py`, `vica_mission_manager/mission_logic.py:202-222`(`IntentData`), `vica_mission_manager/mission_manager_node.py:957-985`
- Modify: 음성 `src/ros_convert.py:26-47`, `tests/test_answer_wiring.py`
- Test: 미션 `test/test_question_record.py`

**Interfaces:**
- Produces:
  - `VicaIntent.msg` 끝에 `string question_id` · `string answer`(yes / no / destination / choice / minutes / other / unclear) · `string answer_value`(목적지 id · 분)
  - 음성 `intent_to_msg`: `hasattr(msg, "question_id")` 일 때만 세 칸을 싣는다(`ask_*` 와 같은 방식)
  - 미션 `question_record.answer_for(open_id: str, msg_question_id: str, answer: str) -> str` — 두 id 가 비지 않고 같을 때만 `answer`, 아니면 `""`
  - 미션 `IntentData.answer: str = ""` · `IntentData.answer_value: str = ""`. 노드 `_on_intent` 는 질문 기록을 닫기 **전에** 열린 질문 id 를 읽어 `answer_for` 로 거른 값을 넣는다.

- [ ] **Step 1: 실패하는 시험 작성**

```python
# 미션 test/test_question_record.py
def test_stale_question_id_is_not_an_answer():
    assert answer_for("q1-3", "q1-3", "yes") == "yes"
    assert answer_for("q1-4", "q1-3", "yes") == ""      # 늦게 온 대답 — 지금처럼 의도로 처리한다
    assert answer_for("", "q1-3", "yes") == "" and answer_for("q1-3", "", "yes") == ""

# 음성 tests/test_answer_wiring.py — test_nothing_new_goes_to_the_mission_yet 를 바꾼다
def test_answer_fields_go_to_the_mission_when_the_message_has_them():
    text = (SRC / "ros_convert.py").read_text()
    assert 'hasattr(msg, "question_id")' in text and "msg.answer_value" in text
```

- [ ] **Step 2:** 두 저장소에서 Run → FAIL
- [ ] **Step 3:** 구현한다.
- [ ] **Step 4:** Run → PASS. 음성 `… tests/test_one_voice_convert.py -q -p no:cacheprovider` 도 PASS.
- [ ] **Step 5: 메시지 빌드는 사용자와 맞춘다.** 주행 묶음이 꺼졌는지 확인한 뒤 `cd ~/VICA-smarthandle/vica_ros2_ws && colcon build --packages-select vica_interfaces`. 빌드 뒤에는 `/vica/intent` 를 쓰는 노드(음성 의도 노드와 미션)를 **함께** 재기동해야 한다 — 한쪽만 다시 뜨면 메시지 꼴이 달라 의도가 전달되지 않는다. 재기동은 사용자에게 요청한다.
- [ ] **Step 6:** 저장소마다 커밋. 미션 `feat(interfaces): VicaIntent 에 대답 칸(question_id·answer·answer_value)`, 음성 `feat(voice): 대답 칸을 메시지에 실음(메시지에 칸이 있을 때만)`.

### Task 3-3: 미션이 대답 칸으로 처리한다 (11-18 설계 4.5, 원칙 6)

**Files:**
- Modify: 미션 `mission_logic.py:1528-1564`(`__init__`)·`1977-2008`(`_react_voice_intent`)·`2563-2572`(같은 곳 재제안 = 승낙), `mission_manager_node.py`(파라미터 `answer_field` 선언·전달, 341 · 439 부근), launch 파일(인자 `answer_field`, 기본 false)
- Test: 미션 `test/test_answer_field_mission.py`(새 파일)

**Interfaces:**
- Consumes: Task 3-2 의 `IntentData.answer`·`answer_value`
- Produces: `MissionLogic(answer_field: bool = False)` · `MissionLogic._intent_from_answer(self, intent: IntentData) -> IntentData`

`answer_field` 가 켜져 있고 `intent.answer` 가 있을 때, 반응표로 가기 전에 바꿔 읽는다:

| `answer` | 바꿔 읽는 의도 |
|---|---|
| `yes` · `no` | `affirm` · `deny`. `CONFIRMING` 의 `yes` 는 `on_confirm_answer(True)` 와 같은 결과(출발) |
| `destination` · `choice` | `navigate`(`matched_destination_id=answer_value`, `need_confirm=True`). `CONFIRMING` 에서 확인 중인 곳과 같아도 출발하지 않고 다시 확인한다 — 대답 칸이 온 발화에는 09-01 규칙(2563-2572)을 쓰지 않는다 |
| `minutes` | `wait`(`wait_minutes=int(answer_value)`) |
| `unclear` | `unknown` |
| `other` · 빈 값 | 그대로 |

- [ ] **Step 1: 실패하는 시험 작성** — `reaction_states` 의 `confirming()`·`asking()`·`intent()` 를 쓴다.

```python
def test_p6_confirm_yes_departs_and_no_folds():
    logic, t = confirming(answer_field=True)
    logic.on_voice_intent(intent("unknown", answer="yes"), t, lookup, BOUNDS, True)
    assert logic.state == State.NAVIGATING
    logic, t = confirming(answer_field=True)
    assert MSG_CONFIRM_TIMEOUT in _says(logic.on_voice_intent(intent("navigate", answer="no"), t, lookup, BOUNDS, True))

def test_p3_same_place_named_again_asks_again_instead_of_departing():
    logic, t = confirming(answer_field=True)          # 화장실 확인 중
    acts = logic.on_voice_intent(intent("navigate", answer="destination", answer_value=TOILET.id), t, lookup, BOUNDS, True)
    assert logic.state == State.CONFIRMING and not any(isinstance(a, Navigate) for a in acts)

def test_p6_minutes_answer_is_a_wait():
    logic, t = asking_wait_time(answer_field=True)
    logic.on_voice_intent(intent("unknown", answer="minutes", answer_value="10"), t, lookup, BOUNDS, True)
    assert logic.state == State.WAITING

def test_other_and_off_follow_the_intent():
    logic, t = confirming(answer_field=True)
    a = _says(logic.on_voice_intent(intent("cancel", answer="other"), t, lookup, BOUNDS, True))
    logic2, t2 = confirming()
    assert a == _says(logic2.on_voice_intent(intent("cancel"), t2, lookup, BOUNDS, True))
    logic3, t3 = confirming()                          # 꺼짐 — 대답 칸을 보지 않는다
    logic3.on_voice_intent(intent("unknown", answer="yes"), t3, lookup, BOUNDS, True)
    assert logic3.state == State.CONFIRMING
```

- [ ] **Step 2:** Run `python3 -m pytest test/test_answer_field_mission.py -q` → FAIL
- [ ] **Step 3:** 구현한다. `reaction_states` 의 상태 만들기 함수가 `**kw` 를 `new_logic` 에 넘기지 않으면 넘기게 고치고, 없는 상태 함수(`asking_wait_time`·`paused`)는 더한다.
- [ ] **Step 4:** Run → PASS. `python3 -m pytest test -q` 통과 수가 기준 이상.
- [ ] **Step 5:** `git commit -m "feat(mission): 대답 칸으로 처리 — 열린 질문의 대답이면 answer 로, 아니면 지금처럼 의도로(파라미터 answer_field, 기본 꺼짐)" -- <고친 파일>`

### Task 3-4: "다시 출발할까요?"를 미션이 묻는다 (11-18 설계 4.4·4.5)

**Files:**
- Modify: 미션 `mission_logic.py`(`_react_paused` 2209-2220 · `MSG_*` 690 부근)
- Modify: 음성 `src/ros_node.py:462-558`·`625`, `src/schema.py:145-157`
- Test: 미션 `test/test_answer_field_mission.py`, 음성 `tests/test_answer_wiring.py`

**Interfaces:**
- Produces:
  - 미션 `MSG_RESUME_CONFIRM = "다시 출발할까요?"`(음성 `replies.RESUME_CONFIRM` 과 같은 글자)
  - `answer_field` 가 켜졌을 때 `PAUSED` 에서: 제안 `resume`(`need_confirm=True`) → `_ask(MSG_RESUME_CONFIRM, "yesno")`. 그 질문이 열려 있지 않은데 확정 `resume` 이 오면 묻는 쪽으로 낮춘다. `affirm`(yes) → 재출발. `deny`(no) → `MSG_PAUSE_KEPT`, 일시정지 유지.
  - 음성: `answer_field.on()` 이면 `should_forward_intent` 가 늘 참이고, 음성이 쥔 재개 질문 길(`reask_held_resume` · `held_resume_carrier` · `_on_listen_state_for_resume`)을 타지 않는다.

- [ ] **Step 1: 실패하는 시험 작성**

```python
# 미션
def test_p3_resume_is_asked_by_the_mission_and_never_starts_unasked():
    logic, t = paused(answer_field=True)
    assert _says(logic.on_voice_intent(intent("resume", need_confirm=False), t, lookup, BOUNDS, True)) == [MSG_RESUME_CONFIRM]
    assert logic.state == State.PAUSED
    logic.on_voice_intent(intent("affirm"), t + 2, lookup, BOUNDS, True)
    assert logic.state == State.NAVIGATING

def test_resume_declined_stays_paused():
    logic, t = paused(answer_field=True)
    logic.on_voice_intent(intent("resume", need_confirm=True), t, lookup, BOUNDS, True)
    assert _says(logic.on_voice_intent(intent("deny"), t + 2, lookup, BOUNDS, True)) == [MSG_PAUSE_KEPT]
    assert logic.state == State.PAUSED

# 음성
def test_resume_proposals_reach_the_mission_when_answers_are_on(monkeypatch):
    held = VicaIntent(intent="resume", need_confirm=True)
    assert should_forward_intent(held) is False
    monkeypatch.setenv("VICA_ANSWER", "on")
    assert should_forward_intent(held) is True
```

- [ ] **Step 2:** Run → FAIL
- [ ] **Step 3:** 구현한다. 꺼졌을 때의 `PAUSED` 반응표 칸은 그대로여야 한다.
- [ ] **Step 4:** 미션 `python3 -m pytest test/test_answer_field_mission.py test/test_reaction_table.py -q` → PASS. 음성 `… tests/test_answer_wiring.py tests/test_one_voice.py tests/test_command_intents.py -q -p no:cacheprovider` → PASS.
- [ ] **Step 5:** 저장소마다 커밋.

### Task 3-5: 대답 칸 — 로컬과 Realtime 을 맞춘다 (11-18 설계 5.1)

**Files:**
- Modify: 음성 `src/answer_field.py`, `src/langchain_intent_parser.py:598-614`(`_ollama_structured`), `src/realtime_intent.py:96-124`, `tests/conftest.py`
- Test: 음성 `tests/test_answer_field.py`, `tests/test_parser_answer.py`, `tests/test_realtime_intent.py`

**Interfaces:**
- Produces:
  - `answer_field.local_on() -> bool` — `on()` 이고 `.env` `VICA_ANSWER_LOCAL=on`. 꺼져 있으면 로컬 스키마에 `answer` 칸이 없고(`narrowed` 를 쓰지 않음) 짧은 대답 규칙(`short_answer`)만 대답 칸을 채운다.
  - Realtime 도구: 대답 칸이 켜져 있고 열린 질문이 있으면 `required` 에 `answer` 를 넣는다(`build_intent_tool(question_open: bool = False)`).

- [ ] **Step 1: 실패하는 시험 작성**

```python
def test_local_schema_has_no_answer_field_unless_asked(monkeypatch):
    monkeypatch.setenv("VICA_ANSWER", "on")
    assert answer_field.local_on() is False
    monkeypatch.setenv("VICA_ANSWER_LOCAL", "on")
    assert answer_field.local_on() is True

def test_realtime_must_fill_answer_when_a_question_is_open(monkeypatch):
    monkeypatch.setenv("VICA_ANSWER", "on")
    assert "answer" in build_intent_tool(question_open=True)["parameters"]["required"]
    assert "answer" not in build_intent_tool()["parameters"]["required"]
```

- [ ] **Step 2:** Run → FAIL
- [ ] **Step 3:** 구현한다.
- [ ] **Step 4:** Run → PASS. 커밋 `feat(voice): 대답 칸 — 로컬은 규칙 대답만(VICA_ANSWER_LOCAL), Realtime 은 질문이 열렸을 때 answer 필수`.

### Task 3-6: 되묻기 사다리 (순수 모듈, 설계 6.8, 원칙 8)

**Files:**
- Create: 미션 `vica_mission_manager/reask_ladder.py`
- Test: 미션 `test/test_reask_ladder.py`

**Interfaces:**
- Produces:

```python
class ReaskLadder:
    def __init__(self, three_step: bool = False) -> None: ...
    step: int                                   # 0~3
    def reset(self) -> None: ...                # 알아들은 말 · 새 질문 · 호출
    def next(self, driving: bool, can_help: bool = True) -> str: ...
    # 옛 방식(three_step=False): 첫 번째 "same", 그 뒤 "end"
    # 세 단계: "retry" → "louder" → (driving 이고 can_help 면 "help", 아니면 "end") → "end"
```

- [ ] **Step 1: 실패하는 시험 작성**

```python
def test_legacy_policy_reasks_the_same_question_once():
    l = ReaskLadder()
    assert [l.next(False), l.next(False), l.next(True)] == ["same", "end", "end"]

def test_p8_three_steps_and_staff_only_while_driving():
    l = ReaskLadder(three_step=True)
    assert [l.next(True), l.next(True), l.next(True), l.next(True)] == ["retry", "louder", "help", "end"]
    l.reset()
    assert [l.next(False), l.next(False), l.next(False)] == ["retry", "louder", "end"]
    l.reset()
    assert [l.next(True, can_help=False) for _ in range(3)] == ["retry", "louder", "end"]

def test_p8_understood_utterance_resets():
    l = ReaskLadder(three_step=True)
    l.next(False); l.reset()
    assert l.step == 0 and l.next(False) == "retry"
```

- [ ] **Step 2~5:** Run `python3 -m pytest test/test_reask_ladder.py -q` → FAIL → 구현 → PASS → 커밋 `feat(mission): 되묻기 사다리 모듈(옛 방식·세 단계)`.

### Task 3-7: 질문마다 따로 있는 "한 번 다시" 표시를 사다리 하나로 (동작 변화 없음)

**Files:**
- Modify: 미션 `mission_logic.py` — 표시 일곱 개와 쓰이는 줄: `_confirm_reasked`(1837 · 2169 · 2171 · 2178 · 5381) · `_cancel_reasked`(1839 · 2083 · 2085 · 3229 · 3240) · `_help_confirm_reasked`(1765 · 3303 · 3305 · 3361 · 3400 · 3427) · `_help_cancel_reasked`(1769 · 3287 · 3289 · 3315 · 3414 · 3433) · `_wait_need_reasked`(1813 · 2304 · 2311 · 2343 · 5196) · `_wait_finish_reasked`(1812 · 2323 · 2326 · 2337 · 2354 · 2753 · 5195) · `_arrival_retried`(1780 · 2249 · 2268 · 4479 · 4608 · 5051 · 5062 · 5160 · 5178)
- Test: 기존 미션 시험 전체

**Interfaces:**
- Consumes: Task 3-6 의 `ReaskLadder`
- Produces:
  - `self._ladder = ReaskLadder(three_step=False)` 하나. 표시를 세우던 자리는 `self._ladder.next(...)` 의 결과로 가르고, 지우던 자리는 `self._ladder.reset()` 이다.
  - `MissionLogic._resay_open_question(self, now: float) -> Optional[list]` — 지금 상태에서 미션이 쥔 열린 질문을 다시 말하는 동작(없으면 None). 표시마다 흩어져 있던 "같은 질문을 다시" 코드를 이 함수가 부른다.
- 접근 질문의 횟수(`_approach_reasks`, 상한 3, `MSG_APPROACH_UNANSWERED`)는 그대로 둔다 — 이미 세 번이고 끝 문장이 따로 있다.

- [ ] **Step 1:** 시작 전에 `python3 -m pytest test -q` 의 통과·실패 수를 적는다.
- [ ] **Step 2:** 표시 하나씩 바꾼다. 하나를 바꿀 때마다 `python3 -m pytest test/test_reaction_rules.py test/test_reaction_table.py test/test_arrival_dialog.py test/test_wake_reaction.py test/test_help_request.py test/test_wait_speaker_flow.py -q` → PASS.
- [ ] **Step 3:** `python3 -m pytest test -q` → Step 1 과 같은 수. 시험을 고쳐야 통과한다면 동작이 바뀐 것이다 — 되돌리고 원인을 찾는다.
- [ ] **Step 4:** `git commit -m "refactor(mission): 질문마다 따로 있던 다시 묻기 표시 일곱 개를 사다리 하나로(동작 그대로)" -- <고친 파일>`

### Task 3-8: 세 단계 되묻기 (설계 6.8 · 7절 P5, 원칙 8 · 31)

**Files:**
- Modify: 미션 `mission_logic.py`(`Say` 257-272 · `__init__` · 사다리를 쓰는 자리 · `on_intent` 2524-2527 · `_say_handed_question` 1963-1975), `mission_manager_node.py`(`_open_question` 2521-2531 · 파라미터 `reask_ladder`), launch 파일
- Modify: 음성 `src/mission_phrases.py`
- Test: 미션 `test/test_reask_three_steps.py`(새 파일), `test/test_spoken_text.py`

**Interfaces:**
- Produces:
  - `MSG_REASK_RETRY = "잘 듣지 못했습니다. 다시 말씀해 주세요."` · `MSG_REASK_LOUDER = "죄송합니다. 더 크게, 또박또박 말씀해 주세요."`
  - `MissionLogic(reask_ladder: bool = False)` — 켜지면 `ReaskLadder(three_step=True)`
  - `Say.question_text: str = ""`(질문 기록에 적을 글 — 비면 `text`) · `Say.step: int = 0`. 노드는 `VicaQuestion.text` 에 `question_text or text` 를, `step` 에 그 값을 싣는다.

켜졌을 때:
- 미션 질문이 열린 채 못 알아들었거나(`unknown`, 대답 칸 `unclear`) 대답이 없으면 사다리를 한 칸 올린다. `retry` → `MSG_REASK_RETRY`, `louder` → `MSG_REASK_LOUDER`. 두 문장 모두 열린 질문의 종류·후보를 그대로 둔 채 말하고(`question_text` = 원래 질문), 질문의 시한을 다시 잰다. `help` → 지금의 도움 확인 길(`_on_help_intent` 가 타는 길: 취소 질문을 닫고 "직원의 도움이 필요하신가요?"). `end` → 그 질문이 끝까지 대답을 못 받았을 때 하던 일을 바로 한다(확인 = `_fold_confirming` 으로 접기 · 취소 확인 = 조용히 닫기 · 대기 질문 = `MSG_CANCEL_KEPT` · 도착 질문 = `MSG_LEAVING_NOTICE`).
- 질문이 없는 상태의 `unknown`(지금은 `on_intent` 2524-2527 에서 말 없이 끝남)도 같은 사다리를 탄다: `retry`·`louder` 를 `kind="open"` 으로 말하고, `end` 면 말하지 않는다. 주행 중(`NAVIGATING`)이면 세 번째가 `help` 다.
- 음성이 넘긴 되묻는 문장(`unknown` 의 `ask_text`)은 말하지 않는다 — 미션이 사다리의 문장을 말한다(원칙 9).
- 알아들은 의도(`unknown` 이 아닌 것)를 처리하면 `reset()`.

- [ ] **Step 1: 실패하는 시험 작성** — `_question(acts)` 는 동작 가운데 `kind` 가 있는 `Say` 를 돌려주는 시험 도우미다.

```python
def test_p8_confirm_question_goes_retry_louder_then_folds_when_not_driving():
    logic, t = confirming(reask_ladder=True)
    r1 = logic.on_voice_intent(intent("unknown"), t, lookup, BOUNDS, True)
    r2 = logic.on_voice_intent(intent("unknown"), t + 5, lookup, BOUNDS, True)
    assert _says(r1) == [MSG_REASK_RETRY] and _says(r2) == [MSG_REASK_LOUDER]
    assert _question(r1).question_text == say_destination(MSG_CONFIRM_PROMPT_FALLBACK, "화장실") and _question(r1).kind == "confirm"
    assert (_question(r1).step, _question(r2).step) == (1, 2)
    logic.on_voice_intent(intent("unknown"), t + 10, lookup, BOUNDS, True)
    assert logic.state != State.CONFIRMING

def test_p21_third_step_offers_staff_only_while_driving():
    logic, t = navigating(reask_ladder=True)
    says = [_says(logic.on_voice_intent(intent("unknown"), t + i, lookup, BOUNDS, True)) for i in range(3)]
    assert says == [[MSG_REASK_RETRY], [MSG_REASK_LOUDER], [MSG_HELP_CONFIRM]]
    logic, t = idle(reask_ladder=True)
    says = [_says(logic.on_voice_intent(intent("unknown"), t + i, lookup, BOUNDS, True)) for i in range(3)]
    assert says == [[MSG_REASK_RETRY], [MSG_REASK_LOUDER], []]

def test_p9_the_voice_handed_retry_sentence_is_not_said_twice():
    logic, t = idle(reask_ladder=True)
    acts = logic.on_voice_intent(intent("unknown", ask_text=MSG_REASK_RETRY, ask_kind="open"), t, lookup, BOUNDS, True)
    assert _says(acts) == [MSG_REASK_RETRY]

def test_p8_silence_and_not_understood_share_one_count():
    logic, t = confirming(reask_ladder=True)
    assert _says(logic.on_tick(t + 16, NavStatus.NONE)) == [MSG_REASK_RETRY]            # 대답 없음
    assert _says(logic.on_voice_intent(intent("unknown"), t + 20, lookup, BOUNDS, True)) == [MSG_REASK_LOUDER]

def test_off_still_reasks_the_same_question_once():
    logic, t = confirming()
    ask = say_destination(MSG_CONFIRM_PROMPT_FALLBACK, "화장실")
    assert _says(logic.on_voice_intent(intent("unknown"), t, lookup, BOUNDS, True)) == [ask]
```

- [ ] **Step 2:** Run `python3 -m pytest test/test_reask_three_steps.py -q` → FAIL
- [ ] **Step 3:** 구현한다. 음성 `mission_phrases.py` 에 `MSG_REASK_LOUDER` 사본을 더한다(`MSG_REASK_RETRY` 는 `replies.RETRY_PROMPT` 와 같은 글자 — 사본을 따로 두지 않고 같다는 것을 `tests/test_mission_phrases.py` 에서 본다).
- [ ] **Step 4:** Run → PASS. 미션 `python3 -m pytest test -q` 통과 수가 기준 이상, 음성 `… tests/test_mission_phrases.py tests/test_spoken_text.py -q -p no:cacheprovider` → PASS.
- [ ] **Step 5:** 저장소마다 커밋. 미션 `feat(mission): 되묻기 세 단계 — 다시 말씀·더 크게 또박또박·(주행 중만)직원 도움, 질문 없는 상태도 되묻는다(파라미터 reask_ladder, 기본 꺼짐)`.

### Task 3-9: "비카야" 뒤 처리 (설계 6.7 · 7절 P1, 원칙 13 · 3)

**Files:**
- Modify: 미션 `mission_logic.py`(`on_wake` 4552-4623 · `_react_voice_intent` 1977-2008 · `__init__`), `mission_manager_node.py`(파라미터 `wake_revive`), launch 파일
- Test: 미션 `test/test_wake_reaction.py`

**Interfaces:**
- Produces:
  - `MissionLogic(wake_revive: bool = False)` · `WAKE_REVIVE_SEC = 40.0`
  - `MissionLogic._is_wake_only(intent: IntentData) -> bool` — `intent == "unknown"` 이고 `reply` 나 `ask_text` 가 `MSG_WAKE_GREETING`
  - `self._wake_folded: Optional[tuple[str, str, float]]` — (`"confirm"`·`"cancel"`·`"help"`, 목적지 id, 시각)

켜졌을 때:
- "비카야"만 들린 의도(`_is_wake_only`)는 어느 상태에서든 되묻기 횟수에 세지 않고, 열린 질문을 다시 말하지 않는다. 음성이 넘긴 "네?"는 지금처럼 말한다.
- 호출(`on_wake`)에서 확인 질문은 지금처럼 조용히 접고(2026-09-01 결정), "안내를 취소할까요?"와 "직원의 도움이 필요하신가요?"도 조용히 접는다. 접을 때 `_wake_folded` 에 적는다.
- `_wake_folded` 가 있고 40 s 안에 승낙(`affirm`, 대답 칸 `yes`)이 오면 **실행하지 않고 그 질문을 다시 한다**: `confirm` → `navigate`(그 목적지, `need_confirm=True`)로 바꿔 지금 길로 · `cancel` → `cancel` 로 · `help` → `help` 로. 그 뒤 `_wake_folded` 를 지운다.
- "비카야"만 들린 의도가 아닌 다른 의도가 오거나 40 s 가 지나면 `_wake_folded` 를 지운다.
- 접근 질문 중(2026-10-09)과 대기 중 첫 호출(2026-10-21)은 그대로 둔다.

- [ ] **Step 1: 실패하는 시험 작성**

```python
def test_p13_confirm_folded_by_wake_then_yes_asks_again_and_does_not_depart():
    logic, t = confirming(wake_revive=True)
    logic.on_wake_call(t)
    acts = logic.on_voice_intent(intent("affirm"), t + 3, lookup, BOUNDS, True)
    assert _says(acts) == [say_destination(MSG_CONFIRM_PROMPT_FALLBACK, "화장실")]
    assert logic.state == State.CONFIRMING and not any(isinstance(a, Navigate) for a in acts)

def test_p13_cancel_question_folded_by_wake_then_yes_asks_again():
    logic, t = navigating_cancel_asked(wake_revive=True)
    logic.on_wake_call(t)
    assert _says(logic.on_voice_intent(intent("affirm"), t + 3, lookup, BOUNDS, True)) == [MSG_CANCEL_CONFIRM]
    assert logic.state == State.NAVIGATING

def test_p13_help_question_folded_by_wake_then_yes_asks_again():
    logic, t = navigating_help_asked(wake_revive=True)
    logic.on_wake_call(t)
    assert _says(logic.on_voice_intent(intent("affirm"), t + 3, lookup, BOUNDS, True)) == [MSG_HELP_CONFIRM]
    assert logic.help_phase == "asking"

def test_p13_revive_expires_and_other_words_clear_it():
    logic, t = confirming(wake_revive=True)
    logic.on_wake_call(t)
    assert _says(logic.on_voice_intent(intent("affirm"), t + 41, lookup, BOUNDS, True)) == []
    logic, t = confirming(wake_revive=True)
    logic.on_wake_call(t)
    logic.on_voice_intent(intent("question"), t + 2, lookup, BOUNDS, True)
    assert _says(logic.on_voice_intent(intent("affirm"), t + 4, lookup, BOUNDS, True)) == []

def test_p13_wake_only_is_not_counted_in_arrival_question():
    logic, t = asking(wake_revive=True)
    wake_only = intent("unknown", ask_text=MSG_WAKE_GREETING, ask_kind="open")
    for i in range(3):
        acts = logic.on_voice_intent(wake_only, t + i, lookup, BOUNDS, True)
        assert MSG_LEAVING_NOTICE not in _says(acts) and logic.state == State.ASKING_NEXT

def test_off_keeps_todays_wake_behaviour():
    logic, t = navigating_cancel_asked()
    logic.on_wake_call(t)
    assert logic.cancel_confirm_pending is True
```

- [ ] **Step 2:** Run `python3 -m pytest test/test_wake_reaction.py -q` → FAIL(새 시험). 상태 만들기 함수 `navigating_cancel_asked`·`navigating_help_asked` 가 없으면 `reaction_states.py` 에 더한다.
- [ ] **Step 3:** 구현한다.
- [ ] **Step 4:** Run → PASS. `python3 -m pytest test -q` 통과 수가 기준 이상.
- [ ] **Step 5:** `git commit -m "feat(mission): 비카야 뒤 처리 — 되돌리기 어려운 질문은 접었다가 승낙이 오면 다시 묻는다, 호출만 들린 것은 세지 않는다(파라미터 wake_revive, 기본 꺼짐)" -- <고친 파일>`

### Task 3-10: 3단계 측정과 닫기

**Files:**
- Create: `~/vica_data/field_test_<날짜>_recut3/scenes.md`

- [ ] **Step 1: "단계 닫기" 1~4번.** 음성은 `--recut --answer` 로 잰다. Mi:dm 은 `VICA_ANSWER_LOCAL` 을 끈 값과 켠 값을 둘 다 재고(11-20 에는 켠 쪽이 `ok` 139 → 117 로 떨어졌다) 나은 쪽을 권한다. Realtime 은 대답 칸을 비운 비율(11-19 약 절반)이 Task 3-5 뒤에 얼마가 됐는지 적는다.
- [ ] **Step 2: 장면표.** 11-18 설계의 장면(온라인 확인 중 "거기 어디야" → 출발하지 않음 · "다시 가자" → 항상 묻는다 · "다시 출발할까요?"에 "아니" · 오프라인 네 장면)에 더해: 확인 질문 중 "비카야" → "네?" → "응, 가자"(다시 확인 질문) · 취소 질문 중 "비카야" → "응" · 도착 질문 중 "비카야"만 세 번 · 확인 질문에 못 알아듣는 말 세 번(주행 밖 / 주행 중) · 보기에서 고르기.
- [ ] **Step 3: "단계 닫기" 5~6번.** 켤 것은 음성 `.env` `VICA_ANSWER=on`, 미션 launch 의 `answer_field`·`reask_ladder`·`wake_revive` 다. 셋을 한꺼번에 켜지 않고 하나씩 켜며 실기한다(대답 칸 → 사다리 → 호출).

---

# 4단계 — 새 기능 (설계 6.4 · 6.5)

### Task 4-1: 로컬 대피에 따로 만든 messages 를 넘긴다 (설계 7절 P3, 원칙 16)

**Files:**
- Modify: 음성 `src/llm_backend.py:137-154`(`invoke`), `src/langchain_intent_parser.py:1038-1050`
- Test: 음성 `tests/test_llm_backend.py`, `tests/test_parser_backend.py`

**Interfaces:**
- Produces: `LlmBackendManager.invoke(messages, local_messages: Optional[Callable[[], list]] = None)` — 로컬로 넘어갈 때 `local_messages()` 가 있으면 그것을 넘긴다. 없으면 지금처럼 같은 `messages`.

- [ ] **Step 1: 실패하는 시험 작성**

```python
def test_p16_local_fallback_gets_its_own_messages():
    seen = {}
    m = LlmBackendManager(cloud=failing, local=lambda msgs: seen.setdefault("local", msgs) or DRAFT)
    m.invoke(["cloud"], local_messages=lambda: ["local"])
    assert seen["local"] == ["local"]

def test_without_it_the_same_messages_are_reused():
    seen = {}
    m = LlmBackendManager(cloud=failing, local=lambda msgs: seen.setdefault("local", msgs) or DRAFT)
    m.invoke(["cloud"])
    assert seen["local"] == ["cloud"]
```

- [ ] **Step 2~5:** Run → FAIL → 구현 → PASS → 커밋 `feat(voice): 로컬 대피에 따로 만든 지시문을 넘길 수 있게(외부 API 전용 기능을 로컬에 싣지 않으려고)`.

### Task 4-2: 정해진 대답 둘 (설계 6.4 · 7절 P2·P4, 원칙 28 · 29)

**Files:**
- Create: 음성 `src/fixed_answers.py`
- Modify: 음성 `src/replies.py`, `src/schema.py`, `src/langchain_intent_parser.py`(`_RecutDraft` 의 값 · 두 지시문 · `_finalize` · `deliver_audio_draft`), `src/realtime_intent.py`, `src/ros_node.py`(`_publish_intent`), `tests/conftest.py`(`"VICA_TALK": "off"`)
- Test: 음성 `tests/test_fixed_answers.py`

**Interfaces:**
- Produces:
  - `fixed_answers.on() -> bool` — `.env` `VICA_TALK=on` 이고 `intent_recut.on()` 일 때만 참(4단계는 2단계 위에서만 켜진다)
  - `replies.places_sentence(names: Sequence[str]) -> str`
  - `fixed_answers.places_names(destinations: Sequence[DestinationData]) -> list[str]` — 갈 수 있는 곳(`is_approachable`)의 name, 목록 순서
  - `fixed_answers.last_said(robot_recent: Sequence[tuple[float, str]]) -> str` — 마지막 말. "네?"는 건너뛴다. 없으면 `""`
  - `fixed_answers.settle_repeat(question: Optional[OpenQuestion], robot_recent) -> VicaIntent`
  - LLM 이 고르는 값에 `repeat`·`places`, `VicaIntentType` 에 `repeat`
- 지시문에 더하는 두 줄(외부 API 지시문에만 — 로컬용 messages 에는 넣지 않는다):
  - `- repeat: 로봇이 방금 한 말을 다시 해 달라고 한다("다시 말해줘", "뭐라고?", "못 들었어"). reply 는 빈 문자열.`
  - `- places: 어디로 갈 수 있는지, 무엇을 할 수 있는지 묻는다("어디 갈 수 있어?", "뭐 할 수 있어?"). reply 는 빈 문자열 — 문장은 시스템이 만든다.`

처리:
- `places` → `VicaIntent(intent="question", reply=places_sentence(places_names(destinations)))`. 음성이 말한다(지금 `question` 의 길). 갈 수 있는 곳이 하나도 없으면 LLM 이 낸 `question` 처리로 둔다.
- `repeat` → 열린 질문이 있고 그 글이 "네?"가 아니면 `VicaIntent(intent="repeat", reply="")` 를 미션에 보낸다(미션이 그 질문을 다시 말한다 — Task 4-3). 아니면 `VicaIntent(intent="question", reply=last_said(...))` 로 음성이 다시 말한다. 다시 할 말이 없으면 `unknown`(되묻는다).

- [ ] **Step 1: 실패하는 시험 작성**

```python
def test_p29_places_sentence_names_at_most_three():
    assert places_sentence(["화장실", "사회복지", "407호"]) == "화장실, 사회복지, 407호로 안내할 수 있습니다."
    assert places_sentence(["화장실", "사회복지", "407호", "식당"]) == "화장실, 사회복지, 407호 등으로 안내할 수 있습니다."
    assert places_sentence(["화장실", "식당"]) == "화장실, 식당으로 안내할 수 있습니다."
    assert places_sentence(["화장실"]) == "화장실로 안내할 수 있습니다."
    assert places_sentence([]) == ""

def test_p18_places_never_names_a_place_the_robot_cannot_go():
    assert "비공개 408호" not in places_names(DESTS_WITH_PRIVATE)

def test_p28_repeat_says_the_open_question_or_the_last_sentence():
    recent = [(1.0, "화장실에 도착했습니다."), (2.0, "네?")]
    assert last_said(recent) == "화장실에 도착했습니다."          # 평서문도 다시 한다, "네?"는 건너뛴다
    asked = OpenQuestion("q", "confirm", "화장실로 안내해드릴까요?", ("t",), 0.0)
    assert settle_repeat(asked, recent).intent == "repeat"         # 미션이 그 질문을 다시 말한다
    wake = OpenQuestion("q", "open", "네?", (), 0.0)
    out = settle_repeat(wake, recent)
    assert (out.intent, out.reply) == ("question", "화장실에 도착했습니다.")
    assert settle_repeat(None, []).intent == "unknown"

def test_off_schema_and_prompts_have_no_repeat_or_places():
    assert "places" not in parser._build_system_prompt(DESTS) and "places" not in json.dumps(build_intent_tool())
```

- [ ] **Step 2:** Run `… tests/test_fixed_answers.py -q -p no:cacheprovider` → FAIL
- [ ] **Step 3:** 구현한다. `repeat` 의 마지막 말은 의도 노드가 가진 `self._robot_recent` 에서 오므로 `settle_repeat` 는 `_publish_intent` 안(미션에 보내기 전)에서 부른다. 조사 로/으로는 `destination_loader._josa_euro` 를 쓴다.
- [ ] **Step 4:** Run → PASS. Task 2-2 · 2-3 의 시험과 `… tests/test_spoken_text.py -q -p no:cacheprovider` 도 PASS.
- [ ] **Step 5:** `git commit -m "feat(voice): 정해진 대답 — 다시 말해줘·갈 수 있는 곳(문장은 코드, VICA_TALK 켰을 때만·외부 API 경로만)" -- <고친 파일>`

### Task 4-3: 미션 — "다시 말해줘"에 열린 질문을 다시 말한다 (원칙 28)

**Files:**
- Modify: 미션 `mission_logic.py`(`_react_voice_intent`), `test/reaction_states.py`, `test/test_reaction_table.py`
- Test: 미션 `test/test_reaction_table.py`, `test/test_reask_three_steps.py`

**Interfaces:**
- Consumes: Task 3-7 의 `_resay_open_question`
- Produces: `repeat` 의도가 오면 `_resay_open_question(now)` 의 결과를 돌려준다(시한을 다시 잰다, 되묻기 횟수는 그대로). 미션이 쥔 질문이 없으면 `[]`.

- [ ] **Step 1: 실패하는 시험 작성** — `REQUESTS["again"] = intent("repeat")`, `COLS` 끝에 `"again"`. `EXPECT` 의 `again` 칸은 미션 질문이 열린 행에서는 `((<그 질문 문장>,), <같은 상태>)`, 나머지는 `((), <같은 상태>)`.

```python
def test_p28_repeat_does_not_use_up_a_reask_step():
    logic, t = confirming(reask_ladder=True)
    logic.on_voice_intent(intent("repeat"), t, lookup, BOUNDS, True)
    assert _says(logic.on_voice_intent(intent("unknown"), t + 3, lookup, BOUNDS, True)) == [MSG_REASK_RETRY]
```

- [ ] **Step 2~5:** Run → FAIL → 구현 → PASS(`python3 -m pytest test -q` 통과 수 기준 이상) → 커밋 `feat(mission): 다시 말해줘 — 열린 질문을 한 번 더, 되묻기 횟수에 세지 않음(원칙 28)`.

### Task 4-4: 잡담 길 (설계 6.5, 원칙 25 · 26)

**Files:**
- Create: 음성 `src/vica_facts.py`
- Modify: 음성 `src/langchain_intent_parser.py`(두 지시문)
- Test: 음성 `tests/test_small_talk_prompt.py`

**Interfaces:**
- Produces: `vica_facts.FACTS: str`(지시문에 넣는 `[비카]` 절) · `vica_facts.TALK_RULE: str`. 둘 다 `fixed_answers.on()` 일 때 외부 API 지시문에만 들어간다.
- `FACTS` 의 내용은 **사용자 확인을 받은 뒤** 넣는다(이름 비카 · 뜻 VISION COMPANION · 시각장애인 안내 로봇 · 할 수 있는 일: 건물 안 목적지까지 안내, 도착한 곳에서 기다리기, 안내 중 직원 부르기). 초안을 사용자에게 보이고 고친 글을 그대로 쓴다.
- `TALK_RULE`: "안내와 상관없는 말이어도 로봇에게 한 말이면 question 으로 한 문장으로 답한다. 비카 자신에 대한 것은 [비카]의 내용으로, 인사나 가벼운 말은 짧게 답한다. 사실이 필요한 것은 [비카]·목적지 목록·[지금 상황]에 있는 것만 말하고, 없으면 "그건 제가 몰라요."라고만 한다. 하지 않을 일을 약속하지 않는다."
- 켜졌을 때 고치는 줄: 글자 지시문 210 의 `question` 정의를 소리 지시문 1216 과 같은 범위("정보 질문이나 로봇에게 건 가벼운 말")로 넓힌다. 소리 지시문 1277 의 "농담·추측·맞장구…는 넣지 않는다"를 "안내에 관한 말에는 농담·추측·맞장구를 넣지 않는다"로 좁힌다(원칙 25).

- [ ] **Step 1: 실패하는 시험 작성**

```python
def test_p26_on_prompts_carry_the_facts_and_the_talk_rule(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on"); monkeypatch.setenv("VICA_TALK", "on")
    for prompt in (parser._build_system_prompt(DESTS), parser.build_audio_prompt(DESTS)):
        assert vica_facts.FACTS in prompt and vica_facts.TALK_RULE in prompt

def test_p16_local_messages_do_not_carry_them(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on"); monkeypatch.setenv("VICA_TALK", "on")
    local = parser._build_system_prompt(DESTS, for_local_backend=True)
    assert vica_facts.FACTS not in local and "places" not in local

def test_off_prompts_are_unchanged():
    assert "VISION COMPANION" not in parser._build_system_prompt(DESTS)
```

- [ ] **Step 2~5:** Run → FAIL → 구현(`_build_system_prompt` 에 `for_local_backend: bool = False` 를 더하고 `parse_intent` 가 Task 4-1 의 `local_messages` 로 넘긴다) → PASS → 커밋 `feat(voice): 잡담 길 — 비카 자신에 대한 내용과 가벼운 말 한 문장(VICA_TALK, 외부 API 경로만)`.

### Task 4-5: 4단계 측정과 닫기

**Files:**
- Create: `~/vica_data/voice_answer_audit_20261113/recut/talk_scenes.csv`(장면 묶음), `~/vica_data/field_test_<날짜>_recut4/scenes.md`

- [ ] **Step 1: 장면 묶음을 만든다**(설계 8절: 옛 기록에 몇 행 없는 종류는 장면 묶음으로 본다). 기록에서 뽑은 예("자기 소개해 봐." · "우리 온 거 환영해?" · "나 배고파")와 종류별 장면 30개 안팎: 자기소개 · 인사 · 고맙다 · 가벼운 농담 · 사실 질문(아는 것 / 모르는 것) · "다시 말해줘"(질문 중 / 평서문 뒤 / "네?" 뒤) · "어디 갈 수 있어?" · "뭐 할 수 있어?". 행마다 기대 행동과 원칙 번호를 적는다.
- [ ] **Step 2: 재생하고 눈으로 읽는다.** 세는 것: 약속하는 문장(원칙 22 — "알겠습니다"·"할게요"·"해 드릴게요"·"드릴까요"가 든 reply), 목적지 목록과 `[비카]` 에 없는 장소·시설을 말한 reply(원칙 11), 두 문장 이상인 reply(원칙 10).
- [ ] **Step 3: "단계 닫기" 1~6번.** 208행은 회귀만 본다. 로컬 경로는 `--path midm --recut` 의 숫자가 2단계와 같은지 본다(로컬에는 이 단계의 기능이 실리지 않는다).

# 음성 시설 예약 — 도구 호출 1~3단계 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 사용자가 말로 시설을 예약하면(시연 대본 D1) 예약 파일에 저장되고 `/vica/reservation/event` 가 나간다 — 외부 API 두 경로(gpt-5.4-mini 글자, Realtime 소리)에서.

**Architecture:** LLM 은 도구 하나를 호출한다(`set_intent` = 지금의 `_IntentDraft`, `reserve_facility` = 예약 요청과 그 대답). 값은 LLM 이 읽고, 코드는 지어낸 값 거르기·날짜 계산·겹침·묻는 문장·저장만 한다. 진행 중인 예약은 `[지금 상황]` 블록의 한 줄로 LLM 에 알린다. 결과는 `VicaIntent.ask_text`(미션이 말함) 또는 `reply`(음성이 말함)로 나간다. 도구 출력은 경로별 `.env` 스위치(기본 꺼짐) 뒤에 두고 재생 측정이 기준을 넘을 때만 켠다. 로컬 모델에는 도구를 주지 않는다.

**Tech Stack:** Python 3.10 · pytest · LangChain(`langchain-openai` 1.5.1 `bind_tools`) · OpenAI Realtime(`openai` 3.1.0) · ROS 2 Humble(rclpy, 음성은 소스에서 바로 실행)

**Spec:** `docs/superpowers/specs/2026-11-18-facility-reservation-tool-calling-design.md` (범위: 9절의 1~3단계. 앱·지도 서버·취소 요청 = 4단계는 이 계획 밖)

## Global Constraints

- 저장소는 음성 `~/VICA-smarthandle/vica-voice-llm` 하나만 고친다. 브랜치 `after_feedback_waiting`, 작업마다 커밋, **푸시하지 않는다**.
- `vica_ros2_ws`(미션·`vica_interfaces`)와 `VICA_Supervisor` 의 diff 는 0줄.
- 커밋 메시지는 기존 문체(한국어, `feat(voice): …`), 끝에 `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.
- 스위치 `VICA_TOOLS_TEXT`·`VICA_TOOLS_AUDIO`(값 `on` 이면 켬, 기본 꺼짐). 둘 다 꺼져 있으면 동작이 지금과 같아야 한다 — 작업 전 기준 **1812 passed**(11-18 밤, 소프트웨어 에코 제거 머지 `a9ef695` 뒤)가 그대로 통과한다.
- **로컬 경로에는 도구를 주지 않는다**(D11). `_build_local_structured` 와 Modelfile 의 diff 는 0줄.
- **예약 대답을 읽는 새 글자 규칙을 만들지 않는다**(D12). 이름·날짜·시각·네/아니요는 지금의 해석 경로(글자 규칙 → LLM, 소리 모드는 Realtime)가 읽는다.
- **저장하지 않은 예약을 완료라고 말하지 않는다.** 문장 ③은 `ReservationStore.add` 가 돌려준 뒤에만 나간다.
- **LLM 에 저장 도구를 주지 않는다.** 이동 도구도 만들지 않는다(`vica-voice-llm/CLAUDE.md`).
- 지시문(`_build_system_prompt`·`build_audio_prompt`)은 고치지 않는다. 도구 규칙은 도구 설명에 적는다.
- 문장 9개는 아래 글자 그대로(설계 6절, 사용자 확정):
  - ① `성함을 말씀해 주세요.`
  - ② `{이름} 님 이름으로 {날짜} {시작}부터 {끝}까지 {시설} 예약할까요?` — 예: `김민수 님 이름으로 내일 오후 두 시부터 네 시까지 세미나실 예약할까요?`
  - ③ `네, 예약이 완료되었습니다. 이제 어디로 갈까요?`
  - ④ `그 시간에는 이미 예약이 있습니다. 다른 시간을 말씀해 주세요.`
  - ⑤ `언제로 예약할까요?`
  - ⑥ `예약하지 않았습니다.`
  - ⑦ `어디로 예약할까요?`
  - ⑧ `그 시간에는 예약할 수 없습니다. 다른 시간을 말씀해 주세요.`
  - ⑨ `지금은 예약을 받을 수 없습니다.`
- 대답을 기다리는 문장(①②③④⑤⑦⑧)은 `ask_text`·`ask_kind` 로 넘겨 미션이 말한다(② `yesno`, 나머지 `open`). ⑥·⑨ 는 음성이 `reply` 로 말한다.
- rclpy 로거를 조건식으로 고르지 않는다(`tests/test_one_voice_wiring.py::test_no_logger_picked_by_condition`).
- 시험·측정·시연 예시의 사람 이름은 흔한 이름을 여러 개 쓴다(김민수 · 이서연 · 박준호 · 정현우 · 강민지). 이름 하나에 맞춘 결과가 되지 않게 한다(사용자 11-19: "굳이 김토니로 할 필요는 없다"). 실제 사람의 정보는 쓰지 않는다.
- 주행 묶음(`vica-drive.target`)이 켜져 있으면 전체 시험과 재생 측정은 미루고 해당 시험 파일만 돌린다. 유닛을 직접 건드리지 않는다. `.env` 스위치를 켜는 일·음성 노드 재시작·외부 API 를 부르는 재생 측정은 그때마다 사용자에게 묻는다.
- 시험 명령: `cd ~/VICA-smarthandle/vica-voice-llm && .venv/bin/python -m pytest -q -p no:cacheprovider`

## Review Focus

1. LLM 이 지어낸 이름(받아쓴 글자에 없는 이름)은 저장되지 않고 ①을 다시 묻는다. → Task 4 `test_values_not_in_the_words_are_dropped`
2. 확인 질문에 "네"가 두 번 와도(메아리·겹친 발화) 예약이 두 건 생기지 않는다. → Task 4 `test_confirm_no_and_second_yes_does_not_save_twice`
3. 예약 파일이 잘려 있으면 덮어쓰지 않는다 — 기존 예약을 날리지 않고 ⑨로 끝난다. → Task 1 `test_broken_file_is_not_overwritten`, Task 4 `test_save_failure_says_unavailable_not_done`
4. 끝 시각이 운영 시간을 넘는 요청("오후 다섯 시" + 2시간)과 이미 지난 시각은 ⑧이고 저장되지 않는다. → Task 4 `test_out_of_hours_and_past_are_refused`
5. 도구 호출이 없거나 두 개 오거나 인자가 빈 `reserve_facility({})` 여도 죽지 않는다 — 없으면 실패 처리, 두 개면 첫 호출, 빈 인자면 ⑦부터. → Task 5 `test_tool_message_shapes`, Task 4 `test_empty_call_asks_facility_first`
6. 같은 첫 요청을 소리·글자 두 경로가 모두 받아도 "성함을 말씀해 주세요."가 한 번만 나간다. → Task 9 `test_audio_reservation_is_under_the_text_first_guard`
7. 도구가 없는 경로(스위치 꺼짐·로컬 폴백)에서 "예약해 줘"가 엉뚱한 이동이 되지 않고 ⑨가 된다. → Task 6 `test_reservation_word_without_tools_is_refused`
8. 예약 질문 중에 잡음이 못 알아들은 말로 읽혀도 진행 중인 예약이 날아가지 않는다. → Task 4 `test_general_result_while_active`

---

### Task 1: 예약 저장 모듈과 점검 CLI (설계 5.4)

**Files:**
- Create: `src/reservation_store.py`, `config/reservations/facilities.yaml`, `tools/reservations.py`
- Test: `tests/test_reservation_store.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Facility(name: str, aliases: tuple[str, ...], default_hours: int, open: str, close: str)` — 시각은 `"HH:MM"`
  - `@dataclass(frozen=True) Reservation(id: str, facility: str, date: str, start: str, end: str, name: str, source: str, created_at: str)` — `date` 는 `YYYY-MM-DD`
  - `reservation_dir() -> Path` — `VICA_RESERVATION_DIR`, 기본 `~/vica_data/reservations`
  - `load_facilities(path: Path) -> list[Facility]` — 파일이 없거나 깨졌으면 `[]`
  - `default_facilities() -> list[Facility]` — `reservation_dir()/facilities.yaml`, 없으면 저장소의 `config/reservations/facilities.yaml`
  - `class ConflictError(Exception)`
  - `class ReservationStore(path: Path)`: `path`(속성) · `list() -> list[Reservation]`(날짜·시작 순) · `conflicts(facility, date, start, end) -> list[Reservation]` · `add(facility, date, start, end, name, now: datetime, source: str = "voice") -> Reservation` · `cancel(reservation_id: str) -> Optional[Reservation]`
  - 파일이 없으면 `list()` 는 `[]`. 파일이 JSON 배열로 읽히지 않으면 `list`·`add`·`cancel` 모두 `OSError`.

- [ ] **Step 1: 실패하는 시험 작성** — `tests/test_reservation_store.py`

```python
"""예약 저장(설계 2026-11-18 5.4)."""
import json
from datetime import datetime

import pytest

from src.reservation_store import ConflictError, Facility, ReservationStore, load_facilities

NOW = datetime(2026, 11, 18, 14, 3, 21)


def test_add_then_list_sorted(tmp_path):
    s = ReservationStore(tmp_path / "reservations.json")
    late = s.add("세미나실", "2026-11-19", "14:00", "16:00", "김민수", NOW)
    early = s.add("세미나실", "2026-11-19", "10:00", "11:00", "이서연", NOW)
    assert [r.id for r in s.list()] == [early.id, late.id]
    assert early.id and early.id != late.id and late.source == "voice"
    on_disk = json.loads((tmp_path / "reservations.json").read_text(encoding="utf-8"))
    assert on_disk[0]["name"] == "이서연" and on_disk[1]["end"] == "16:00"


def test_overlap_is_same_facility_same_day_only(tmp_path):
    s = ReservationStore(tmp_path / "reservations.json")
    s.add("세미나실", "2026-11-19", "14:00", "16:00", "김민수", NOW)
    assert s.conflicts("세미나실", "2026-11-19", "15:00", "17:00")
    assert not s.conflicts("세미나실", "2026-11-19", "16:00", "18:00")   # 맞닿는 것은 겹침이 아니다
    assert not s.conflicts("상담실", "2026-11-19", "14:00", "16:00")
    assert not s.conflicts("세미나실", "2026-11-20", "14:00", "16:00")
    with pytest.raises(ConflictError):
        s.add("세미나실", "2026-11-19", "15:00", "17:00", "박준호", NOW)
    assert len(s.list()) == 1


def test_missing_file_is_empty_and_add_creates_the_folder(tmp_path):
    s = ReservationStore(tmp_path / "new" / "reservations.json")
    assert s.list() == []
    s.add("상담실", "2026-11-19", "09:00", "11:00", "박준호", NOW)
    assert (tmp_path / "new" / "reservations.json").exists()


def test_broken_file_is_not_overwritten(tmp_path):
    path = tmp_path / "reservations.json"
    path.write_text('[{"id": "r1", "facil', encoding="utf-8")
    s = ReservationStore(path)
    with pytest.raises(OSError):
        s.add("세미나실", "2026-11-19", "14:00", "16:00", "김민수", NOW)
    with pytest.raises(OSError):
        s.list()
    assert path.read_text(encoding="utf-8") == '[{"id": "r1", "facil'


def test_cancel_removes_one_and_returns_it(tmp_path):
    s = ReservationStore(tmp_path / "reservations.json")
    a = s.add("세미나실", "2026-11-19", "14:00", "16:00", "김민수", NOW)
    assert s.cancel(a.id) == a and s.list() == [] and s.cancel(a.id) is None


def test_load_facilities(tmp_path):
    path = tmp_path / "facilities.yaml"
    path.write_text("- name: 세미나실\n  aliases: [세미나룸]\n  default_hours: 2\n  open: '09:00'\n  close: '18:00'\n", encoding="utf-8")
    assert load_facilities(path) == [Facility("세미나실", ("세미나룸",), 2, "09:00", "18:00")]
    assert load_facilities(tmp_path / "none.yaml") == []
```

- [ ] **Step 2: 실패 확인** — `.venv/bin/python -m pytest tests/test_reservation_store.py -q -p no:cacheprovider` → `ModuleNotFoundError: src.reservation_store`
- [ ] **Step 3: `src/reservation_store.py` 구현** — 쓰기는 같은 폴더의 임시 파일에 쓴 뒤 `os.replace`. `add` 는 쓰기 직전에 파일을 다시 읽어 겹침을 본다. id 는 겹치지 않으면 모양은 자유.
- [ ] **Step 4: `config/reservations/facilities.yaml` 작성** — 세미나실(별칭 `세미나룸`)·상담실, 둘 다 `default_hours: 2`, `open: '09:00'`, `close: '18:00'`.
- [ ] **Step 5: `tools/reservations.py` 작성** — `python -m tools.reservations list | cancel <id>`. `ReservationStore(reservation_dir() / "reservations.json")` 만 부른다(실기에서 같은 시간을 다시 쓰려고 지우는 용도).
- [ ] **Step 6: 통과 확인** — 같은 명령 → 6 passed
- [ ] **Step 7: 커밋** — `git add src/reservation_store.py config/reservations/facilities.yaml tools/reservations.py tests/test_reservation_store.py` · `feat(voice): 예약 저장 모듈 — 겹침·원자적 쓰기·깨진 파일은 덮어쓰지 않음, 점검 CLI`

---

### Task 2: 날짜 계산 · 낱말 있는지 보기 · 말로 읽기 (설계 5.2-2, 6절)

**Files:**
- Create: `src/reservation_time.py`
- Test: `tests/test_reservation_time.py`

**Interfaces:**
- Produces:
  - `resolve_date(expr: str, today: date) -> Optional[date]` — LLM 이 준 날짜 표현을 날짜로
  - `valid_hhmm(value: str) -> Optional[str]` — `"9:30"` → `"09:30"`, 시각이 아니면 `None`
  - `has_date_word(text: str) -> bool`, `has_time_word(text: str) -> bool`, `has_span_word(text: str) -> bool`
  - `speak_date(day: date, today: date) -> str`, `speak_time(hhmm: str) -> str`, `speak_range(start: str, end: str) -> str`
- 규칙(시험이 다 정하지 못하는 것)
  - 이 모듈은 글자에서 값을 읽지 않는다. `has_*` 는 그런 낱말이 있는지만 본다(LLM 이 지어낸 값을 거르는 데 쓴다).
  - `has_time_word`: 오전·오후·아침·낮·저녁·밤·새벽, 또는 수(한글 수사·숫자) + "시". "시간"의 "시"는 세지 않는다.
  - `has_span_word`: "까지" · "시간" · "동안".
  - 지난 날짜("11월 1일"을 11-18 에 말함)도 올해 날짜로 돌려준다 — 거절은 Task 4 가 한다.
  - `speak_range` 는 끝이 시작과 같은 오전·오후면 끝의 표시를 뺀다.

- [ ] **Step 1: 실패하는 시험 작성** — `tests/test_reservation_time.py`

```python
"""예약 날짜 계산·낱말 보기·말로 읽기(설계 2026-11-18 5.2·6절)."""
from datetime import date

import pytest

from src.reservation_time import (has_date_word, has_span_word, has_time_word, resolve_date, speak_date,
                                  speak_range, speak_time, valid_hhmm)

TODAY = date(2026, 11, 18)   # 수요일


@pytest.mark.parametrize("expr, expected", [
    ("오늘", date(2026, 11, 18)), ("내일", date(2026, 11, 19)), ("모레", date(2026, 11, 20)),
    ("글피", date(2026, 11, 21)), ("다음 주 월요일", date(2026, 11, 23)), ("다음주 월요일", date(2026, 11, 23)),
    ("이번 주 금요일", date(2026, 11, 20)), ("금요일", date(2026, 11, 20)), ("11월 20일", date(2026, 11, 20)),
    ("20일", date(2026, 11, 20)), ("11월 1일", date(2026, 11, 1)), ("2026-11-20", date(2026, 11, 20)),
    ("언젠가", None), ("", None),
])
def test_resolve_date(expr, expected):
    assert resolve_date(expr, TODAY) == expected


def test_valid_hhmm():
    assert valid_hhmm("14:00") == "14:00" and valid_hhmm("9:30") == "09:30"
    assert valid_hhmm("25:00") is None and valid_hhmm("오후 두 시") is None and valid_hhmm("") is None


def test_word_presence():
    assert has_date_word("내일 세 시") and has_date_word("11월 20일에") and has_date_word("금요일")
    assert not has_date_word("세미나실 예약해 줘.")
    assert has_time_word("내일 세 시") and has_time_word("14시에") and has_time_word("오후에 쓸게요")
    assert not has_time_word("세미나실 예약해 줘.") and not has_time_word("세미나실이 어디 있어?")
    assert not has_time_word("세 시간만 쓸게요")          # 쓰는 시간이지 시각이 아니다
    assert has_span_word("두 시부터 세 시까지") and has_span_word("세 시간만 쓸게요") and has_span_word("두 시간 동안")
    assert not has_span_word("내일 오후 두 시에 세미나실 예약해 줘.")


def test_speaking():
    assert [speak_date(date(2026, 11, d), TODAY) for d in (18, 19, 20, 23)] == ["오늘", "내일", "모레", "11월 23일"]
    assert speak_time("14:00") == "오후 두 시" and speak_time("13:30") == "오후 한 시 반"
    assert speak_time("10:10") == "오전 열 시 십 분" and speak_time("12:00") == "오후 열두 시"
    assert speak_range("14:00", "16:00") == "오후 두 시부터 네 시까지"
    assert speak_range("11:00", "13:00") == "오전 열한 시부터 오후 한 시까지"
```

- [ ] **Step 2: 실패 확인** — `.venv/bin/python -m pytest tests/test_reservation_time.py -q -p no:cacheprovider` → `ModuleNotFoundError`
- [ ] **Step 3: `src/reservation_time.py` 구현** — `langchain_intent_parser` 는 import 하지 않는다(langchain 을 끌어온다).
- [ ] **Step 4: 통과 확인** — 같은 명령 → 17 passed
- [ ] **Step 5: 커밋** — `feat(voice): 예약 날짜 계산·낱말 보기·말로 읽기`

---

### Task 3: 문장 9개 (설계 6절)

**Files:**
- Create: `src/reservation_phrases.py`
- Modify: `src/mission_phrases.py`(`standalone_prewarm`)
- Test: `tests/test_reservation_phrases.py`, `tests/test_mission_phrases.py`

**Interfaces:**
- Consumes: `reservation_time.speak_date`·`speak_range`
- Produces: 상수 `ASK_NAME`(①) · `DONE`(③) · `TAKEN`(④) · `ASK_WHEN`(⑤) · `NOT_RESERVED`(⑥) · `ASK_WHERE`(⑦) · `BAD_TIME`(⑧) · `UNAVAILABLE`(⑨) · `FIXED`(이 8개의 tuple) · `confirm_sentence(name: str, day: date, start: str, end: str, facility: str, today: date) -> str`(②)

- [ ] **Step 1: 실패하는 시험 작성** — `tests/test_reservation_phrases.py`

```python
"""예약 문장(설계 2026-11-18 6절) — 글자가 사용자 확정본과 같아야 한다."""
from datetime import date

from src import reservation_phrases as p


def test_fixed_sentences_are_the_confirmed_text():
    assert p.ASK_NAME == "성함을 말씀해 주세요."
    assert p.DONE == "네, 예약이 완료되었습니다. 이제 어디로 갈까요?"
    assert p.TAKEN == "그 시간에는 이미 예약이 있습니다. 다른 시간을 말씀해 주세요."
    assert p.ASK_WHEN == "언제로 예약할까요?"
    assert p.NOT_RESERVED == "예약하지 않았습니다."
    assert p.ASK_WHERE == "어디로 예약할까요?"
    assert p.BAD_TIME == "그 시간에는 예약할 수 없습니다. 다른 시간을 말씀해 주세요."
    assert p.UNAVAILABLE == "지금은 예약을 받을 수 없습니다."
    assert len(p.FIXED) == 8 and len(set(p.FIXED)) == 8


def test_confirm_sentence_is_the_demo_line():
    line = p.confirm_sentence("김민수", date(2026, 11, 19), "14:00", "16:00", "세미나실", date(2026, 11, 18))
    assert line == "김민수 님 이름으로 내일 오후 두 시부터 네 시까지 세미나실 예약할까요?"
```

- [ ] **Step 2: 실패 확인** → `ModuleNotFoundError`
- [ ] **Step 3: `src/reservation_phrases.py` 구현**
- [ ] **Step 4: 미리 합성** — `mission_phrases.standalone_prewarm` 이 돌려주는 목록에 `reservation_phrases.FIXED` 를 더한다(시그니처 그대로). `tests/test_mission_phrases.py` 에 `test_reservation_sentences_are_prewarmed` 를 더해, 그 파일의 기존 `standalone_prewarm` 호출과 같은 인자로 불러 8개가 모두 들어 있는지 본다.
- [ ] **Step 5: 통과 확인** — `.venv/bin/python -m pytest tests/test_reservation_phrases.py tests/test_mission_phrases.py -q -p no:cacheprovider`
- [ ] **Step 6: 커밋** — `feat(voice): 예약 문장 9개(사용자 확정본)·미리 합성 목록`

---

### Task 4: 예약 진행 상태 (설계 5.2·5.3)

**Files:**
- Create: `src/reservation_flow.py`
- Test: `tests/test_reservation_flow.py`

**Interfaces:**
- Consumes: Task 1 `ReservationStore`·`Facility`·`Reservation`·`ConflictError`, Task 2 전부, Task 3 전부, `src.question_record.OpenQuestion`, `src.schema.VicaIntent`
- Produces:
  - `@dataclass(frozen=True) ReserveCall(facility: str = "", date: str = "", start_time: str = "", end_time: str = "", name: str = "")` — LLM 이 부른 도구의 인자(빈 값 = 말하지 않음)
  - `@dataclass(frozen=True) FlowOutput(say: str, kind: str, saved: Optional[Reservation] = None)` — `kind` 는 `"open"`·`"yesno"`·`""`(`""` = 대답을 기다리지 않는 안내)
  - `start_gate(dialog_state: str, question: Optional[OpenQuestion], own_text: str) -> str` — `"ok"`·`"busy"`·`"mute"`
  - `class ReservationFlow(store, facilities: Sequence[Facility], clock: Callable[[], datetime] = datetime.now)`
    - `store`(속성) · `active: bool`(속성) · `asked_text: str`(지금 넘긴 질문 문장, 없으면 `""`)
    - `on_call(call: ReserveCall, text: str, gate: str) -> Optional[FlowOutput]` — 진행 중이면 `gate` 를 보지 않는다. 진행 중이 아니고 `"busy"` 면 ⑨, `"mute"` 면 `None`
    - `on_general(intent: VicaIntent) -> Optional[FlowOutput]` — 진행 중일 때 일반 해석 결과를 받는다(아래 규칙)
    - `on_question_record(text: str) -> None` — 미션의 질문 기록이 바뀔 때마다 부른다(`""` = 열린 질문 없음). 내가 넘긴 질문이 한 번 열렸다가(`text == asked_text`) 다른 것으로 바뀌면 진행 상태를 버린다. 새 질문을 넘기면 "열렸다"를 다시 센다.
    - `situation_line() -> str` — 진행 중이 아니면 `""`. 진행 중이면 한 줄: `[진행 중인 예약] 시설 세미나실 · 날짜 내일 · 시작 14:00 · 끝 16:00 · 이름 (아직 없음) · 방금 물은 것: "성함을 말씀해 주세요."` (없는 값은 `(아직 없음)`, 날짜는 `speak_date`)
    - `drop() -> None`
  - `to_intent(out: FlowOutput) -> VicaIntent` — 질문은 `intent="clarify"`, `reply=""`, `ask_text`, `ask_kind`, `need_confirm=False`, `confidence=1.0`. 안내는 `intent="question"`, `reply=문장`, `need_confirm=False`, `confidence=1.0`.
  - `created_event(res: Reservation, now: datetime) -> str` — JSON, 키 `event`(`"reservation_created"`)·`id`·`facility`·`date`·`start`·`end`·`name`·`source`·`timestamp`
- 규칙(시험이 다 정하지 못하는 것)
  - `start_gate`: `dialog_state == "idle"` 이고 (`question is None` 또는 `question.kind == "open"` 또는 `question.text == own_text`)면 `"ok"`. `"awaiting_user"` 와 순찰 상태(`src/patrol_sound.py` 의 `STATE_*`)는 `"mute"`. 나머지는 `"busy"`.
  - 값 받기는 설계 5.2-2 그대로: LLM 값을 쓰고, 글자에 근거가 없는 값만 버린다(`name` 은 글자에 그 이름이 있어야, `facility` 는 이름·별칭이 있어야, `date`·`start_time` 은 `has_date_word`·`has_time_word`, `end_time` 은 `has_span_word`). `date` 는 `resolve_date`, 시각은 `valid_hhmm`.
  - 묻는 순서: 시설(⑦) → 날짜·시각(⑤) → 시각 검사(⑧) → 겹침(④) → 이름(①) → 확인(②). ④·⑧ 뒤에는 시각을 비우고 시설·이름·날짜는 쥔다. 날짜가 이미 지난 날이면 날짜도 비운다.
  - `on_general`: `affirm` 이고 확인(②)을 기다리는 중이면 저장 → ③. `deny`·`cancel` 이면 ⑥ 뒤 버림. `navigate`·`pause`·`resume`·`wait`·`finish`·`help` 면 버리고 `None`. 그 밖(`unknown`·`question`·`clarify`·확인 단계가 아닌 `affirm`)은 그대로 두고 `None`.
  - 저장한 뒤와 거절 뒤에는 진행 상태를 닫는다. `ConflictError` 면 ④, `OSError` 면 ⑨ 뒤 버림.

- [ ] **Step 1: 실패하는 시험 작성** — `tests/test_reservation_flow.py`

```python
"""예약 진행 상태(설계 2026-11-18 5.2·5.3)."""
import json
from datetime import datetime

import pytest

from src import reservation_phrases as p
from src.question_record import OpenQuestion
from src.reservation_flow import (FlowOutput, ReservationFlow, ReserveCall, created_event, start_gate, to_intent)
from src.reservation_store import Facility, ReservationStore
from src.schema import VicaIntent

NOW = datetime(2026, 11, 18, 14, 3, 21)        # 수요일
FACILITIES = [Facility("세미나실", ("세미나룸",), 2, "09:00", "18:00"), Facility("상담실", (), 2, "09:00", "18:00")]
DEMO = "김민수 님 이름으로 내일 오후 두 시부터 네 시까지 세미나실 예약할까요?"
YES = VicaIntent(intent="affirm", need_confirm=False)
NO = VicaIntent(intent="deny", need_confirm=False)


def q(text, kind="open"):
    return OpenQuestion(id="q1", kind=kind, text=text, option_ids=(), opened_at=0.0)


@pytest.fixture
def flow(tmp_path):
    return ReservationFlow(ReservationStore(tmp_path / "reservations.json"), FACILITIES, clock=lambda: NOW)


def call(flow, text, **kw):
    return flow.on_call(ReserveCall(**kw), text, "ok")


def first(flow, text="내일 오후 두 시에 세미나실 예약해 줘."):
    return call(flow, text, facility="세미나실", date="내일", start_time="14:00")


def test_demo_script(flow):
    assert first(flow) == FlowOutput(p.ASK_NAME, "open")
    assert call(flow, "김민수", name="김민수") == FlowOutput(DEMO, "yesno")
    out = flow.on_general(YES)
    assert out.say == p.DONE and out.kind == "open"
    assert (out.saved.facility, out.saved.date, out.saved.start, out.saved.end, out.saved.name) == (
        "세미나실", "2026-11-19", "14:00", "16:00", "김민수")
    assert flow.store.list() == [out.saved] and not flow.active


def test_values_not_in_the_words_are_dropped(flow):
    out = call(flow, "세미나실 예약해 줘.", facility="세미나실", date="오늘", start_time="14:00", name="홍길동")
    assert out == FlowOutput(p.ASK_WHEN, "open")
    assert call(flow, "내일 오후 두 시", date="내일", start_time="14:00") == FlowOutput(p.ASK_NAME, "open")
    assert call(flow, "음", name="홍길동") == FlowOutput(p.ASK_NAME, "open")     # 글자에 없는 이름 — 다시 묻는다
    assert call(flow, "제 이름은 김민수예요.", name="김민수") == FlowOutput(DEMO, "yesno")


def test_the_models_time_is_used(flow):
    call(flow, "상담실 오늘 오후 네 시에 쓸 수 있어?", facility="상담실", date="오늘", start_time="16:00")
    out = call(flow, "이서연입니다.", name="이서연")
    assert out.say == "이서연 님 이름으로 오늘 오후 네 시부터 여섯 시까지 상담실 예약할까요?"


def test_end_time_only_when_spoken(flow):
    call(flow, "내일 두 시부터 세 시까지 세미나실 예약해 줘.", facility="세미나실", date="내일",
         start_time="14:00", end_time="15:00")
    assert "오후 두 시부터 세 시까지" in call(flow, "김민수", name="김민수").say
    flow.drop()
    call(flow, "모레 오후 두 시에 한 시간만 세미나실 쓸게요.", facility="세미나실", date="모레",
         start_time="14:00", end_time="15:00")
    assert "오후 두 시부터 세 시까지" in call(flow, "김민수", name="김민수").say
    flow.drop()
    first(flow, "내일 오후 두 시에 세미나실 예약해 줘.")          # 끝을 말하지 않았다
    flow.drop()
    call(flow, "내일 오후 두 시에 세미나실 예약해 줘.", facility="세미나실", date="내일", start_time="14:00", end_time="17:00")
    assert "오후 두 시부터 네 시까지" in call(flow, "김민수", name="김민수").say     # 지어낸 끝은 버리고 기본 2시간


def test_empty_call_asks_facility_first(flow):
    assert call(flow, "예약 좀 하려고.") == FlowOutput(p.ASK_WHERE, "open")
    assert call(flow, "세미나실이요.", facility="세미나실") == FlowOutput(p.ASK_WHEN, "open")


def test_taken_slot_then_another_time(flow):
    flow.store.add("세미나실", "2026-11-19", "14:00", "16:00", "이서연", NOW)
    assert first(flow) == FlowOutput(p.TAKEN, "open")
    assert call(flow, "오후 네 시", start_time="16:00") == FlowOutput(p.ASK_NAME, "open")   # 날짜(내일)는 쥐고 있다
    assert "오후 네 시부터 여섯 시까지" in call(flow, "김민수", name="김민수").say


def test_out_of_hours_and_past_are_refused(flow):
    late = call(flow, "내일 오후 다섯 시에 세미나실 예약해 줘.", facility="세미나실", date="내일", start_time="17:00")
    assert late == FlowOutput(p.BAD_TIME, "open")
    flow.drop()
    past = call(flow, "오늘 오전 아홉 시에 세미나실 예약해 줘.", facility="세미나실", date="오늘", start_time="09:00")
    assert past == FlowOutput(p.BAD_TIME, "open") and flow.store.list() == []


def test_gate(flow):
    assert flow.on_call(ReserveCall(facility="세미나실"), "세미나실 예약해 줘.", "busy") == FlowOutput(p.UNAVAILABLE, "")
    assert flow.on_call(ReserveCall(facility="세미나실"), "세미나실 예약해 줘.", "mute") is None
    assert not flow.active
    first(flow)
    assert flow.on_call(ReserveCall(name="김민수"), "김민수", "busy") == FlowOutput(DEMO, "yesno")   # 진행 중이면 문을 다시 보지 않는다
    assert start_gate("idle", None, "") == "ok" and start_gate("idle", q("네?"), "") == "ok"
    assert start_gate("idle", q("화장실로 안내해드릴까요?", "confirm"), "") == "busy"
    assert start_gate("idle", q(DEMO, "yesno"), DEMO) == "ok"
    assert start_gate("navigating", None, "") == "busy" and start_gate("", None, "") == "busy"
    assert start_gate("awaiting_user", None, "") == "mute" and start_gate("patrolling", None, "") == "mute"


def test_confirm_no_and_second_yes_does_not_save_twice(flow):
    first(flow)
    call(flow, "김민수", name="김민수")
    assert flow.on_general(YES).saved is not None
    assert flow.on_general(YES) is None and len(flow.store.list()) == 1
    first(flow, "모레 오후 두 시에 세미나실 예약해 줘.")
    call(flow, "김민수", name="김민수")
    assert flow.on_general(NO) == FlowOutput(p.NOT_RESERVED, "") and not flow.active and len(flow.store.list()) == 1


def test_save_failure_says_unavailable_not_done(flow):
    first(flow)
    call(flow, "김민수", name="김민수")
    flow.store.path.write_text('[{"id": "r1", "facil', encoding="utf-8")
    assert flow.on_general(YES) == FlowOutput(p.UNAVAILABLE, "") and not flow.active


def test_general_result_while_active(flow):
    first(flow)                                                   # 이름을 기다리는 중
    for kind in ("unknown", "question", "clarify", "affirm"):
        assert flow.on_general(VicaIntent(intent=kind, need_confirm=False)) is None and flow.active
    assert flow.on_general(VicaIntent(intent="navigate", destination_candidate="화장실")) is None and not flow.active
    first(flow)
    assert flow.on_general(VicaIntent(intent="cancel", need_confirm=False)) == FlowOutput(p.NOT_RESERVED, "")


def test_question_closing_drops_the_draft(flow):
    first(flow)
    flow.on_question_record("")                 # 앞 질문("네?")이 닫힘 — 내 질문은 아직 열린 적 없다
    assert flow.active
    flow.on_question_record(p.ASK_NAME)         # 미션이 내 질문을 열었다
    call(flow, "김민수", name="김민수")          # 새 질문(②)을 넘겼다
    flow.on_question_record("")                 # ①이 닫힘 — ②는 아직 안 열렸다
    assert flow.active
    flow.on_question_record(DEMO)
    flow.on_question_record("네?")              # 대답 없이 다시 불렀다
    assert not flow.active


def test_same_call_twice_asks_the_same_question(flow):
    assert first(flow) == first(flow) == FlowOutput(p.ASK_NAME, "open")


def test_situation_line(flow):
    assert flow.situation_line() == ""
    first(flow)
    line = flow.situation_line()
    assert line.startswith("[진행 중인 예약]") and "세미나실" in line and "내일" in line and "14:00" in line
    assert "(아직 없음)" in line and p.ASK_NAME in line


def test_to_intent_and_event(flow):
    ask = to_intent(FlowOutput(p.ASK_NAME, "open"))
    assert (ask.intent, ask.reply, ask.ask_text, ask.ask_kind, ask.need_confirm) == ("clarify", "", p.ASK_NAME, "open", False)
    say = to_intent(FlowOutput(p.NOT_RESERVED, ""))
    assert (say.intent, say.reply, say.ask_text, say.need_confirm) == ("question", p.NOT_RESERVED, "", False)
    res = flow.store.add("세미나실", "2026-11-19", "14:00", "16:00", "김민수", NOW)
    event = json.loads(created_event(res, NOW))
    assert event["event"] == "reservation_created" and event["id"] == res.id and event["name"] == "김민수"
    assert set(event) == {"event", "id", "facility", "date", "start", "end", "name", "source", "timestamp"}
```

- [ ] **Step 2: 실패 확인** — `.venv/bin/python -m pytest tests/test_reservation_flow.py -q -p no:cacheprovider` → `ModuleNotFoundError`
- [ ] **Step 3: `src/reservation_flow.py` 구현**
- [ ] **Step 4: 통과 확인** — 같은 명령 → 15 passed
- [ ] **Step 5: 커밋** — `feat(voice): 예약 진행 상태 — 값은 LLM, 코드는 지어낸 값 거르기·겹침·저장 뒤에만 완료 문장`

---

### Task 5: 도구 정의와 글자 경로 도구 출력 (설계 5.1)

**Files:**
- Create: `src/tool_calls.py`
- Modify: `src/langchain_intent_parser.py`(`_get_structured_llm` 아래 새 함수, `get_backend_manager`, `parse_intent`), `.env.example`
- Test: `tests/test_tool_calls.py`, `tests/test_parser_tools.py`

**Interfaces:**
- Consumes: Task 4 `ReserveCall`, Task 1 `default_facilities`
- Produces:
  - `tool_calls.SET_INTENT = "set_intent"`, `tool_calls.RESERVE = "reserve_facility"`
  - `tool_calls.text_tools_on() -> bool`(`VICA_TOOLS_TEXT=on`), `tool_calls.audio_tools_on() -> bool`(`VICA_TOOLS_AUDIO=on`)
  - `tool_calls.reserve_tool(facility_names: Sequence[str], with_heard_text: bool = False) -> dict` — `{"type": "function", "function": {"name", "description", "parameters"}}`. `parameters.properties` 는 `facility`(enum = 이름들)·`date`·`start_time`·`end_time`·`name`(+ `heard_text`), `required` 는 `[]`(with_heard_text 면 `["heard_text"]`).
  - `tool_calls.set_intent_tool(parameters: dict) -> dict` — 같은 꼴, 이름 `set_intent`
  - `tool_calls.reserve_call(args: dict) -> ReserveCall` — 모르는 키는 버리고 `None`·비문자열은 `""`
  - `langchain_intent_parser._set_intent_tool() -> dict` — `tool_calls.set_intent_tool(_IntentDraft 의 JSON 스키마)`
  - `langchain_intent_parser._draft_from_tool_message(msg) -> _IntentDraft | ReserveCall` — `msg.tool_calls` 의 첫 호출. 호출이 없거나 이름을 모르거나 인자가 `_IntentDraft` 와 안 맞으면 `ValueError`.
  - `langchain_intent_parser._get_tool_llm(model: str, *, timeout: float = 15, max_retries: int = 1)` — `.invoke(messages) -> _IntentDraft | ReserveCall`
  - `parse_intent(...) -> VicaIntent | ReserveCall` — LLM 이 `reserve_facility` 를 부르면 `_finalize` 를 거치지 않고 `ReserveCall` 을 그대로 돌려준다.
- 도구 설명 글자(설계 5.1, 규칙을 여기에 적는다):
  - `reserve_facility`: `시설을 예약해 달라는 말, 그 시간에 쓸 수 있는지 묻는 말, 진행 중인 예약에 대한 대답(이름·날짜·시각·시설)에 호출한다. 사용자가 이번에 말한 값만 넣는다. 어디로 가자, 안내해 달라는 말은 이동이므로 set_intent 를 호출한다. 실제 예약은 사용자 확인 뒤 시스템이 한다.`
  - `date`: `사용자가 말한 날짜 표현 그대로. 예: 오늘, 내일, 모레, 다음 주 월요일, 11월 20일` · `start_time`: `시작 시각, 24시간 HH:MM` · `end_time`: `끝 시각이나 쓰는 시간을 말했을 때만, 끝 시각을 24시간 HH:MM 으로` · `name`: `예약자 이름. 사용자가 말했을 때만` · `heard_text`: `들린 말을 한국어로 그대로 적는다`
  - `set_intent`: `시설 예약이 아닌 모든 말의 의도를 정한다.`
- `set_intent` 의 `parameters` 는 `_IntentDraft.model_json_schema()` 에서 만든다(칸과 설명이 지금 구조화 출력과 같아야 한다).
- 도구 출력은 `PROVIDER == "openai"` 이고 `text_tools_on()` 일 때만 쓴다. `get_backend_manager` 의 클라우드 쪽과 `parse_intent(model=…)` 직접 호출 쪽 둘 다 같은 조건으로 고른다. 로컬(`_build_local_structured`)은 그대로.

- [ ] **Step 1: 실패하는 시험 작성**

`tests/test_tool_calls.py`
```python
"""도구 정의(설계 2026-11-18 5.1)."""
from src import tool_calls
from src.reservation_flow import ReserveCall


def test_reserve_tool_shape():
    tool = tool_calls.reserve_tool(["세미나실", "상담실"])["function"]
    assert tool["name"] == "reserve_facility" and "이동" in tool["description"] and "진행 중인 예약" in tool["description"]
    props = tool["parameters"]["properties"]
    assert set(props) == {"facility", "date", "start_time", "end_time", "name"}
    assert props["facility"]["enum"] == ["세미나실", "상담실"] and tool["parameters"]["required"] == []
    heard = tool_calls.reserve_tool(["세미나실"], with_heard_text=True)["function"]["parameters"]
    assert "heard_text" in heard["properties"] and heard["required"] == ["heard_text"]


def test_reserve_call_keeps_only_known_text_values():
    call = tool_calls.reserve_call({"facility": "세미나실", "date": "내일", "start_time": "14:00", "name": None, "junk": 1})
    assert call == ReserveCall(facility="세미나실", date="내일", start_time="14:00")


def test_switches_default_off(monkeypatch):
    monkeypatch.delenv("VICA_TOOLS_TEXT", raising=False)
    monkeypatch.delenv("VICA_TOOLS_AUDIO", raising=False)
    assert not tool_calls.text_tools_on() and not tool_calls.audio_tools_on()
    monkeypatch.setenv("VICA_TOOLS_TEXT", "on")
    assert tool_calls.text_tools_on() and not tool_calls.audio_tools_on()
```

`tests/test_parser_tools.py`
```python
"""글자 경로 도구 출력(설계 2026-11-18 5.1)."""
import pytest
from langchain_core.messages import AIMessage

from src import langchain_intent_parser as parser
from src.reservation_flow import ReserveCall
from src.schema import DestinationData


def msg(*calls):
    return AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": f"c{i}"} for i, (n, a) in enumerate(calls)])


def test_tool_message_shapes():
    draft = parser._draft_from_tool_message(msg(("set_intent", {"intent": "navigate", "destination_candidate": "화장실"})))
    assert draft.intent == "navigate" and draft.destination_candidate == "화장실"
    call = parser._draft_from_tool_message(msg(("reserve_facility", {"facility": "세미나실", "date": "내일"}),
                                               ("set_intent", {"intent": "unknown"})))
    assert call == ReserveCall(facility="세미나실", date="내일")
    assert parser._draft_from_tool_message(msg(("reserve_facility", {}))) == ReserveCall()
    for bad in (msg(), msg(("go_to", {"location": "화장실"})), msg(("set_intent", {"intent": "fly"}))):
        with pytest.raises(ValueError):
            parser._draft_from_tool_message(bad)


def test_parse_intent_hands_back_the_reserve_call(monkeypatch):
    class Manager:
        has_local = False

        def invoke(self, messages):
            return ReserveCall(facility="세미나실", date="내일", start_time="14:00")

    monkeypatch.setattr(parser, "get_backend_manager", lambda: Manager())
    dests = [DestinationData(id="restroom", name="화장실")]
    out = parser.parse_intent("내일 오후 두 시에 세미나실 예약해 줘.", dests)
    assert out == ReserveCall(facility="세미나실", date="내일", start_time="14:00")


def test_set_intent_tool_has_the_draft_fields():
    names = set(parser._set_intent_tool()["function"]["parameters"]["properties"])
    assert names == set(parser._IntentDraft.model_fields)


def test_local_backend_is_untouched_by_the_switch(monkeypatch):
    monkeypatch.setenv("VICA_TOOLS_TEXT", "on")
    import inspect
    assert "bind_tools" not in inspect.getsource(parser._build_local_structured)
```

- [ ] **Step 2: 실패 확인** — `.venv/bin/python -m pytest tests/test_tool_calls.py tests/test_parser_tools.py -q -p no:cacheprovider`
- [ ] **Step 3: `src/tool_calls.py` 구현**
- [ ] **Step 4: `langchain_intent_parser.py`** — `_set_intent_tool()`, `_draft_from_tool_message`, `_get_tool_llm`(`ChatOpenAI(model, temperature=0, timeout, max_retries).bind_tools([_set_intent_tool(), reserve_tool(이름들)], tool_choice="required")`, 시설 이름은 `default_facilities()`), 조건에 따라 고르는 자리 두 곳, `parse_intent` 의 `isinstance(draft, ReserveCall)` 갈래.
- [ ] **Step 5: `.env.example`** 에 `VICA_TOOLS_TEXT=off`·`VICA_TOOLS_AUDIO=off`·`VICA_RESERVATION_DIR` 설명 한 줄씩.
- [ ] **Step 6: 통과 확인** — 새 두 파일 통과, 그리고 전체 `.venv/bin/python -m pytest -q -p no:cacheprovider` → 기존 1812 + 새 시험 모두 통과
- [ ] **Step 7: 커밋** — `feat(voice): 글자 경로 도구 출력 — set_intent·reserve_facility, 스위치 VICA_TOOLS_TEXT(기본 꺼짐), 로컬은 그대로`

---

### Task 6: 음성 노드 배선 — 글자 경로 (설계 5.3·5.5)

**Files:**
- Modify: `src/reservation_flow.py`(갈림 함수), `src/ros_node.py`(`__init__`, `_on_user_text`, `_on_question`, `_publish_intent`, `_situation_block`, `_shadow_text`)
- Test: `tests/test_reservation_flow.py`(갈림), `tests/test_reservation_wiring.py`(새, `tests/test_one_voice_wiring.py` 와 같은 소스 구조 시험)

**Interfaces:**
- Consumes: Task 4·5 전부
- Produces:
  - `reservation_flow.handle_result(flow: Optional[ReservationFlow], result, text: str, *, gate: str, tools_used: bool) -> tuple[Optional[FlowOutput], Optional[VicaIntent]]`
    - `flow is None`: `(None, result)`.
    - `result` 가 `ReserveCall`: `(flow.on_call(result, text, gate), None)`.
    - 진행 중: `flow.on_general(result)` 가 값을 주면 `(그 값, None)`.
    - `tools_used` 가 거짓이고 글자에 `예약` 이 있고 `result.intent` 가 `affirm`·`deny`·`cancel`·`pause`·`resume` 이 아니고 `safety_flag != "emergency"`: `gate == "mute"` 면 `(None, None)`, 아니면 `(FlowOutput(UNAVAILABLE, ""), None)`.
    - 그 밖: `(None, result)`.
  - `ros_node`: `self._reservation: Optional[ReservationFlow]`(두 스위치 중 하나라도 켜졌을 때만 만든다), 발행자 `self._reservation_pub`(`std_msgs/String`, `/vica/reservation/event`, depth 10), `_publish_intent(..., scripted: bool = False)`, `_reservation_say(self, out: FlowOutput, text: str, link_text: str = "") -> None`
- 배선 규칙
  - `_situation_block`: 진행 중이면 돌려주는 글 끝에 `flow.situation_line()` 을 한 줄 더한다.
  - `_on_user_text`: `parse_intent` 결과를 `handle_result` 에 넣는다. `gate = start_gate(self._robot_state.dialog_state, self._question, flow.asked_text)`, `tools_used = text_tools_on() and self._backend.state is not BackendState.LOCAL`. `FlowOutput` 이면 `_reservation_say`, 의도면 지금처럼 `_publish_intent`, 둘 다 `None` 이면 아무것도 내지 않는다. 어느 쪽이든 `self._last_text_publish_t`·`_last_text_publish_intent` 는 지금처럼 채운다.
  - `_reservation_say`: `out.saved` 가 있으면 먼저 `created_event` 를 발행하고, `self._publish_intent(to_intent(out), text, scripted=True, link_text=link_text)`.
  - `_publish_intent` 의 `scripted=True`: 재청취 기각(`is_relisten_noise`)·`_apply_misheard_rule`·`quiet_for_mission` 을 건너뛴다. 대화 기록의 로봇 줄은 `ask_text or reply`.
  - `_on_question` 은 질문 기록을 받을 때마다 `self._reservation.on_question_record(msg.text if msg.id else "")` 를 부른다(흐름이 있을 때).
  - `self._history.begin_turn(...)` 이 참(대화 맥락을 비움)인 두 자리에서 `self._reservation.drop()` 을 부른다.
  - `_shadow_text` 는 `ReserveCall` 이 와도 죽지 않고 `intent=reserve_facility` 로 로그만 남긴다.

- [ ] **Step 1: 실패하는 시험 작성**

`tests/test_reservation_flow.py` 에 더함
```python
from src.reservation_flow import handle_result


def test_routing(flow):
    nav = VicaIntent(intent="navigate", destination_candidate="화장실", matched_destination_id="restroom")
    assert handle_result(None, nav, "화장실로 가자.", gate="ok", tools_used=True) == (None, nav)
    out, passed = handle_result(flow, ReserveCall(facility="세미나실", date="내일", start_time="14:00"),
                                "내일 오후 두 시에 세미나실 예약해 줘.", gate="ok", tools_used=True)
    assert out == FlowOutput(p.ASK_NAME, "open") and passed is None
    out, passed = handle_result(flow, ReserveCall(name="김민수"), "김민수", gate="ok", tools_used=True)
    assert out == FlowOutput(DEMO, "yesno") and passed is None
    out, passed = handle_result(flow, YES, "그래", gate="ok", tools_used=True)
    assert out.say == p.DONE and passed is None                   # 예약이 받은 승낙은 미션에 보내지 않는다
    assert handle_result(flow, nav, "화장실로 가자.", gate="ok", tools_used=True) == (None, nav)


def test_reservation_word_without_tools_is_refused(flow):
    nav = VicaIntent(intent="navigate", destination_candidate="세미나실")
    assert handle_result(flow, nav, "세미나실 예약해 줘.", gate="ok", tools_used=False) == (FlowOutput(p.UNAVAILABLE, ""), None)
    assert handle_result(flow, nav, "세미나실 예약해 줘.", gate="mute", tools_used=False) == (None, None)
    assert handle_result(flow, nav, "예약은 됐고 화장실 가자.", gate="ok", tools_used=True) == (None, nav)
    assert handle_result(flow, YES, "네 예약이요", gate="ok", tools_used=False) == (None, YES)
```

`tests/test_reservation_wiring.py` — `test_one_voice_wiring.py` 의 `SRC`·`_method`·`_calls`·`_branch_tests` 도우미를 가져다 쓴다.
```python
def test_model_result_goes_through_handle_result():
    assert _calls(_method("_on_user_text"), "handle_result")
    body = ast.unparse(_method("_on_user_text"))
    assert body.index("parse_intent(") < body.index("handle_result(")


def test_situation_block_carries_the_draft_line():
    assert "situation_line(" in ast.unparse(_method("_situation_block"))


def test_scripted_skips_the_three_filters():
    node = _method("_publish_intent")
    assert "scripted" in [a.arg for a in node.args.args + node.args.kwonlyargs]
    for dotted in ("is_relisten_noise", "self._apply_misheard_rule", "quiet_for_mission"):
        call = _calls(node, dotted)[0]
        assert any("scripted" in test for test, _ in _branch_tests(node, call)), dotted


def test_event_is_published_before_the_sentence():
    body = ast.unparse(_method("_reservation_say"))
    assert body.index("self._reservation_pub.publish(") < body.index("self._publish_intent(")
    assert "scripted=True" in body


def test_flow_is_built_only_when_a_switch_is_on():
    src = SRC.read_text()
    assert "text_tools_on()" in src and "audio_tools_on()" in src and "/vica/reservation/event" in src


def test_question_record_and_cleared_context_reach_the_flow():
    assert "on_question_record(" in ast.unparse(_method("_on_question"))
    assert SRC.read_text().count("self._reservation.drop()") >= 2
```

- [ ] **Step 2: 실패 확인** — `.venv/bin/python -m pytest tests/test_reservation_flow.py tests/test_reservation_wiring.py -q -p no:cacheprovider`
- [ ] **Step 3: `reservation_flow.py` 에 `handle_result` 구현**
- [ ] **Step 4: `ros_node.py` 배선** — 위 배선 규칙대로.
- [ ] **Step 5: 통과 확인** — 두 파일 통과 뒤 전체 시험 통과(스위치가 꺼진 기본 상태에서 기존 시험이 그대로여야 한다)
- [ ] **Step 6: 커밋** — `feat(voice): 예약 흐름 배선(글자 경로) — [지금 상황]에 진행 중인 예약, 코드가 만든 문장은 기각·대꾸 바꾸기를 건너뜀, /vica/reservation/event`

---

### Task 7: 재생 측정 도구 · 글자 경로 켜기 (설계 7절)

**Files:**
- Create: `tools/reserve_cases.py`, `tools/reserve_replay.py`
- Modify: `tools/answer_replay.py`(`--tools` 인자, `replay` 의 결과 처리), `docs/superpowers/specs/2026-11-18-facility-reservation-tool-calling-design.md`(7절에 결과)
- Test: `tests/test_reserve_replay.py`, `tests/test_answer_replay.py`

**Interfaces:**
- Consumes: Task 4·5·6
- Produces:
  - `tools.reserve_cases.CASES: list[Case]` — `Case(id: str, setup: tuple, text: str, expect: dict)`. `setup` 은 그 발화 전에 진행 상태에 넣을 `(ReserveCall, 글자)` 들(빈 tuple = 첫 요청).
    - 첫 요청 15개: `~/vica_data/tool_call_test_20261118/measure.py` 의 `ROWS` 가운데 R1~R12(R8 의 이름은 `정현우`로 바꾼다 — `정현우 이름으로 내일 오전 열한 시 상담실 예약해 줘.`), 그리고 E1 `내일 두 시부터 세 시까지 세미나실 예약해 줘.`(14:00~15:00) · E2 `모레 오전 열 시부터 열두 시까지 상담실 예약해 줘.`(10:00~12:00) · E3 `내일 오후 두 시에 한 시간만 세미나실 쓸게요.`(14:00~15:00).
    - 그 밖 13개: 같은 파일의 N1~N13.
    - 이어진 대답 10개(F1~F10, `setup` 으로 진행 상태를 만든 뒤 `[지금 상황]` 한 줄과 넘긴 질문을 붙임): 이름 질문에 `김민수` · `이서연입니다.` · `제 이름은 박준호예요.` · `아니, 됐어.`(거절) · `화장실 먼저 가자.`(이동), ⑤ 뒤 `내일 오후 세 시`, ⑦ 뒤 `세미나실이요.`, 확인에 `그래.`(승낙) · `아니요.`(거절) · `아니 네 시로 해 줘.`(시작 16:00 으로 다시 확인).
  - `tools.reserve_replay.summarize(rows: list[dict]) -> dict` — 키 `first_ok`·`first_total`·`stray_reserve`·`follow_ok`·`follow_total`
  - `python -m tools.reserve_replay --path gpt|realtime` — 운영 지시문(`parse_intent` 그대로)에 `situation=flow.situation_line()` 과 대화 기록(넘긴 질문을 로봇 줄로)을 붙여 돌리고, 결과를 `handle_result` 에 넣어 진행 상태와 저장 결과로 채점한다. `첫 요청 n/15 · 그 밖에서 reserve 호출 n/13 · 이어진 대답 n/10 · 평균 응답` 을 찍고 `~/vica_data/tool_call_test_20261118/reserve_replay_<경로>_<시각>.jsonl` 에 남긴다.
  - `tools.answer_replay`: `--tools` 를 주면 import 전에 `VICA_TOOLS_TEXT=on`·`VICA_TOOLS_AUDIO=on` 을 넣는다. 해석 결과가 `ReserveCall` 인 행은 `wrong` 으로 센다(208행에는 예약 발화가 없다).

- [ ] **Step 1: 실패하는 시험 작성**

`tests/test_reserve_replay.py`
```python
"""예약 묶음 재생의 채점(설계 2026-11-18 7절)."""
from tools.reserve_cases import CASES
from tools.reserve_replay import summarize


def test_case_set_shape():
    kinds = [c.id[0] for c in CASES]
    assert len(CASES) == 38 and len({c.id for c in CASES}) == 38
    assert kinds.count("R") == 12 and kinds.count("E") == 3 and kinds.count("N") == 13 and kinds.count("F") == 10
    assert all(c.setup for c in CASES if c.id.startswith("F")) and not any(c.setup for c in CASES if c.id[0] in "REN")


def test_summary_counts():
    rows = [{"id": "R1", "ok": True, "reserve_called": True, "wall_s": 1.0},
            {"id": "E1", "ok": False, "reserve_called": True, "wall_s": 1.0},
            {"id": "N3", "ok": False, "reserve_called": True, "wall_s": 1.0},
            {"id": "F1", "ok": True, "reserve_called": True, "wall_s": 1.0},
            {"id": "F8", "ok": False, "reserve_called": False, "wall_s": 1.0}]
    s = summarize(rows)
    assert (s["first_ok"], s["first_total"], s["stray_reserve"], s["follow_ok"], s["follow_total"]) == (1, 2, 1, 1, 2)
```

`tests/test_answer_replay.py` 에 더함 — 기존 `replay` 시험과 같은 가짜 행으로:
```python
def test_reserve_call_on_a_labeled_row_counts_as_wrong():
    from src.reservation_flow import ReserveCall
    run = replay(ROWS, lambda row: (ReserveCall(facility="세미나실"), 0.1), lambda path: None)
    assert run["counts"].get("wrong", 0) == len(ROWS)
```
(`ROWS`·`replay` 의 인자 꼴은 그 파일의 기존 시험을 따른다.)

- [ ] **Step 2: 실패 확인**
- [ ] **Step 3: 구현** — `tools/reserve_cases.py`, `tools/reserve_replay.py`, `tools/answer_replay.py`.
- [ ] **Step 4: 통과 확인** — 두 시험 파일과 전체 시험
- [ ] **Step 5: 커밋** — `feat(voice): 예약 묶음 재생·208행 재생의 도구 출력 스위치`
- [ ] **Step 6: 측정(사용자에게 알리고 실행 — OpenAI API 약 250번)**
  - `.venv/bin/python -m tools.reserve_replay --path gpt`
    합격: 첫 요청 **15/15**, 그 밖에서 reserve 호출 **0/13**, 이어진 대답 **10/10**.
    이어진 대답이 여럿 틀리면(`[지금 상황]` 한 줄만으로 대답을 잇지 못함) 켜지 않고, 대화 기록에 도구 호출 기록을 넣는 방식으로 바꾸는 안을 수치와 함께 사용자에게 보고한다(설계 7절).
  - `.venv/bin/python -m tools.answer_replay --path gpt --tools --labels ~/vica_data/voice_answer_audit_20261113/labeled.csv --rounds 1`
    합격: 208행(`confirmed == 'ear-1117'` 행을 `tools.answer_grade.grade` 로 다시 채점한 값) 잘못된 행동 **14 이하**(11-18 기준 10~12 + 2).
  - 결과를 설계서 7절 끝에 표로 적고 커밋(`docs(voice): 예약 도구 출력 재생 결과 — gpt`). 기준을 못 넘으면 켜지 않고 틀린 행을 보고한다.
- [ ] **Step 7: 켜기와 실기(사용자와 함께)** — 사용자가 `.env` 에 `VICA_TOOLS_TEXT=on` 을 넣고 글자 모드(`VICA_INTENT_INPUT=text`, `VICA_LLM_PROVIDER=openai`)로 음성 노드를 다시 띄운다. 주행 없이 설계 7절 장면을 한다: 대본 그대로 · 이미 예약 있음 · 확인에 "아니" · 이름을 잘못 들었을 때 "아니" · 이름 질문 중 "화장실로 가자" · 날짜 없이 "세미나실 예약해 줘" · "두 시부터 세 시까지" · 이름 질문에 대답 없이 지나간 뒤 다시 요청 · 오프라인(로컬)에서 "세미나실 예약해 줘"(⑨).
  확인: `ros2 topic echo /vica/reservation/event --once`(저장한 내용과 같은가), `python -m tools.reservations list`, 음성·미션 로그에서 한 질문에 두 목소리 0·저장 없는 ③ 0. 결과는 `~/vica_data/field_test_<날짜>/RESULT.md`.

---

### Task 8: Realtime 도구 출력 (설계 5.1)

**Files:**
- Modify: `src/realtime_intent.py`(`RealtimeResult`, `_session_config`, `_ask`, `_parse_done`, 새 `ask_text`), `src/langchain_intent_parser.py`(`parse_intent_audio`)
- Test: `tests/test_realtime_intent.py`, `tests/test_parser_audio.py`

**Interfaces:**
- Consumes: Task 5 `tool_calls`, Task 4 `ReserveCall`
- Produces:
  - `RealtimeResult` 에 `tool: str = "set_intent"` 칸(맨 뒤).
  - `audio_tools_on()` 이면 세션 `tools` = `[build_intent_tool(), 평평하게 편 reserve_tool(이름들, with_heard_text=True)]`, `tool_choice` = `"required"`(세션과 `response.create` 둘 다). 꺼져 있으면 지금 그대로(`set_intent` 강제).
  - `_parse_done(done) -> tuple[str, dict, dict]` — `(도구 이름, 인자, usage)`. `set_intent`·`reserve_facility`(켜졌을 때만) 말고는 `ValueError("응답에 도구 호출이 없다")`.
  - `RealtimeIntentClient.ask_text(text: str, history, instructions: str) -> RealtimeResult` — 소리 대신 `input_text` 항목을 넣는 재생용(`tools/reserve_replay --path realtime` 이 쓴다). 운영 경로는 부르지 않는다.
  - `parse_intent_audio(...) -> tuple[VicaIntent | ReserveCall, str, float, dict]` — 도구가 `reserve_facility` 면 `(reserve_call(인자), heard, 지연, {"src": "llm", "usage": …})`. `_audio_shared_guess`·`deliver_audio_draft` 는 거치지 않는다.

- [ ] **Step 1: 실패하는 시험 작성** — `tests/test_realtime_intent.py` 의 기존 가짜 연결(`types.SimpleNamespace` 로 만든 `response.done`)을 그대로 써서 더한다.

```python
class TestTools:
    def test_off_keeps_the_forced_single_tool(self, monkeypatch):
        monkeypatch.delenv("VICA_TOOLS_AUDIO", raising=False)
        cfg = RealtimeIntentClient(connect=lambda: None)._session_config()
        assert [t["name"] for t in cfg["tools"]] == ["set_intent"]
        assert cfg["tool_choice"] == {"type": "function", "name": "set_intent"}

    def test_on_offers_both_and_requires_one(self, monkeypatch):
        monkeypatch.setenv("VICA_TOOLS_AUDIO", "on")
        cfg = RealtimeIntentClient(connect=lambda: None)._session_config()
        assert [t["name"] for t in cfg["tools"]] == ["set_intent", "reserve_facility"]
        assert cfg["tool_choice"] == "required" and "heard_text" in cfg["tools"][1]["parameters"]["properties"]

    def test_parse_done_names_the_tool(self, monkeypatch):
        monkeypatch.setenv("VICA_TOOLS_AUDIO", "on")
        done = fake_done("reserve_facility", {"facility": "세미나실", "date": "내일", "heard_text": "내일 세미나실 예약해 줘"})
        name, args, _usage = RealtimeIntentClient._parse_done(done)
        assert name == "reserve_facility" and args["facility"] == "세미나실"
        with pytest.raises(ValueError):
            RealtimeIntentClient._parse_done(fake_done("go_to", {"location": "화장실"}))
```
(`fake_done(name, args)` 는 그 파일의 기존 가짜 `response.done` 만들기를 함수로 뺀 것.)

`tests/test_parser_audio.py` 에 더함 — 기존 시험이 `get_realtime_client` 를 바꿔 끼우는 방식 그대로:
```python
def test_reserve_tool_result_is_handed_back(monkeypatch):
    result = RealtimeResult(draft={"facility": "세미나실", "date": "내일", "start_time": "14:00"},
                            heard_text="내일 오후 두 시에 세미나실 예약해 줘", latency_sec=1.0, tool="reserve_facility")
    monkeypatch.setattr(parser, "get_realtime_client", lambda: types.SimpleNamespace(ask=lambda *a, **k: result))
    out, heard, dt, info = parser.parse_intent_audio(b"\x00\x00", DESTS)
    assert out == ReserveCall(facility="세미나실", date="내일", start_time="14:00")
    assert heard == "내일 오후 두 시에 세미나실 예약해 줘" and info["src"] == "llm"
```

- [ ] **Step 2: 실패 확인**
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — 두 파일과 전체 시험
- [ ] **Step 5: 커밋** — `feat(voice): Realtime 도구 출력 — set_intent·reserve_facility, 스위치 VICA_TOOLS_AUDIO(기본 꺼짐)`

---

### Task 9: 음성 노드 배선 — 소리 경로 (설계 5.3)

**Files:**
- Modify: `src/ros_node.py`(`_on_user_audio`)
- Test: `tests/test_reservation_wiring.py`

**Interfaces:**
- Consumes: Task 6 `handle_result`·`_reservation_say`, Task 8 `parse_intent_audio`
- 배선 규칙
  - 예약 질문이 열려 있어도 소리 경로는 평소대로 Realtime 을 부른다(D12). `_situation_block()` 이 진행 중인 예약 한 줄을 이미 싣는다(Task 6).
  - `parse_intent_audio` 결과는, 지금 있는 두 가드(멈춤 말, `self._last_text_publish_t > turn_started` 로 글자가 먼저 발행했는지) **뒤에서** `handle_result(..., heard, gate=…, tools_used=audio_tools_on())` 를 거친다. `FlowOutput` 이면 `self._audio_turn.update(handled=True, …)` 뒤 `_reservation_say(out, heard)`. 둘 다 `None` 이면 `handled=True` 만 남기고 끝낸다.
  - `[RT]` 로그 줄은 `ReserveCall` 일 때 `intent=reserve_facility` 로 찍는다.

- [ ] **Step 1: 실패하는 시험 작성** — `tests/test_reservation_wiring.py` 에 더함

```python
def test_audio_reservation_is_under_the_text_first_guard():
    body = ast.unparse(_method("_on_user_audio"))
    guard = body.index("self._last_text_publish_t > turn_started", body.index("detect_emergency("))
    assert guard < body.index("handle_result(")
    assert _calls(_method("_on_user_audio"), "self._reservation_say")


def test_audio_path_is_not_bypassed_for_reservation_answers():
    body = ast.unparse(_method("_on_user_audio"))
    assert "asked_text" not in body[:body.index("parse_intent_audio(")]      # 예약 질문이라고 소리 경로를 비키지 않는다
```

- [ ] **Step 2: 실패 확인**
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인** — 이 파일과 전체 시험
- [ ] **Step 5: 커밋** — `feat(voice): 예약 흐름 배선(소리 경로) — Realtime 결과를 예약 흐름으로`

---

### Task 10: Realtime 재생 측정 · 소리 경로 켜기 (설계 7절)

**Files:**
- Modify: `tools/reserve_replay.py`(`--path realtime` 은 `ask_text` 로), 설계서 7절

- [ ] **Step 1: 측정(사용자에게 알리고 실행 — Realtime 약 250번, 주행 꺼졌을 때)**
  - `.venv/bin/python -m tools.reserve_replay --path realtime`
    합격: 첫 요청 **15/15**, 그 밖에서 reserve 호출 **0/13**, 이어진 대답 **10/10**.
  - `.venv/bin/python -m tools.answer_replay --path realtime --tools --labels ~/vica_data/voice_answer_audit_20261113/labeled.csv --rounds 1`
    합격: 208행 잘못된 행동 **15 이하**(11-18 기준 11~13 + 2).
  - 결과를 설계서 7절에 더해 커밋(`docs(voice): 예약 도구 출력 재생 결과 — Realtime`). 기준을 못 넘으면 켜지 않고 보고한다.
- [ ] **Step 2: 켜기와 실기(사용자와 함께)** — 사용자가 `.env` 에 `VICA_TOOLS_AUDIO=on` 을 넣고 소리 모드로 음성 노드를 다시 띄운다. Task 7 Step 7 의 장면을 소리 모드에서 한다. 더 볼 것: Realtime 이 이름을 어떻게 들었는지(`[RT]` 로그의 `heard` 와 ②의 이름), 첫 요청에 ①이 한 번만 나가는지.
- [ ] **Step 3: 마무리** — 전체 시험 통과 확인, 설계서 9절의 1~3단계에 끝난 날짜를 적고 커밋. 4단계(앱·지도 서버·취소)는 다음 계획.

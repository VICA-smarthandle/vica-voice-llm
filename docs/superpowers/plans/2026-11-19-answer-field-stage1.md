# 3-나-2 1단계 — 대답 칸(`answer`) 측정만 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** LLM 세 경로(로컬 midm2-mini · gpt-5.4-mini · Realtime)가 열린 질문을 알고 대답 값 `answer` 를 내게 하고, 208행 재생으로 지금 기준 점수와 견준다. 미션으로 나가는 메시지는 바꾸지 않는다.

**Architecture:** 판단은 새 순수 모듈 `src/answer_field.py` 하나에 둔다(허용 값·`[지금 상황]` 질문 줄·이미 있는 짧은 대답 규칙·값 채우기·로컬 스키마 좁히기). 해석기·Realtime 도구·음성 노드는 그 함수를 부르는 줄만 더한다. 전부 `.env` 스위치 `VICA_ANSWER`(기본 `off`) 뒤에 둔다 — 음성은 운영 브랜치의 소스에서 바로 돌기 때문에, 꺼져 있으면 LLM 에 가는 스키마·지시문·도구 인자가 지금과 같아야 한다. 재생 도구는 `--answer` 로 스위치를 켜고 `answer` 로 채점한다.

**Tech Stack:** Python 3.10 · pydantic v2 · langchain(`ChatOllama`·`ChatOpenAI`) · OpenAI Realtime · pytest

**Spec:** `docs/superpowers/specs/2026-11-18-answer-interpretation-design.md` 4.1·4.2·4.3·5절 1단계 (바탕 `2026-11-14-question-aware-answers-design.md` 4.1 질문 종류 표·D11)

**이 계획에 없는 것(뒤 단계, 따로 계획):** 2단계 = 설계 4.4·4.5(`VicaIntent.msg` 칸, 미션의 `answer` 처리, 음성이 쥔 재개 확인 없애기, 소리 모드에서 규칙 대답을 실제로 쓰기, `_followup_until`·`ASK_FRESH_SEC` 중 내보내는 것을 바꾸는 자리). 3단계 = 4.6 말투. 1단계 측정이 기준보다 나쁘면 2단계로 가지 않는다(설계 5절).

## Global Constraints

- 음성 저장소 `~/VICA-smarthandle/vica-voice-llm` 만 고친다. ROS 저장소 diff 0줄. 브랜치 `after_feedback_waiting`, 작업마다 커밋, **푸시하지 않는다**(사용자 11-18 "푸시 보류").
- 같은 작업 폴더에서 예약 세션이 함께 일한다. 작업을 시작할 때마다 `git status --short` 를 보고, `src/langchain_intent_parser.py`·`src/realtime_intent.py`·`src/ros_node.py`·`tools/answer_replay.py`·`tools/answer_grade.py`·`tests/conftest.py`·`.env.example` 에 내 것이 아닌 미커밋 변경이 있으면 멈추고 사용자에게 알린다. 커밋은 `git commit -- <이 작업의 파일>` 로만 한다.
- 스위치 `VICA_ANSWER`: `on` 일 때만 켜진다. **꺼져 있으면** LLM 에 가는 스키마·지시문·도구 인자와 `/vica/intent`·TTS 로 나가는 것이 지금과 같다. **켜도** 1단계에서는 미션으로 나가는 메시지가 같다 — `src/ros_convert.py` diff 0줄.
- 지시문 본문(`_build_system_prompt`·`build_audio_prompt` 의 f-string)은 고치지 않는다. 열린 질문은 맨 끝 `[지금 상황]` 줄로만 넣는다(설계 2절).
- 새 글자 규칙을 만들지 않는다. 짧은 대답 규칙은 `handle_mode.AFFIRMATIVES`·`NEGATIVES`·`SOFT_AFFIRMATIVES` 와 `parse_wait_minutes` 만 쓴다 — 새 낱말 목록·정규식 금지.
- `answer` 값은 정확히 `yes` · `no` · `destination` · `choice` · `minutes` · `other` · `unclear` · 빈 값. `repeat`·`not_for_robot` 은 쓰지 않는다(설계 P1).
- 새 멘트 없음. rclpy 로거를 조건식으로 고르지 않는다(`tests/test_one_voice_wiring.py::test_no_logger_picked_by_condition`).
- 주행 묶음(`vica-drive.target`)이 켜져 있으면 전체 시험·Ollama 재생을 하지 않고 해당 시험 파일만 `nice -n 19` 로 돌린다. 상태 확인과 실행을 한 명령 줄에 넣지 않는다. 유닛을 직접 멈추거나 재시작하지 않는다.
- 외부 API 측정(gpt-5.4-mini · Realtime)은 실행 전에 사용자 승인을 받는다(Task 6).
- 커밋 메시지는 기존 문체(한국어, `feat(voice): …`), 끝에 `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.
- 시험 명령: `cd ~/VICA-smarthandle/vica-voice-llm && .venv/bin/python -m pytest -q -p no:cacheprovider`. 작업 전 기준 1915 passed(11-19 11:3x, 예약 세션 기록) — Task 1 을 시작할 때 다시 재서 진행 기록에 적는다.

## Review Focus

1. 스위치가 꺼진 채로는 구조화 출력 스키마·`set_intent` 인자·지시문 끝이 지금과 같다(운영 중인 로봇의 동작이 커밋만으로 바뀌면 안 된다). → Task 2 `test_off_changes_nothing`, Task 3 `test_tool_has_no_answer_when_off`
2. LLM 이 허용 밖의 값("banana", 확인 질문에 `choice`)이나 값 없는 `destination`·`minutes` 를 내도 죽지 않고 `unclear` 가 된다. → Task 1 `test_fill_clamps`, Task 2 `test_tool_message_accepts_any_answer_text`
3. "음" 한마디는 확인 질문에서 `yes` 가 되지 않는다(09-28 결정: 혼잣소리로 출발하지 않는다). → Task 1 `test_short_answer_whole_yes_no`
4. "네 화장실로 가자"·"그래 맞아"처럼 통째 짧은 대답이 아닌 말에는 규칙이 끼어들지 않는다(LLM 몫). → Task 1 `test_short_answer_only_whole_utterance`
5. 열린 질문이 없으면 `answer` 는 빈 값이고 로컬 스키마에 칸이 생기지 않는다(대부분의 발화에서 토큰·지연이 늘지 않게). → Task 1 `test_narrowed`, Task 2 `test_no_question_no_answer`
6. 예약 흐름: LLM 이 `reserve_facility` 를 부르면 `ReserveCall` 이 그대로 나온다(대답 채우기가 끼어들지 않는다). → Task 2 `test_reserve_call_passes_through`
7. 그림자 비교 스레드·워밍업이 동시에 해석기를 불러도 질문 종류가 섞이지 않는다. → Task 2 `test_kind_reaches_the_local_schema_and_resets`

---

### Task 1: `answer_field` — 순수 함수와 스위치 (설계 4.1·4.2·4.3-3)

**Files:**
- Create: `src/answer_field.py`
- Modify: `src/schema.py`(`VicaIntent` 끝에 칸 3개), `tests/conftest.py`(`_DEFAULTS` 에 `"VICA_ANSWER": "off"`), `.env.example`(`VICA_TOOLS_*` 묶음 아래)
- Test: `tests/test_answer_field.py`

**Interfaces:**
- Consumes: `src.question_record.OpenQuestion`(`id`·`kind`·`text`), `src.handle_mode`(`AFFIRMATIVES`·`NEGATIVES`·`SOFT_AFFIRMATIVES`·`normalize_short_reply`), `src.langchain_intent_parser.parse_wait_minutes`(함수 안에서 지연 import — 순환 import 방지)
- Produces:
  - `ANSWERS = ("yes", "no", "destination", "choice", "minutes", "other", "unclear")`
  - `ANSWER_DESCRIPTION: str` — 스키마 칸 설명(아래 Step 3)
  - `on() -> bool` — 환경 변수 `VICA_ANSWER` 가 `on` 일 때만 참(매번 읽는다)
  - `allowed(kind: str) -> tuple[str, ...]`
  - `question_line(question: Optional[OpenQuestion]) -> str`
  - `with_question(block: str, question: Optional[OpenQuestion]) -> str`
  - `short_answer(text: str, question: Optional[OpenQuestion], local: bool = False) -> Optional[tuple[str, str]]`
  - `fill(intent: VicaIntent, question: Optional[OpenQuestion], ruled: Optional[tuple[str, str]] = None, llm_answer: str = "") -> VicaIntent`
  - `narrowed(base: type[BaseModel], kind: str) -> type[BaseModel]`
  - `log_line(intent: VicaIntent, question: Optional[OpenQuestion]) -> str`
  - `VicaIntent.question_id: str = ""` · `VicaIntent.answer: str = ""` · `VicaIntent.answer_value: str = ""`

- [ ] **Step 1: 실패하는 시험 작성** — `tests/test_answer_field.py`

```python
"""대답 칸(3-나-2 1단계, 설계 2026-11-18 4.1~4.3)."""
import pytest
from pydantic import BaseModel

from src import answer_field as af
from src.question_record import OpenQuestion
from src.schema import VicaIntent

BASE = ("no", "destination", "other", "unclear")


def q(kind="confirm", text="사회복지로 안내해드릴까요?"):
    return OpenQuestion(id="q1", kind=kind, text=text, option_ids=(), opened_at=0.0)


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("VICA_ANSWER", "on")


def test_switch_defaults_to_off(monkeypatch):
    assert af.on() is False
    monkeypatch.setenv("VICA_ANSWER", "ON")
    assert af.on() is True
    monkeypatch.setenv("VICA_ANSWER", "1")      # on 이라는 글자만 켠다(VICA_TOOLS_* 와 같은 규칙)
    assert af.on() is False


def test_allowed_by_kind():
    for kind in ("open", "confirm", "yesno", "yesno_inverted"):
        assert set(af.allowed(kind)) == {"yes", *BASE}
    assert set(af.allowed("choice")) == {"choice", *BASE}
    assert set(af.allowed("minutes")) == {"minutes", *BASE}
    assert af.allowed("") == af.ANSWERS and af.allowed("새종류") == af.ANSWERS


def test_question_line(on):
    assert af.question_line(q()) == ('- 로봇이 물은 질문: "사회복지로 안내해드릴까요?" '
                                     "(확인 질문 — answer 는 yes·no·destination·other·unclear 중 하나)")
    inverted = af.question_line(q("yesno_inverted", "안내가 필요 없으신가요?"))
    assert "필요 없다는 뜻이면 yes, 필요하다는 뜻이면 no" in inverted
    assert af.question_line(None) == ""


def test_with_question(on):
    line = af.question_line(q())
    assert af.with_question("", q()) == "\n[지금 상황]\n" + line + "\n"
    assert af.with_question("\n[지금 상황] (미션…)\n- 건물/층: 모름\n", q()) == (
        "\n[지금 상황] (미션…)\n- 건물/층: 모름\n" + line + "\n")
    assert af.with_question("블록", None) == "블록"


def test_off_is_inert():
    intent = VicaIntent(intent="affirm")
    assert af.with_question("블록", q()) == "블록"
    assert af.short_answer("네", q("yesno")) is None
    assert af.fill(intent, q(), ("yes", ""), "yes") is intent


@pytest.mark.parametrize("text,kind,out", [
    ("네.", "confirm", ("yes", "")), ("네. 네.", "yesno", ("yes", "")), ("아니요", "yesno", ("no", "")),
    ("그래.", "yesno_inverted", ("yes", "")),      # 들은 대로 — 뜻(필요 없음)은 미션이 질문 종류로 읽는다
    ("음", "confirm", ("unclear", "")),            # 혼잣소리는 확인 질문의 승낙이 아니다(09-28)
    ("음", "yesno", ("yes", "")),
])
def test_short_answer_whole_yes_no(on, text, kind, out):
    assert af.short_answer(text, q(kind)) == out


@pytest.mark.parametrize("text,kind", [
    ("네 화장실로 가자", "confirm"), ("그래 맞아.", "yesno"), ("부요.", "yesno"),
    ("네", "open"), ("네", "choice"), ("네", "minutes"), ("삼십분.", "confirm"),
])
def test_short_answer_only_whole_utterance(on, text, kind):
    assert af.short_answer(text, q(kind)) is None


def test_short_answer_minutes(on):
    m = q("minutes", "몇 분쯤 걸리실까요?")
    assert af.short_answer("삼십분.", m) == ("minutes", "30")
    assert af.short_answer("한 십 분 정도.", m) == ("minutes", "10")
    assert af.short_answer("한 시간.", m) == ("minutes", "60")
    assert af.short_answer("십 분에서 십오 분", m, local=True) == ("minutes", "15")
    assert af.short_answer("십 분에서 십오 분", m) == ("minutes", "23")   # 로컬 규칙 밖은 큰 쪽 × 1.5(지금 _finalize 와 같다)
    assert af.short_answer("삼십분.", None) is None


def test_fill_rule_first_then_llm(on):
    nav = VicaIntent(intent="navigate", matched_destination_id="info", need_confirm=True)
    ruled = af.fill(nav, q("yesno"), ("yes", ""), "destination")
    assert (ruled.question_id, ruled.answer, ruled.answer_value) == ("q1", "yes", "")
    assert (ruled.intent, ruled.matched_destination_id, ruled.need_confirm) == ("navigate", "info", True)
    by_llm = af.fill(nav, q(), None, "destination")
    assert (by_llm.answer, by_llm.answer_value) == ("destination", "info")
    waited = af.fill(VicaIntent(intent="wait", wait_minutes=10), q("minutes"), None, "minutes")
    assert (waited.answer, waited.answer_value) == ("minutes", "10")
    empty = af.fill(VicaIntent(intent="pause"), q(), None, "")
    assert (empty.question_id, empty.answer) == ("q1", "")


def test_fill_clamps(on):
    nav = VicaIntent(intent="navigate", matched_destination_id="", need_confirm=True)
    assert af.fill(nav, q(), None, "destination").answer == "unclear"          # 목적지 id 없음
    assert af.fill(VicaIntent(intent="wait"), q("minutes"), None, "minutes").answer == "unclear"   # 분 없음(-1)
    assert af.fill(nav, q(), None, "choice").answer == "unclear"               # 확인 질문에 허용 밖
    assert af.fill(nav, q(), None, "banana").answer == "unclear"
    urgent = VicaIntent(intent="unknown", safety_flag="emergency")
    assert af.fill(urgent, q(), ("yes", ""), "yes") is urgent
    assert af.fill(nav, None, None, "yes") is nav


def test_narrowed():
    class Draft(BaseModel):
        intent: str = ""

    assert af.narrowed(Draft, "") is Draft
    model = af.narrowed(Draft, "minutes")
    schema = model.model_json_schema()
    assert set(schema["properties"]["answer"]["enum"]) == {"minutes", *BASE}
    assert "answer" in schema["required"]
    assert af.narrowed(Draft, "minutes") is model and af.narrowed(Draft, "confirm") is not model


def test_log_line(on):
    filled = af.fill(VicaIntent(intent="affirm"), q("yesno"), ("yes", ""))
    assert af.log_line(filled, q("yesno")) == "[대답] answer=yes value=- 질문=q1 kind=yesno"
    assert af.log_line(VicaIntent(intent="affirm"), None) == "[대답] 없음"
```

- [ ] **Step 2: 시험이 실패하는지 확인**

Run: `nice -n 19 .venv/bin/python -m pytest tests/test_answer_field.py -q -p no:cacheprovider`
Expected: 수집 단계 ImportError(`src.answer_field` 없음). 빈 함수·상수만 먼저 넣어 단정 실패(AssertionError)까지 확인한다.

- [ ] **Step 3: `src/answer_field.py` 구현, `VicaIntent` 칸 3개 추가**

정하는 것만 적는다.

- `allowed(kind)`: 공통 `("no", "destination", "other", "unclear")` 앞에 종류별 한 값 — `open`·`confirm`·`yesno`·`yesno_inverted` 는 `yes`, `choice` 는 `choice`, `minutes` 는 `minutes`. 모르는 종류·빈 값은 `ANSWERS` 전부.
- `question_line`: `- 로봇이 물은 질문: "{text}" ({이름} — answer 는 {허용 값을 ·로 이은 것} 중 하나)`. 이름: `open` 열린 질문 · `confirm` 확인 질문 · `choice` 고르기 질문 · `yesno` 네·아니요 질문 · `minutes` 시간 질문 · `yesno_inverted` 부정 질문 · 모르는 종류는 `질문`. `yesno_inverted` 만 닫는 괄호 앞에 `. 필요 없다는 뜻이면 yes, 필요하다는 뜻이면 no` 를 덧붙인다(지금 지시문의 affirm·deny 기준과 같은 방향).
- `with_question`: 꺼졌거나 질문이 없으면 `block` 그대로. `block` 이 비면 `"\n[지금 상황]\n" + 줄 + "\n"`, 아니면 `block.rstrip("\n") + "\n" + 줄 + "\n"`.
- `short_answer`: 꺼졌거나 질문이 없으면 `None`. `confirm`·`yesno`·`yesno_inverted` 에서 `normalize_short_reply(text)` 가 `NEGATIVES` → `("no", "")`, `AFFIRMATIVES` → `("yes", "")`, `SOFT_AFFIRMATIVES` → `confirm` 이면 `("unclear", "")` 아니면 `("yes", "")`. `minutes` 에서 `parse_wait_minutes(text, range_factor=1.0 if local else 1.5)` 가 값을 주면 `("minutes", str(값))`. 그 밖은 `None`.
- `fill`: 꺼짐·질문 없음·`safety_flag == "emergency"` 면 받은 `intent` 를 그대로(같은 객체). 아니면 `question_id=question.id` 와 함께 — `ruled` 가 있으면 그 값, 없으면 `llm_answer.strip()`: 빈 값은 빈 값, `allowed(question.kind)` 밖은 `unclear`, `destination`·`choice` 는 값 = `intent.matched_destination_id`(비면 `unclear`), `minutes` 는 값 = `str(intent.wait_minutes)`(`<= 0` 이면 `unclear`). `model_copy(update=…)` 로 돌려준다.
- `narrowed`: `kind` 가 비면 `base`. 아니면 `pydantic.create_model(f"{base.__name__}_{kind}", __base__=base, answer=(Literal[allowed(kind)], Field(description=ANSWER_DESCRIPTION)))` — 기본값 없음(구조화 출력이 값을 강제한다). `(base, kind)` 로 캐시한다.
- `log_line`: `answer` 가 비면 `[대답] 없음`, 아니면 `[대답] answer={answer} value={answer_value or '-'} 질문={question_id} kind={question.kind}`.
- `ANSWER_DESCRIPTION`(그대로 쓴다):

```python
ANSWER_DESCRIPTION = (
    "[지금 상황]에 '로봇이 물은 질문' 줄이 있을 때만 채운다 — 그 질문에 대한 이 말의 대답. "
    "yes(승낙) · no(거절) · destination(다른 곳으로 가자 — destination_candidate 에 그곳) · "
    "choice(고르기 질문의 보기 중 하나 — destination_candidate 에 그곳) · minutes(시간 — wait_minutes 에 분) · "
    "other(질문과 상관없는 명령이나 질문) · unclear(못 알아들음). 그 줄에 적힌 값만 쓴다. 줄이 없으면 빈 문자열.")
```

- `.env.example` 에 넣을 줄(`VICA_RESERVATION_DIR` 줄 아래):

```
# ---- 대답 칸 (2026-11-18 설계 answer-interpretation, 1단계 측정) ----
# on 이면 LLM 이 열린 질문의 종류를 알고 대답 값(answer)을 함께 낸다. 1단계에서는 로그·재생 채점에만 쓴다.
# VICA_ANSWER=off            # off | on — 바꾼 뒤 음성 노드(LLM)를 재시작한다
```

- [ ] **Step 4: 시험 통과 확인**

Run: `nice -n 19 .venv/bin/python -m pytest tests/test_answer_field.py tests/test_one_voice.py tests/test_one_voice_convert.py -q -p no:cacheprovider`
Expected: 전부 통과(`VicaIntent` 칸 추가가 한 목소리·메시지 변환 시험을 깨지 않는다).

- [ ] **Step 5: 커밋**

```bash
git add src/answer_field.py tests/test_answer_field.py
git commit -m "feat(voice): 대답 칸 순수 함수 — 허용 값·질문 줄·짧은 대답 규칙·값 채우기, 스위치 VICA_ANSWER(기본 꺼짐)" -- src/answer_field.py src/schema.py tests/test_answer_field.py tests/conftest.py .env.example
```

---

### Task 2: 글자 경로 — `answer` 칸·질문 줄·규칙 먼저 (설계 4.1·4.2·4.3)

**Files:**
- Modify: `src/langchain_intent_parser.py` — `_IntentDraft` 아래(새 클래스), `_get_structured_llm`·`_build_local_structured`·`_set_intent_tool`·`_draft_from_tool_message`·`_build_system_prompt`·`parse_intent`
- Test: `tests/test_parser_answer.py`

**Interfaces:**
- Consumes: Task 1 의 `answer_field`(`on`·`ANSWERS`·`ANSWER_DESCRIPTION`·`with_question`·`short_answer`·`fill`·`narrowed`)
- Produces:
  - `class _AnswerDraft(_IntentDraft)` — `answer: str = Field(default="", description=ANSWER_DESCRIPTION, json_schema_extra={"enum": ["", *ANSWERS]})`. 형은 `str` 이다 — 외부 API 가 허용 밖의 값을 내도 검증에서 죽지 않고 `fill` 이 `unclear` 로 바꾼다.
  - `_draft_cls() -> type[_IntentDraft]` — `_AnswerDraft if answer_field.on() else _IntentDraft`
  - `_ANSWER_KIND: ContextVar[str]`(기본 `""`) — 이번 호출의 열린 질문 종류
  - `_ollama_structured(llm)` — 로컬 구조화 출력 runnable
  - `_build_system_prompt(destinations, robot_state=None, local=False, situation="", question=None) -> str`
  - `parse_intent(user_text, destinations, history=None, robot_state=None, model=None, situation="", question: Optional[OpenQuestion] = None) -> "VicaIntent | ReserveCall"`

- [ ] **Step 1: 실패하는 시험 작성** — `tests/test_parser_answer.py`

```python
"""글자 경로의 대답 칸(3-나-2 1단계)."""
import pytest
from langchain_core.messages import AIMessage

from src import answer_field as af
from src import langchain_intent_parser as parser
from src.question_record import OpenQuestion
from src.reservation_flow import ReserveCall
from src.schema import DestinationData, RobotState

DESTS = [DestinationData(id="welfare", name="사회복지", confirm_prompt="사회복지로 안내해드릴까요?"),
         DestinationData(id="info", name="안내소", confirm_prompt="안내소로 안내해드릴까요?")]


def q(kind="confirm", text="사회복지로 안내해드릴까요?"):
    return OpenQuestion(id="q1", kind=kind, text=text, option_ids=("welfare",), opened_at=0.0)


class Manager:
    """get_backend_manager 대역 — 호출 때의 질문 종류를 적어 둔다."""
    has_local = False

    def __init__(self, draft=None):
        self.draft, self.kinds = draft, []

    def invoke(self, messages):
        self.kinds.append(parser._ANSWER_KIND.get())
        if self.draft is None:
            raise AssertionError("LLM 을 부르면 안 되는 자리")
        return self.draft


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("VICA_ANSWER", "on")


def use(monkeypatch, manager):
    monkeypatch.setattr(parser, "get_backend_manager", lambda: manager)
    return manager


def test_off_changes_nothing(monkeypatch):
    assert parser._draft_cls() is parser._IntentDraft
    assert "answer" not in parser._set_intent_tool()["function"]["parameters"]["properties"]
    assert parser._build_system_prompt(DESTS, question=q()) == parser._build_system_prompt(DESTS)
    use(monkeypatch, Manager(parser._IntentDraft(intent="navigate", destination_candidate="안내소")))
    out = parser.parse_intent("안내소로 가자", DESTS, history=[AIMessage(q().text)], question=q())
    assert (out.answer, out.question_id) == ("", "")


def test_schema_and_prompt_when_on(on):
    assert parser._draft_cls() is parser._AnswerDraft
    prop = parser._set_intent_tool()["function"]["parameters"]["properties"]["answer"]
    assert prop["enum"] == ["", *af.ANSWERS] and prop["description"] == af.ANSWER_DESCRIPTION
    assert parser._build_system_prompt(DESTS, question=q()).endswith(af.with_question("", q()))
    assert parser._build_system_prompt(DESTS) == parser._build_system_prompt(DESTS, question=None)


def test_llm_answer_is_filled_and_the_intent_stays(on, monkeypatch):
    history = [AIMessage(q().text)]
    use(monkeypatch, Manager(parser._AnswerDraft(intent="navigate", destination_candidate="안내소", answer="destination")))
    # "아니 안내소로 가자"는 쓰지 않는다 — 글자 경로는 부정 낱말이 섞이면 LLM 없이 deny 로 끝낸다(지금 동작, 2단계에서 다룬다).
    out = parser.parse_intent("안내소로 가자", DESTS, history=history, question=q())
    use(monkeypatch, Manager(parser._IntentDraft(intent="navigate", destination_candidate="안내소")))
    old = parser.parse_intent("안내소로 가자", DESTS, history=history)
    assert (out.question_id, out.answer, out.answer_value) == ("q1", "destination", "info")
    assert (out.intent, out.matched_destination_id, out.need_confirm, out.reply) == (
        old.intent, old.matched_destination_id, old.need_confirm, old.reply)


def test_rule_answers_without_the_llm(on, monkeypatch):
    use(monkeypatch, Manager())
    out = parser.parse_intent("네.", DESTS, history=[AIMessage("여기서 대기할까요?")],
                              question=q("yesno", "여기서 대기할까요?"))
    assert (out.intent, out.answer) == ("affirm", "yes")


def test_rule_beats_the_llm_answer(on, monkeypatch):
    monkeypatch.setenv("VICA_LOCAL_RULES", "1")
    monkeypatch.setenv("VICA_INTENT_INPUT", "text")
    manager = use(monkeypatch, Manager(parser._AnswerDraft(intent="unknown", reply="네?", answer="other")))
    out = parser.parse_intent("네", DESTS, history=[], robot_state=RobotState(dialog_state="idle"),
                              question=q("yesno", "여기서 대기할까요?"))
    assert len(manager.kinds) == 1          # 로컬 규칙은 대답을 기다리는 단계가 아니면 LLM 에 묻는다 — 의도는 그 길 그대로
    assert (out.intent, out.answer) == ("unknown", "yes")


def test_no_question_no_answer(on, monkeypatch):
    manager = use(monkeypatch, Manager(parser._AnswerDraft(intent="navigate", destination_candidate="안내소", answer="yes")))
    out = parser.parse_intent("안내소로 가자", DESTS)
    assert (out.answer, out.question_id) == ("", "") and manager.kinds == [""]


def test_reserve_call_passes_through(on, monkeypatch):
    use(monkeypatch, Manager(ReserveCall(facility="세미나실")))
    assert parser.parse_intent("세미나실 예약해 줘", DESTS, question=q("open", "네?")) == ReserveCall(facility="세미나실")


def test_tool_message_accepts_any_answer_text(on):
    msg = AIMessage(content="", tool_calls=[{"name": "set_intent", "id": "c0",
                                             "args": {"intent": "unknown", "answer": "banana"}}])
    assert parser._draft_from_tool_message(msg).answer == "banana"


class FakeOllama:
    def __init__(self):
        self.schemas = []

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        return type("R", (), {"invoke": staticmethod(lambda messages: schema)})()


def test_kind_reaches_the_local_schema_and_resets(on, monkeypatch):
    llm = FakeOllama()
    runnable = parser._ollama_structured(llm)
    assert runnable.invoke([]) is parser._IntentDraft                 # 열린 질문 없음 — 칸 없는 지금 스키마
    token = parser._ANSWER_KIND.set("minutes")
    try:
        picked = runnable.invoke([])
    finally:
        parser._ANSWER_KIND.reset(token)
    assert picked is af.narrowed(parser._IntentDraft, "minutes")
    manager = use(monkeypatch, Manager(parser._AnswerDraft(intent="unknown", answer="unclear")))
    parser.parse_intent("뭐라고", DESTS, question=q("minutes", "몇 분쯤 걸리실까요?"))
    assert manager.kinds == ["minutes"] and parser._ANSWER_KIND.get() == ""


def test_local_schema_is_the_old_one_when_off():
    llm = FakeOllama()
    assert parser._ollama_structured(llm) is not None and llm.schemas == [parser._IntentDraft]
```

- [ ] **Step 2: 시험이 실패하는지 확인**

Run: `nice -n 19 .venv/bin/python -m pytest tests/test_parser_answer.py -q -p no:cacheprovider`
Expected: FAIL — `_draft_cls`·`_AnswerDraft`·`_ANSWER_KIND`·`_ollama_structured` 없음, `parse_intent` 에 `question` 인자 없음.

- [ ] **Step 3: 해석기 수정**

- `_AnswerDraft`·`_draft_cls`·`_ANSWER_KIND` 를 `_IntentDraft` 바로 아래에 둔다.
- `_ollama_structured(llm)`: 꺼져 있으면 `llm.with_structured_output(_IntentDraft)` 를 그대로 돌려준다(지금과 같은 객체 꼴). 켜져 있으면 호출 때 `_ANSWER_KIND.get()` 으로 `answer_field.narrowed(_IntentDraft, kind)` 스키마의 구조화 LLM 을 골라 부르는 `RunnableLambda` — 종류마다 한 번 만들어 dict 에 둔다. `_get_structured_llm` 의 ollama 갈래와 `_build_local_structured` 의 마지막 줄을 이 함수로 바꾼다.
- `_get_structured_llm` 의 openai 갈래·`_set_intent_tool`·`_draft_from_tool_message` 의 `_IntentDraft` 를 `_draft_cls()` 로 바꾼다.
- `_build_system_prompt`: 끝의 `+ ledger` 를 `+ answer_field.with_question(ledger, question)` 로 바꾼다. f-string 본문은 그대로.
- `parse_intent`: 맨 앞에서 `ruled = answer_field.short_answer(user_text, question, local)`. `VicaIntent` 를 돌려주는 네 자리(지름길 · `_mission_question_answer` · LLM 실패 · `_finalize`)를 `answer_field.fill(결과, question, ruled, getattr(draft, "answer", ""))` 로 감싼다(LLM 을 안 거친 자리는 `llm_answer=""`). LLM 호출은 `_ANSWER_KIND.set(question.kind if question is not None and answer_field.on() else "")` 과 `finally: reset` 사이에 둔다. `ReserveCall` 은 지금처럼 그대로 돌려준다. `_build_system_prompt` 에 `question=question` 을 넘긴다.

- [ ] **Step 4: 시험 통과 확인**

Run: `nice -n 19 .venv/bin/python -m pytest tests/test_parser_answer.py tests/test_parser_tools.py tests/test_parser_backend.py tests/test_command_intents.py -q -p no:cacheprovider`
Expected: 전부 통과.

- [ ] **Step 5: 커밋**

```bash
git add tests/test_parser_answer.py
git commit -m "feat(voice): 글자 경로 대답 칸 — _AnswerDraft·[지금 상황] 질문 줄·짧은 대답 규칙 먼저, 로컬은 질문 종류별 스키마" -- src/langchain_intent_parser.py tests/test_parser_answer.py
```

---

### Task 3: 소리 경로 — Realtime 도구 인자와 질문 줄 (설계 4.1·4.2)

**Files:**
- Modify: `src/realtime_intent.py`(`build_intent_tool`), `src/langchain_intent_parser.py`(`parse_intent_audio`·`parse_intent_realtime_text`·`_deliver_realtime_result`)
- Test: `tests/test_realtime_intent.py`(끝에 추가), `tests/test_parser_audio.py`(끝에 추가)

**Interfaces:**
- Consumes: Task 1 `answer_field`, Task 2 `_draft_cls`
- Produces:
  - `build_intent_tool(description=…)` — 켜져 있으면 `properties["answer"] = {"type": "string", "enum": ["", *ANSWERS], "description": ANSWER_DESCRIPTION}`. `required` 는 그대로 `["intent", "heard_text"]`.
  - `parse_intent_audio(pcm16_16k, destinations, history=None, robot_state=None, situation="", directory_block="", question=None)` · `parse_intent_realtime_text(text, …, question=None)` — 반환 꼴은 그대로, `VicaIntent` 에 `answer` 가 채워진다.
  - `_deliver_realtime_result(result, destinations, robot_state, question=None)`

- [ ] **Step 1: 실패하는 시험 작성**

`tests/test_realtime_intent.py` 끝:

```python
def test_tool_has_no_answer_when_off():
    from src.realtime_intent import build_intent_tool
    assert "answer" not in build_intent_tool()["parameters"]["properties"]


def test_tool_has_the_answer_field_when_on(monkeypatch):
    from src import answer_field as af
    from src.realtime_intent import build_intent_tool
    monkeypatch.setenv("VICA_ANSWER", "on")
    tool = build_intent_tool()
    assert tool["parameters"]["properties"]["answer"] == {
        "type": "string", "enum": ["", *af.ANSWERS], "description": af.ANSWER_DESCRIPTION}
    assert tool["parameters"]["required"] == ["intent", "heard_text"]
```

`tests/test_parser_audio.py` 끝:

```python
def _minutes_q():
    from src.question_record import OpenQuestion
    return OpenQuestion(id="q7", kind="minutes", text="몇 분쯤 걸리실까요?", option_ids=(), opened_at=0.0)


def test_realtime_answer_is_filled(monkeypatch):
    from src.realtime_intent import RealtimeResult
    monkeypatch.setenv("VICA_ANSWER", "on")
    result = RealtimeResult(draft={"intent": "wait", "wait_minutes": 15, "answer": "minutes"},
                            heard_text="삼십분", latency_sec=1.0)
    intent, heard, _dt, _info = parser._deliver_realtime_result(result, [], None, question=_minutes_q())
    # Realtime 이 낸 값 그대로다 — 받아쓴 글자의 규칙(30분)과 견주는 일은 노드 로그·재생 도구가 한다.
    assert (intent.question_id, intent.answer, intent.answer_value, heard) == ("q7", "minutes", "15", "삼십분")
    clamped = RealtimeResult(draft={"intent": "unknown", "answer": "yes"}, heard_text="네", latency_sec=1.0)
    assert parser._deliver_realtime_result(clamped, [], None, question=_minutes_q())[0].answer == "unclear"


def test_realtime_answer_is_empty_when_off():
    from src.realtime_intent import RealtimeResult
    result = RealtimeResult(draft={"intent": "wait", "wait_minutes": 15, "answer": "minutes"},
                            heard_text="삼십분", latency_sec=1.0)
    assert parser._deliver_realtime_result(result, [], None, question=_minutes_q())[0].answer == ""


def test_audio_prompt_carries_the_question_line(monkeypatch):
    from src import answer_field as af
    from src.realtime_intent import RealtimeResult
    seen = {}

    class Client:
        def ask(self, pcm, history, instructions):
            seen["instructions"] = instructions
            return RealtimeResult(draft={"intent": "unknown"}, heard_text="", latency_sec=0.1)

    monkeypatch.setattr(parser, "get_realtime_client", lambda: Client())
    parser.parse_intent_audio(b"", [], question=_minutes_q())
    assert "로봇이 물은 질문" not in seen["instructions"]
    monkeypatch.setenv("VICA_ANSWER", "on")
    parser.parse_intent_audio(b"", [], question=_minutes_q())
    assert seen["instructions"].endswith(af.with_question("", _minutes_q()))
```

(`tests/test_parser_audio.py` 는 이미 `from src import langchain_intent_parser as parser` 로 부른다.)

- [ ] **Step 2: 시험이 실패하는지 확인**

Run: `nice -n 19 .venv/bin/python -m pytest tests/test_realtime_intent.py tests/test_parser_audio.py -q -p no:cacheprovider`
Expected: 새 시험 5개 FAIL(`question` 인자 없음, `answer` 칸 없음), 나머지 통과.

- [ ] **Step 3: 구현**

- `build_intent_tool`: 지금 dict 를 만든 뒤 `answer_field.on()` 이면 `answer` 칸만 더한다.
- `parse_intent_audio`·`parse_intent_realtime_text`: `build_audio_prompt(destinations, robot_state, answer_field.with_question(situation, question), directory_block)`, `_deliver_realtime_result(result, destinations, robot_state, question)`.
- `_deliver_realtime_result`: `_IntentDraft(**result.draft)` 를 `_draft_cls()(**result.draft)` 로 바꾸고, 마지막에 `intent = answer_field.fill(intent, question, None, getattr(draft, "answer", ""))` — `_audio_short_answer` 가 의도를 고친 뒤에 둔다. `ReserveCall` 갈래는 그대로.

- [ ] **Step 4: 시험 통과 확인**

Run: `nice -n 19 .venv/bin/python -m pytest tests/test_realtime_intent.py tests/test_parser_audio.py tests/test_realtime_conn_local_1024.py tests/test_reserve_replay.py -q -p no:cacheprovider`
Expected: 전부 통과.

- [ ] **Step 5: 커밋**

```bash
git commit -m "feat(voice): 소리 경로 대답 칸 — Realtime set_intent 인자 answer·[지금 상황] 질문 줄(스위치 꺼짐이면 그대로)" -- src/realtime_intent.py src/langchain_intent_parser.py tests/test_realtime_intent.py tests/test_parser_audio.py
```

---

### Task 4: 음성 노드 배선 — 질문 넘기기와 기록만 (설계 4.2, 5절 "미션으로 나가는 것은 지금과 같다")

**Files:**
- Modify: `src/ros_node.py` — import, `_on_user_text`, `_on_user_audio`, `_publish_intent`, `_situation_block`, 새 메서드 `_log_transcript_rule`
- Test: `tests/test_answer_wiring.py`

**Interfaces:**
- Consumes: Task 1 `answer_field`(`on`·`short_answer`·`log_line`), Task 2·3 의 `question=` 인자
- Produces: 로그 두 줄 — 발화마다 `[대답] …`(`answer_field.log_line`), 소리 모드에서 받아쓴 글자가 규칙에 맞을 때 `[대답] 받아쓴 글자 규칙={answer}/{value or '-'} · Realtime={answer or '-'}/{value or '-'} · kind={kind} · 글자='{text}'`. 2단계가 이 줄로 규칙 대답과 Realtime 대답이 실제로 얼마나 갈리는지 본다.

- [ ] **Step 1: 실패하는 시험 작성** — `tests/test_answer_wiring.py`

```python
"""대답 칸 배선(3-나-2 1단계) — ros_node 소스 구조로 확인한다. 1단계는 기록만 한다."""
from __future__ import annotations

import ast
from pathlib import Path

from tests.test_one_voice_wiring import _calls, _method

SRC = Path(__file__).resolve().parents[1] / "src"


def _kw(call: ast.Call) -> dict:
    return {k.arg: ast.unparse(k.value) for k in call.keywords}


def test_text_path_passes_the_open_question():
    (call,) = _calls(_method("_on_user_text"), "parse_intent")
    assert _kw(call).get("question") == "self._question"


def test_audio_path_passes_the_question_it_started_with():
    fn = _method("_on_user_audio")
    (call,) = _calls(fn, "parse_intent_audio")
    assert _kw(call).get("question") == "question"
    snapshot = [u for u in _calls(fn, "self._audio_turn.update") if "question" in _kw(u)]
    assert snapshot and snapshot[0].lineno < call.lineno      # 그림자 갈래가 같은 질문으로 규칙을 본다


def test_time_based_awaiting_line_is_replaced_only_when_on():
    assert "answer_field.on()" in ast.unparse(_method("_situation_block"))


def test_answer_is_logged_after_publishing():
    fn = _method("_publish_intent")
    (log,) = _calls(fn, "answer_field.log_line")
    assert log.lineno > max(c.lineno for c in _calls(fn, "self._intent_pub.publish"))


def test_transcript_rule_is_only_logged():
    fn = _method("_log_transcript_rule")
    assert _calls(fn, "answer_field.short_answer")
    assert not _calls(fn, "self._intent_pub.publish") and not _calls(fn, "self._publish_intent")
    assert _calls(_method("_on_user_text"), "self._log_transcript_rule")


def test_nothing_new_goes_to_the_mission_yet():
    assert "answer" not in (SRC / "ros_convert.py").read_text()
    for name in ("mission_question.py", "one_voice.py"):
        assert "answer_field" not in (SRC / name).read_text()
```

- [ ] **Step 2: 시험이 실패하는지 확인**

Run: `nice -n 19 .venv/bin/python -m pytest tests/test_answer_wiring.py -q -p no:cacheprovider`
Expected: 마지막 시험만 통과, 나머지 FAIL.

- [ ] **Step 3: 배선**

- import: `from . import answer_field, local_rules, question_record`.
- `_on_user_text`: `parse_intent(…, question=self._question)`. 소리 경로가 이미 처리한 갈래(`audio_turn_applies` → `_shadow_text`)에서 `self._shadow_text(text, turn)` 앞에 `self._log_transcript_rule(text, turn)`.
- `_on_user_audio`: `self._audio_turn.update(history=…, robot_state=…)` 에 `question=question` 을 더하고(`question = self._question` 을 그 앞에서 잡는다) `parse_intent_audio(…, question=question)`.
- `_log_transcript_rule(self, text: str, turn: dict) -> None`: `ruled = answer_field.short_answer(text, turn.get("question"))` 가 `None` 이면 끝. 아니면 위 Produces 의 글자로 `info` 한 줄. 내보내는 것은 없다.
- `_publish_intent`: `question_record.log_line` 줄 바로 아래에 `if answer_field.on(): self.get_logger().info(answer_field.log_line(intent, question))`.
- `_situation_block`: `awaiting = time.time() < self._followup_until and not answer_field.on()` — 켜져 있으면 "방금 질문하고 답을 기다리는 중" 줄 대신 질문 줄(해석기가 붙인다)이 나간다. `is_relisten_noise`·`_apply_misheard_rule`·`mission_question.ASK_FRESH_SEC` 는 내보내는 것을 바꾸므로 건드리지 않는다(2단계).

- [ ] **Step 4: 시험 통과 확인**

Run: `nice -n 19 .venv/bin/python -m pytest tests/test_answer_wiring.py tests/test_one_voice_wiring.py tests/test_wake_echo_wiring.py tests/test_reservation_wiring.py -q -p no:cacheprovider`
Expected: 전부 통과.

- [ ] **Step 5: 커밋**

```bash
git add tests/test_answer_wiring.py
git commit -m "feat(voice): 대답 칸 노드 배선 — 열린 질문을 해석기에 넘기고 [대답] 로그만(미션으로 나가는 것은 그대로)" -- src/ros_node.py tests/test_answer_wiring.py
```

---

### Task 5: 재생 도구 — `answer` 로 채점 (설계 5절)

**Files:**
- Modify: `tools/answer_replay.py`, `tools/answer_grade.py`
- Test: `tests/test_answer_replay.py`(끝에 추가), `tests/test_answer_grade.py`(끝에 추가)

**Interfaces:**
- Consumes: Task 1~3
- Produces:
  - `tools.answer_grade.pred_of(intent, kind: str, question: str, destinations) -> tuple[str, str]` — `intent.answer` 가 있으면 `(intent.answer, intent.answer_value)`, 없으면 `answer_of(…)`(규칙 지름길처럼 `answer` 를 안 채운 길은 기준 점수와 같은 어댑터로 읽는다)
  - `tools.answer_replay.answer_env(on: bool) -> dict` — `{"VICA_ANSWER": "on" | "off"}`
  - `tools.answer_replay.question_for(row: dict, destinations) -> OpenQuestion` — `id="replay"`, `kind` 는 행의 값(`other` 는 `open` — LLM 이 지은 질문은 운영에서 `open` 으로 넘어간다), `option_ids` 는 `confirm` 이면 `question_destination_id` 한 개, `choice` 면 `src.one_voice._choice_option_ids(질문, destinations)`
  - `tools.answer_replay.with_transcript_rule(intent, row: dict, question) -> intent` — 소리 경로 행에 D11 을 적용: `answer_field.short_answer(text_for(row), question)` 가 값을 주면 `answer`·`answer_value` 를 그 값으로 바꾼 사본, 아니면(또는 `ReserveCall` 이면) 그대로
  - CLI `--answer`(스위치를 켠 채로 잰다) — 결과 파일 `replay_<path>_answer_<YYYYMMDD-HHMM>.json·.md`

- [ ] **Step 1: 실패하는 시험 작성**

`tests/test_answer_grade.py` 끝:

```python
def test_pred_of_prefers_the_answer_field(dests):
    from tools.answer_grade import pred_of
    said = VicaIntent(intent="navigate", matched_destination_id="d1", answer="yes")
    assert pred_of(said, "yesno", "여기서 대기할까요?", dests) == ("yes", "")
    plain = VicaIntent(intent="navigate", matched_destination_id="d1")
    assert pred_of(plain, "yesno", "여기서 대기할까요?", dests) == answer_of(plain, "yesno", "여기서 대기할까요?", dests)
```

`tests/test_answer_replay.py` 끝:

```python
def test_answer_switch_env():
    assert answer_replay.answer_env(True) == {"VICA_ANSWER": "on"}
    assert answer_replay.answer_env(False) == {"VICA_ANSWER": "off"}


def test_question_for_maps_the_row(dests):
    row = {"kind": "other", "question": "무엇을 도와드릴까요?"}
    asked = answer_replay.question_for(row, dests)
    assert (asked.id, asked.kind, asked.text, asked.option_ids) == ("replay", "open", "무엇을 도와드릴까요?", ())
    confirm = answer_replay.question_for({"kind": "confirm", "question": "사회복지로 안내해드릴까요?"}, dests)
    assert confirm.option_ids == ("d1",)
    choice = answer_replay.question_for({"kind": "choice", "question": "사회복지와 화장실 중 어디로 갈까요?"}, dests)
    assert choice.option_ids == ("d1", "d2")


def test_replay_grades_by_the_answer_field(rows_labeled, dests):
    said_yes = VicaIntent(intent="navigate", matched_destination_id="d1", answer="yes")
    run = replay(rows_labeled[:1], lambda row: (said_yes, 0.5), lambda _p: dests)
    assert run["rows"][0]["status"] == "ok" and run["rows"][0]["pred"] == "yes"


def test_transcript_rule_wins_on_the_audio_path(monkeypatch, rows_labeled, dests):
    from src.reservation_flow import ReserveCall
    monkeypatch.setenv("VICA_ANSWER", "on")
    row = {**rows_labeled[1], "stt": "삼십분."}                 # kind=minutes
    asked = answer_replay.question_for(row, dests)
    heard = VicaIntent(intent="wait", wait_minutes=15, answer="minutes", answer_value="15")
    fixed = answer_replay.with_transcript_rule(heard, row, asked)
    assert (fixed.answer, fixed.answer_value, fixed.wait_minutes) == ("minutes", "30", 15)
    long = {**rows_labeled[1], "stt": "화장실 가고 싶어"}
    assert answer_replay.with_transcript_rule(heard, long, asked) is heard
    call = ReserveCall(facility="세미나실")
    assert answer_replay.with_transcript_rule(call, row, asked) is call
```

(두 시험 파일의 `dests` fixture 는 `d1` 사회복지 · `d2` 화장실이다.)

- [ ] **Step 2: 시험이 실패하는지 확인**

Run: `nice -n 19 .venv/bin/python -m pytest tests/test_answer_replay.py tests/test_answer_grade.py -q -p no:cacheprovider`
Expected: 새 시험 5개 FAIL(`pred_of`·`answer_env`·`question_for`·`with_transcript_rule` 없음), 나머지 통과.

- [ ] **Step 3: 구현**

- `replay()`: `answer_of(intent, …)` 를 `pred_of(intent, …)` 로 바꾸고 `why` 끝에 `/answer={intent.answer or '-'}` 를 붙인다.
- `main()`: `--answer` 를 받아 `os.environ.update(answer_env(args.answer))` 를 `tools_env` 줄 옆에(해석기 import 전에) 둔다. `interpret_text` 는 `parse_intent(…, question=question_for(row, dests))`, `interpret_audio` 는 `parse_intent_audio(…, question=asked)` 뒤에 `with_transcript_rule(intent, row, asked)`. 결과 파일 이름에 `_answer` 를 `_tools` 와 같은 방식으로 붙이고, `setup` 글자에 `대답 칸 켬|끔` 을 더한다.
- `--answer` 없이 돌린 결과는 지금과 같다(스위치 꺼짐 → `question` 을 넘겨도 `answer` 는 빈 값 → `pred_of` 가 `answer_of` 로 읽는다).

- [ ] **Step 4: 시험 통과 확인**

Run: `nice -n 19 .venv/bin/python -m pytest tests/test_answer_replay.py tests/test_answer_grade.py tests/test_reserve_replay.py -q -p no:cacheprovider`
Expected: 전부 통과.

- [ ] **Step 5: 전체 시험(주행 묶음이 꺼져 있을 때)과 커밋**

Run: `.venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: 작업 전 기준 + 새 시험 수만큼 통과, 실패 0.

```bash
git commit -m "feat(voice): 208행 재생이 대답 칸으로 채점 — --answer, 소리 경로는 받아쓴 글자의 짧은 대답 규칙 먼저" -- tools/answer_replay.py tools/answer_grade.py tests/test_answer_replay.py tests/test_answer_grade.py
```

---

### Task 6: 208행 측정과 판정 (설계 5절 1단계)

**Files:**
- 결과: `~/vica_data/voice_answer_audit_20261113/replay_<path>[_answer]_<stamp>.json·.md`, `answer_stage1_summary_<YYYYMMDD>.txt`, 그 폴더 `README.md` 끝
- Modify: `docs/superpowers/specs/2026-11-18-answer-interpretation-design.md` 5절 아래에 결과 표

**Interfaces:**
- Consumes: Task 5 의 `--answer`, 채점 도구 `~/vica_data/voice_answer_audit_20261113/summarize_confirmed.py`(`--date`·`--extra PATH=FILE`)
- Produces: 경로별 잘못된 행동 수·지연 p50·p90 표와 판정(2단계로 갈지)

- [ ] **Step 1: 조건 확인** — 주행 묶음이 꺼져 있는지(`systemctl --user is-active vica-drive.target` → `inactive`), Ollama 가 떠 있는지(`curl -s localhost:11434/api/tags` 에 `midm2-mini`), 미커밋 변경이 없는지 본다. 하나라도 아니면 사용자에게 알리고 기다린다.

- [ ] **Step 2: 사용자 승인** — 외부 API 호출 수를 말하고 승인받는다: gpt-5.4-mini 208행 × 2회, Realtime 208행 × 3회(저장 소리가 있는 행만 간다).

- [ ] **Step 3: 대조 — 스위치 끔, 로컬 1회**

Run: `.venv/bin/python -m tools.answer_replay --path midm --labels ~/vica_data/voice_answer_audit_20261113/labeled.csv`
Expected: 208행 기준 잘못된 행동 17(11-18 과 같다). 다르면 원인(예약 커밋·목적지 파일)을 먼저 찾는다 — 측정을 이어 가지 않는다.

- [ ] **Step 4: 측정 — 스위치 켬**

```bash
.venv/bin/python -m tools.answer_replay --path midm --answer --labels ~/vica_data/voice_answer_audit_20261113/labeled.csv
.venv/bin/python -m tools.answer_replay --path gpt --answer --rounds 2 --labels ~/vica_data/voice_answer_audit_20261113/labeled.csv
.venv/bin/python -m tools.answer_replay --path realtime --answer --rounds 3 --labels ~/vica_data/voice_answer_audit_20261113/labeled.csv
```

Expected: 경로마다 `→ …/replay_<path>_answer_<stamp>.json`. 오류가 5건 이어져 멈추면(스키마 거절 등) 그 오류 글자를 보고 원인을 고친 뒤 다시 돌린다.

- [ ] **Step 5: 208행 기준으로 채점**

Run: `.venv/bin/python ~/vica_data/voice_answer_audit_20261113/summarize_confirmed.py --date 20261118 --extra midm=<Step 3 파일> --extra midm=<Step 4 midm 파일> --extra gpt=<gpt 파일> --extra realtime=<realtime 파일> > ~/vica_data/voice_answer_audit_20261113/answer_stage1_summary_<YYYYMMDD>.txt`
Expected: 경로마다 회차별 표(앞이 오늘 재생, 뒤가 11-18 기준).

- [ ] **Step 6: 판정** — 잘못된 행동이 Mi:dm 19 이하 · gpt-5.4-mini 14 이하 · Realtime 15 이하면 합격(기준 17 · 10~12 · 11~13 에서 2 넘게 나빠지지 않음). 함께 적는다: Realtime 의 짧은 대답 7행(분 3 · "안내가 필요 없으신가요?" 3 · 본인 확인 "네. 네." 1)이 맞음으로 바뀌었는지, 지연 p50·p90 이 11-18 표보다 얼마나 늘었는지, 새로 틀린 행의 목록.

- [ ] **Step 7: 기록과 보고** — 결과 표를 설계서 5절 아래와 측정 폴더 `README.md` 끝에 적고 커밋한다(`docs(voice): 3-나-2 1단계 측정 결과`). 사용자에게 표와 판정을 보고한다. 합격이면 `.env` 의 `VICA_ANSWER=on` 을 켤지와 2단계 계획을 묻고, 불합격이면 2단계로 가지 않고 틀린 행을 보여 준다.

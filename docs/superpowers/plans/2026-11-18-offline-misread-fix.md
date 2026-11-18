# 3-나-1 오프라인 일시정지·재개 오독 수리 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 글자 경로에서 일시정지·재개 말을 LLM 없이 규칙으로 읽고, 로봇 "네?"가 섞인 글자를 정리하고, 일시정지 중의 거절이 일시정지를 풀지 않게 한다.

**Architecture:** 음성 저장소에 순수 함수 `wake_echo.strip_wake_echo` 를 새로 두고 `ros_node._on_user_text` 맨 앞에서 부른다(질문 기록 `/vica/question` 이 `kind=open`·"네?"일 때만). 글자 규칙은 기존 `langchain_intent_parser._shortcut_intent` 에 낱말을 더한다. 미션은 `mission_logic.on_intent` 의 거절 갈래 한 줄을 고친다. LLM 지시문·출력 형식·대화 기록·소리 경로는 건드리지 않는다.

**Tech Stack:** Python 3.10 · pytest · ROS 2 Humble(rclpy, 미션은 `colcon build` 뒤 설치본 실행 · 음성은 소스에서 바로 실행)

**Spec:** `docs/superpowers/specs/2026-11-18-offline-misread-fix-design.md` (바탕 `2026-11-14-question-aware-answers-design.md`)

## Global Constraints

- 저장소 둘: 음성 `~/VICA-smarthandle/vica-voice-llm`, ROS `~/VICA-smarthandle/vica_ros2_ws`. 브랜치는 둘 다 `after_feedback_waiting`. 작업마다 커밋, **푸시하지 않는다**(사용자 11-18 "푸시 보류").
- 커밋 메시지는 기존 문체(한국어, `feat(voice): …`), 끝에 `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.
- **확인 없이 출발하는 길을 만들지 않는다.** 재개 규칙의 결과는 `resume`·`need_confirm=True`·`reply=RESUME_CONFIRM`("다시 출발할까요?")이다. 예외는 그 질문이 이미 열려 있을 때뿐(기존 `_finalize` 의 `pending_command == "resume"` 와 같은 동작).
- **소리 경로는 고치지 않는다**: `src/realtime_intent.py`, `parse_intent_audio`, `ros_node._on_user_audio` 의 diff 는 0줄.
- 새 멘트 없음. 문구는 `src/replies.py` 의 `WAKE_GREETING`("네?")·`RESUME_CONFIRM`·`PAUSE_ACK` 와 미션의 기존 상수만 쓴다.
- rclpy 로거를 조건식으로 고르지 않는다(`tests/test_one_voice_wiring.py::test_no_logger_picked_by_condition`).
- 미션 빌드·유닛 재시작은 그때마다 사용자에게 묻는다. 유닛을 직접 `systemctl stop` 하지 않는다(모드 서비스 또는 사용자).
- 주행 묶음(`vica-drive.target`)이 켜져 있으면 Ollama 재생과 전체 시험은 미루고 해당 시험 파일만 돌린다.
- 시험 명령 · 작업 전 기준
  - 음성: `cd ~/VICA-smarthandle/vica-voice-llm && .venv/bin/python -m pytest -q -p no:cacheprovider` → 1626 passed
  - 미션: `cd ~/VICA-smarthandle/vica_ros2_ws/src/vica_mission_manager && source /opt/ros/humble/setup.bash && source ~/VICA-smarthandle/vica_ros2_ws/install/setup.bash && python3 -m pytest test -q -p no:cacheprovider` → 5 failed(`test_nav_cancel_race` 2 · `test_progress_narration` 3, 작업 전부터) · 나머지 통과

## Review Focus

1. "네." 한마디(뒤에 말 없음)가 호출 응답 뒤에 오면 그대로 남는다 — 빈 글자가 되어 발화가 사라지면 안 된다. → Task 1 `test_bare_ne_is_kept`
2. "네가 누구야"·"네네 알겠어"처럼 "네"로 시작하는 다른 낱말은 떼지 않는다. → Task 1 `test_other_words_starting_with_ne_are_kept`
3. 열린 질문이 "네?"가 아닐 때(확인 질문·"다시 출발할까요?"·질문 없음)의 "네 화장실로 가자"는 떼지 않는다 — 승낙의 "네"가 사라지면 안 된다. → Task 1 `test_only_after_the_wake_greeting`
4. 글자를 떼도 화자 판정(`SpeakerLink`, 발화 글자로 맞춘다)이 끊기지 않는다 — 끊기면 대기 중 본인 확인이 판정 없이 돈다. → Task 2 `test_speaker_link_uses_the_raw_text`
5. 일시정지가 아닌 상태(쉬는 중·주행 중·대기 중·손 놓침 정지)의 "출발"·"다시 가자"는 규칙이 잡지 않는다. → Task 3 `test_resume_words_only_while_paused`
6. 일시정지 중 거절 뒤에 "다시 가자"가 실제로 통한다(보관한 목적지가 남는다). → Task 4 `test_rejected_request_while_paused_keeps_the_pause`

---

### Task 1: 호출 응답 메아리 떼기 — 순수 함수 (음성, 설계 4.1)

**Files:**
- Create: `src/wake_echo.py`
- Test: `tests/test_wake_echo.py`

**Interfaces:**
- Consumes: `src.question_record.OpenQuestion`(칸 `kind`·`text`), `src.replies.WAKE_GREETING`
- Produces: `strip_wake_echo(text: str, question: Optional[OpenQuestion]) -> str` — 뗄 것이 없으면 받은 `text` 를 그대로 돌려준다.

- [ ] **Step 1: 실패하는 시험 작성** — `tests/test_wake_echo.py`

```python
"""호출 응답 "네?" 메아리 떼기(3-나-1, 설계 2026-11-18 4.1)."""
import pytest

from src.question_record import OpenQuestion
from src.replies import WAKE_GREETING
from src.wake_echo import strip_wake_echo


def q(kind="open", text=WAKE_GREETING):
    return OpenQuestion(id="q1", kind=kind, text=text, option_ids=(), opened_at=0.0)


@pytest.mark.parametrize("said, want", [
    ("네 잠깐만.", "잠깐만."),                 # 11-18 13:15:23
    ("네. 다시 가자.", "다시 가자."),           # 11-18 13:15:50
    ("네, 화장실로 가자", "화장실로 가자"),
    ("네? 화장실로 가자", "화장실로 가자"),
    ("네.다시 가자", "다시 가자"),
    ("  네 나 배 아파", "나 배 아파"),
    ("네 네 네", "네 네"),                     # 한 낱말만 뗀다
])
def test_leading_ne_is_stripped_after_the_wake_greeting(said, want):
    assert strip_wake_echo(said, q()) == want


@pytest.mark.parametrize("said", ["네", "네.", "네?", "네!"])
def test_bare_ne_is_kept(said):
    assert strip_wake_echo(said, q()) == said


@pytest.mark.parametrize("said", ["네가 누구야", "네네 알겠어", "네모난 방으로 가자", "그래 네 말이 맞아", "화장실로 가자"])
def test_other_words_starting_with_ne_are_kept(said):
    assert strip_wake_echo(said, q()) == said


@pytest.mark.parametrize("question", [
    None,
    q(kind="confirm", text="화장실로 안내해드릴까요?"),
    q(kind="yesno", text="다시 출발할까요?"),
    q(kind="open", text="어디로 안내해드릴까요?"),
    q(kind="yesno", text=WAKE_GREETING),
])
def test_only_after_the_wake_greeting(question):
    assert strip_wake_echo("네 화장실로 가자", question) == "네 화장실로 가자"
```

- [ ] **Step 2: 실패 확인** — `.venv/bin/python -m pytest tests/test_wake_echo.py -q -p no:cacheprovider` → `ModuleNotFoundError: No module named 'src.wake_echo'`

- [ ] **Step 3: `strip_wake_echo` 구현** — `src/wake_echo.py`. 조건은 `question is not None and question.kind == "open" and question.text.strip() == WAKE_GREETING`. 떼는 꼴은 정규식 하나: `^\s*네(?:[.,?!]+\s*|\s+)(?=\S)` (맨 앞 "네" + 문장부호나 공백, 뒤에 글자가 남을 때만). 모듈 머리말에 원인(호출 확정과 동시에 듣기 창이 열리고 "네?"가 재생된다, 11-18 13:15:19)과 근거 수치(22일 자료 "네?" 뒤 378건 중 36건)를 한두 줄로 적는다.

- [ ] **Step 4: 통과 확인** — 같은 명령 → 21 passed

- [ ] **Step 5: 커밋**

```bash
git add src/wake_echo.py tests/test_wake_echo.py
git commit -m "feat(voice): 호출 응답 \"네?\" 메아리 떼기 — 열린 질문이 \"네?\"일 때 맨 앞 '네' 한 낱말(3-나-1, 설계 4.1)"
```

---

### Task 2: 메아리 떼기 배선 — `ros_node._on_user_text` (음성, 설계 4.1)

**Files:**
- Modify: `src/ros_node.py` — `_on_user_text`(295줄~), `_publish_intent`(390줄~, 화자 판정 찾는 줄 399), import 줄(59줄 근처)
- Test: `tests/test_wake_echo_wiring.py` (새 파일 — `ros_node` 는 rclpy 없이 못 띄워 소스 구조로 확인, `tests/test_one_voice_wiring.py` 와 같은 방식)

**Interfaces:**
- Consumes: Task 1 `strip_wake_echo(text, question)`, `self._question`(`_on_question` 이 채우는 열린 질문), `tests.test_one_voice_wiring._method`·`_calls`
- Produces: `_publish_intent(self, intent, text, llm_first=False, link_text: str = "")` — `link_text` 는 화자 판정을 찾을 때 쓰는 원문 글자. 비면 `text` 로 찾는다(소리 경로의 두 호출은 그대로).

- [ ] **Step 1: 실패하는 시험 작성** — `tests/test_wake_echo_wiring.py`

```python
"""호출 응답 메아리 떼기 배선(3-나-1) — ros_node 소스 구조로 확인한다."""
from __future__ import annotations

import ast

from tests.test_one_voice_wiring import _calls, _method


def test_user_text_is_stripped_before_any_rule():
    fn = _method("_on_user_text")
    (strip,) = _calls(fn, "strip_wake_echo")
    assert ast.unparse(strip.args[1]) == "self._question"
    for later in ("stop_pause_intent_for", "detect_emergency", "parse_intent"):
        assert strip.lineno < min(c.lineno for c in _calls(fn, later)), later


def test_speaker_link_uses_the_raw_text():
    """화자 판정은 호출어 노드가 낸 원문 글자로 찾는다 — 뗀 글자로 찾으면 판정이 끊긴다."""
    fn = _method("_on_user_text")
    (strip,) = _calls(fn, "strip_wake_echo")
    raw = ast.unparse(strip.args[0])
    publishes = _calls(fn, "self._publish_intent")
    assert len(publishes) == 2
    for call in publishes:
        kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
        assert kw.get("link_text") == raw
    (take,) = _calls(_method("_publish_intent"), "self._spk_link.take")
    assert ast.unparse(take.args[0]) == "link_text or text"


def test_audio_path_is_untouched():
    assert not _calls(_method("_on_user_audio"), "strip_wake_echo")
```

- [ ] **Step 2: 실패 확인** — `.venv/bin/python -m pytest tests/test_wake_echo_wiring.py -q -p no:cacheprovider` → 앞 두 시험 FAIL(`strip_wake_echo` 호출 없음 — 풀기 실패 `ValueError`), 세 번째 PASS

- [ ] **Step 3: 구현** — `src/ros_node.py`
  - `from .wake_echo import strip_wake_echo` 추가.
  - `_on_user_text`: 지금의 `text = msg.data.strip()` 을 `raw = msg.data.strip()` 로 받고, 빈 글자 검사 뒤 `text = strip_wake_echo(raw, self._question)`. 달라졌으면 `self.get_logger().info(f"앞 '네' 뗌: '{raw}' → '{text}'")` 한 줄. 그 아래는 지금처럼 `text` 를 쓴다(멈춤 말·긴급어·글자 규칙·LLM·대화 기록).
  - `_on_user_text` 안의 `self._publish_intent(...)` 두 곳(멈춤 말 갈래, 맨 끝)에 `link_text=raw` 를 넘긴다.
  - `_publish_intent` 에 `link_text: str = ""` 매개변수를 더하고 `self._spk_link.take(link_text or text, time.time())` 로 바꾼다. 다른 `text` 사용처는 그대로.

- [ ] **Step 4: 통과 확인** — `.venv/bin/python -m pytest tests/test_wake_echo_wiring.py tests/test_one_voice_wiring.py -q -p no:cacheprovider` → 12 passed. 이어서 `source /opt/ros/humble/setup.bash && source ~/VICA-smarthandle/vica_ros2_ws/install/setup.bash && .venv/bin/python -c "import src.ros_node"` → 오류 없음. `git diff --stat HEAD -- src/realtime_intent.py` → 출력 없음.

- [ ] **Step 5: 커밋**

```bash
git add src/ros_node.py tests/test_wake_echo_wiring.py
git commit -m "feat(voice): 글자 경로 발화에서 호출 응답 메아리를 뗀 뒤 해석한다 — 화자 판정은 원문 글자로 찾는다(3-나-1)"
```

---

### Task 3: 일시정지·재개 글자 규칙 — `_shortcut_intent` (음성, 설계 4.2)

**Files:**
- Modify: `src/langchain_intent_parser.py` — `_PAUSE_WORDS`(406줄), 새 상수 `_RESUME_WORDS`, `_shortcut_intent` 의 `if word in _PAUSE_WORDS:` 바로 아래(758줄 근처)
- Test: `tests/test_command_intents.py`(클래스 추가)

**Interfaces:**
- Consumes: `_shortcut_intent(user_text, history, destinations, dialog_state="", local=False) -> Optional[VicaIntent]`, 그 안의 `pending_command`(직전 로봇 말이 `CANCEL_CONFIRM`/`RESUME_CONFIRM` 이면 `"cancel"`/`"resume"`), `_normalize_short_reply`(문장부호·공백 제거)
- Produces: 없음(이 함수의 결과가 달라질 뿐). `dialog_state` 는 `parse_intent` 가 로컬 규칙(`VICA_LOCAL_RULES=1`·글자 입력)일 때만 넘기므로 재개 규칙은 오프라인 실행에서만 걸린다 — `parse_intent` 는 고치지 않는다.

- [ ] **Step 1: 실패하는 시험 작성** — `tests/test_command_intents.py` 끝에 추가(import 에 `_shortcut_intent`, `is_instant_utterance` 를 더한다)

```python
class TestPauseResumeTextRules:
    """일시정지·재개 말은 LLM 없이 읽는다(3-나-1, 설계 2026-11-18 4.2). 11-18 실기: 로컬 LLM 이 이 말들을 407호 이동으로 읽었다."""

    @pytest.mark.parametrize("said", ["잠깐", "잠깐.", "잠시", "잠깐만", "잠시만요"])
    def test_short_pause_words(self, said):
        result = _shortcut_intent(said, [], [DEST])
        assert (result.intent, result.need_confirm, result.reply) == ("pause", False, PAUSE_ACK)
        assert is_instant_utterance(said)

    @pytest.mark.parametrize("said", [
        "다시 가자.", "다시 가", "다시 가줘", "다시 출발", "다시 출발.", "다시 출발해", "다시 출발해줘", "다시 출발하자",
        "출발", "출발해", "출발해줘", "출발하자", "계속 가자", "계속 가", "계속 가줘",
        "다시 가요", "다시 출발해요", "다시 출발해 주세요", "출발해요", "출발해 주세요", "계속 가요", "계속 가 주세요",
    ])
    def test_resume_words_ask_before_moving(self, said):
        result = _shortcut_intent(said, [], [DEST], dialog_state="paused", local=True)
        assert (result.intent, result.need_confirm, result.reply) == ("resume", True, RESUME_CONFIRM)

    @pytest.mark.parametrize("dialog_state", ["", "idle", "navigating", "waiting", "asking_next", "paused_handle"])
    def test_resume_words_only_while_paused(self, dialog_state):
        for said in ("다시 가자.", "출발"):
            assert _shortcut_intent(said, [], [DEST], dialog_state=dialog_state, local=True) is None

    @pytest.mark.parametrize("said", ["다시 가자고 했잖아", "출발지가 어디야", "화장실로 다시 가자", "다시"])
    def test_longer_sentences_are_left_to_the_llm(self, said):
        assert _shortcut_intent(said, [], [DEST], dialog_state="paused", local=True) is None

    def test_resume_word_answers_the_open_resume_question(self):
        # "다시 출발할까요?"에 "다시 가자" — 같은 질문을 또 하지 않는다(_finalize 의 pending_command 와 같은 동작).
        result = _shortcut_intent("다시 가자", [AIMessage(RESUME_CONFIRM)], [DEST], dialog_state="paused", local=True)
        assert (result.intent, result.need_confirm, result.reply) == ("resume", False, "")

    def test_resume_word_during_cancel_question_asks_anew(self):
        result = _shortcut_intent("다시 가자", [AIMessage(CANCEL_CONFIRM)], [DEST], dialog_state="paused", local=True)
        assert (result.intent, result.need_confirm, result.reply) == ("resume", True, RESUME_CONFIRM)
```

파일 머리에 `import pytest` 가 없으면 더한다.

- [ ] **Step 2: 실패 확인** — `.venv/bin/python -m pytest tests/test_command_intents.py -q -p no:cacheprovider -k TestPauseResumeTextRules` → "잠깐"·"잠깐."·"잠시" 3개와 재개 낱말 22개 · 열린 질문 2개가 `AttributeError: 'NoneType' object has no attribute 'intent'` 로 FAIL, 나머지 PASS

- [ ] **Step 3: 구현** — `src/langchain_intent_parser.py`
  - `_PAUSE_WORDS` 에 `"잠깐"`, `"잠시"` 추가.
  - `_RESUME_WORDS`(정규화 뒤 통째 일치): `다시가자 다시가 다시가줘 다시출발 다시출발해 다시출발해줘 다시출발하자 출발 출발해 출발해줘 출발하자 계속가자 계속가 계속가줘 다시가요 다시출발해요 다시출발해주세요 출발해요 출발해주세요 계속가요 계속가주세요`. 주석에 근거(11-18 실기 재생 "다시 가자"류 0/8, 미션 안내 "다시 출발하려면 말씀해 주세요")와 한계(받아쓰기가 깨진 꼴은 넣지 않는다)를 적는다.
  - `_shortcut_intent` 의 일시정지 낱말 검사 바로 아래에 재개 규칙: `dialog_state == "paused" and word in _RESUME_WORDS` 이면 — `pending_command == "resume"` 일 때 `VicaIntent(intent="resume", confidence=1.0, reply="", need_confirm=False)`, 아니면 `VicaIntent(intent="resume", confidence=1.0, reply=RESUME_CONFIRM, need_confirm=True)`.

- [ ] **Step 4: 통과 확인** — `.venv/bin/python -m pytest tests/test_command_intents.py tests/test_local_rules.py tests/test_affirm_deny.py tests/test_spoken_text.py -q -p no:cacheprovider` → 전부 통과. 주행 묶음이 꺼져 있으면 전체 시험 → 1626 + 새 시험 수만큼 passed.

- [ ] **Step 5: 커밋**

```bash
git add src/langchain_intent_parser.py tests/test_command_intents.py
git commit -m "feat(voice): 일시정지·재개 말은 글자 규칙으로 — '잠깐'·'잠시', 일시정지 중 '다시 가자'류는 \"다시 출발할까요?\"(3-나-1, 설계 4.2)"
```

---

### Task 4: 일시정지 중 거절은 일시정지를 둔다 (미션, 설계 4.3)

**Files:**
- Modify: `~/VICA-smarthandle/vica_ros2_ws/src/vica_mission_manager/vica_mission_manager/mission_logic.py` — `on_intent` 의 확정 요청 거절 갈래(`reason = check_gate(...)` 아래 `elif self.state not in _WAIT_STATES: self._to_idle()`, 2595줄 근처)
- Test: `~/VICA-smarthandle/vica_ros2_ws/src/vica_mission_manager/test/test_reaction_rules.py`

**Interfaces:**
- Consumes: `reaction_states.paused()`(409호로 가다 "잠깐"으로 선 상태), `intent()`, `lookup`, `BOUNDS`, `MSG_PRIVATE_DEST`, `MSG_RESUMED`, `say_destination`
- Produces: 없음(상태 전이 하나가 달라진다: 일시정지 중 확정 요청 거절 → `State.PAUSED` 유지).

- [ ] **Step 1: 실패하는 시험 작성** — `test/test_reaction_rules.py` 끝에 추가(import 에 `MSG_RESUMED` 를 더한다)

```python
def test_rejected_request_while_paused_keeps_the_pause():
    """일시정지 중 갈 수 없는 곳 요청 — 거절만 말하고 일시정지는 둔다(11-18 실기 13:15:52: paused → idle 로
    보관한 목적지가 사라져 "다시 가자"가 "다시 출발할 안내가 없습니다"가 됐다)."""
    closed = ROOM.__class__(**{**ROOM.__dict__, "id": "vip", "name": "원장실", "authorization": "private"})
    look = lambda i: closed if i == "vip" else lookup(i)   # noqa: E731
    logic, t = paused()
    acts = logic.on_voice_intent(intent(matched_destination_id="vip"), t, look, BOUNDS, True)
    assert _says(acts) == [MSG_PRIVATE_DEST]
    assert logic.state == State.PAUSED
    acts = logic.on_voice_intent(intent("resume"), t + 2, look, BOUNDS, True)
    assert _says(acts) == [say_destination(MSG_RESUMED, "409호")]
    assert logic.state == State.NAVIGATING


def test_unknown_place_while_paused_keeps_the_pause():
    logic, t = paused()
    logic.on_voice_intent(intent(matched_destination_id="nowhere"), t, lookup, BOUNDS, True)
    assert logic.state == State.PAUSED
```

- [ ] **Step 2: 실패 확인** — (미션 시험 환경에서) `python3 -m pytest test/test_reaction_rules.py -q -p no:cacheprovider -k "while_paused"` → 둘 다 FAIL(`State.IDLE != State.PAUSED`)

- [ ] **Step 3: 구현** — 그 `elif` 를 `elif self.state not in _WAIT_STATES and self.state != State.PAUSED:` 로. 주석 한 줄: 일시정지 중 거절도 대기처럼 그대로 둔다 — 보관한 목적지로 "다시 가자"가 통해야 한다(11-18 실기).

- [ ] **Step 4: 통과 확인** — `python3 -m pytest test -q -p no:cacheprovider` → 5 failed(기준과 같은 5개) · 나머지 통과(작업 전보다 2개 많음). `test_reaction_table.py` 의 `PAUSED` 행은 그대로 통과해야 한다.

- [ ] **Step 5: 커밋** (ROS 저장소)

```bash
cd ~/VICA-smarthandle/vica_ros2_ws
git add src/vica_mission_manager/vica_mission_manager/mission_logic.py src/vica_mission_manager/test/test_reaction_rules.py
git commit -m "fix(mission): 일시정지 중 확정 요청이 거절돼도 일시정지를 둔다 — 보관한 목적지로 '다시 가자'가 통하게(11-18 실기)"
```

---

### Task 5: 재생·빌드·실기 확인 (설계 5절)

**Files:**
- Modify(저장소 밖 측정 도구): `~/vica_data/field_test_20261118/hist_replay.py` — 재생 때 Task 1 의 떼기를 건다
- Modify: `docs/superpowers/specs/2026-11-18-offline-misread-fix-design.md` 5절에 결과 수치

**Interfaces:**
- Consumes: Task 1~4 의 결과. `hist_replay.py` 의 발화 dict 칸 `q_kind`(열린 질문 종류)·`history_for()`.

- [ ] **Step 1: 측정 도구에 떼기 적용** — `cmd_run` 에서 발화마다, `u["q_kind"] == "open"` 이고 그 발화 직전 기록의 마지막 로봇 말이 "네?"이면 `OpenQuestion(id="r", kind="open", text="네?", option_ids=(), opened_at=0.0)` 로 `strip_wake_echo(u["text"], …)` 를 건 글자를 `parse_intent`·`_shortcut_intent` 에 넣는다. 후보 이름 `R`(기록은 `V0` 와 같다).

- [ ] **Step 2: 74발화 재생(주행 묶음이 꺼져 있을 때만)** — `systemctl --user is-active vica-drive.target` 가 `inactive` 인지 보고, Ollama 가 떠 있지 않으면 스택과 같은 설정으로 켠다(`OLLAMA_KEEP_ALIVE=-1 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_GO_TEMPLATE=1`). `cd ~/VICA-smarthandle/vica-voice-llm && .venv/bin/python ~/vica_data/field_test_20261118/hist_replay.py run R && … report`.
  기대: `off-26`·`off-28`·`on-13`·`on-14`·`on-17`·`on-21` 이 `규칙` 으로 처리되고 맞음, `on-22`("그래")는 고정 문장(719a5e0) 덕에 맞음. 잘못된 행동 수가 `V0` 의 14 보다 줄어든다(계산값 9 — `off-29`·`off-30` 은 로그의 상태가 쉬는 중이라 재생에서는 그대로 남을 수 있다. 그 둘은 Task 4 가 실기에서 푼다).

- [ ] **Step 3: 208행 재생(같은 조건)** — `.venv/bin/python -m tools.answer_replay --path midm --labels ~/vica_data/voice_answer_audit_20261113/labeled.csv --rounds 1`. 정답 파일 옆에 생기는 `replay_midm_<시각>.md` 의 잘못된 행동 수를 본다. 기대: 14 이하(11-17 기준 `baseline_summary_20261117.txt`). 넘으면 늘어난 행을 적어 사용자에게 보고하고 멈춘다.

- [ ] **Step 4: 미션 빌드·재시작(사용자 승인 뒤)** — 주행 묶음이 꺼진 때에 `cd ~/VICA-smarthandle/vica_ros2_ws && source /opt/ros/humble/setup.bash && colcon build --packages-select vica_mission_manager`. 설치본과 소스의 `mission_logic.py` diff 0 확인. 주행 켜기는 앱(모드 서비스) 또는 사용자.

- [ ] **Step 5: 실기(오프라인, 사용자와 함께 — 주행 장면이라 손잡이 잡을 사람·비상 정지 준비)**
  1. 안내 주행 중 "비카야" → "잠깐" → 미션 "잠시 멈추겠습니다. 다시 출발하려면 말씀해 주세요." (음성 로그에 LLM 호출 없이 `intent=pause`)
  2. "비카야" → "다시 가자" → 미션 "다시 출발할까요?" → "네" → "…로 다시 출발합니다." (음성 로그 `앞 '네' 뗌` 이 찍히면 그 줄도 확인)
  3. 다시 "잠깐" → "비카야" → "407호로 가자" → "비공개 목적지는 안내할 수 없습니다." → 미션 로그 `paused -> paused` → "비카야" → "다시 가자" → "다시 출발할까요?" → "네" → 출발
  4. (Task 6) 다시 "잠깐" → "비카야" → "사회복지로 가자" → 미션 "사회복지로 안내해드릴까요?" → "아니" → **"알겠습니다. 다시 출발하려면 말씀해 주세요."** → 미션 로그 `confirming -> paused` → "비카야" → "다시 가자" → "다시 출발할까요?" → "네" → 원래 목적지로 출발
  합격: 네 장면 모두 위 말·상태대로, 음성 로그에 407호 `navigate` 가 2번 장면에서 0건.

- [ ] **Step 6: 결과 기록·커밋** — 설계서 5절에 재생 수치(74발화 `R`, 208행)와 실기 결과를 적고 커밋. 측정 파일은 `~/vica_data/field_test_20261118/` 에 둔다.

```bash
git add docs/superpowers/specs/2026-11-18-offline-misread-fix-design.md
git commit -m "docs(voice): 3-나-1 재생·실기 결과"
```

---

### Task 6: 일시정지 중 제안이 접히면 일시정지로 (미션, 설계 4.4 — 최종 검토 I4 뒤 추가, 사용자 11-18)

**Files:**
- Modify: `~/VICA-smarthandle/vica_ros2_ws/src/vica_mission_manager/vica_mission_manager/mission_logic.py` — 새 상수 `MSG_PAUSE_KEPT`, 표시 `self._pause_hold`, `on_intent`(제안으로 CONFIRMING 에 들어가는 자리), `_fold_confirming`, `on_confirm_answer`(아니요), `on_tick`(확인 무응답), 출발 시작 블록, `_to_idle`
- Test: `test/test_reaction_rules.py`

**Interfaces:**
- Produces: `MSG_PAUSE_KEPT = "알겠습니다. 다시 출발하려면 말씀해 주세요."`(사용자 지정 글자 그대로). 반응표(`test_reaction_table.py`)의 `PAUSED` 행은 그대로다 — 첫 반응(확인 질문·CONFIRMING)은 바뀌지 않는다.

- [x] 시험 5개: 아니요 → `[MSG_PAUSE_KEPT]`·`State.PAUSED`·이어진 resume 이 409호로 출발 / 무응답(16 s 에 한 번 다시 묻고 `confirm_timeout_sec` 뒤) → 같음 / `on_wake_call` → `[MSG_WAKE_GREETING]`·`State.PAUSED` / "네" → 엘리베이터로 출발 / 새 안내를 취소한 뒤의 제안 거절은 `State.IDLE`(낡은 보류 없음)
- [x] 실패 확인: 앞 3개 FAIL(`State.IDLE != State.PAUSED` 등), 뒤 2개는 지금 동작을 고정하는 시험이라 PASS
- [x] 구현 뒤 `python3 -m pytest test -q -p no:cacheprovider` → 5 failed(기준과 같은 5개) · 1780 passed
- [ ] 미션 빌드(주행 묶음이 꺼진 때) → Task 5 Step 5 실기 4번 장면

**이 계획 밖(설계 6절):** 지시문 예시 `"407호 가자"` 바꾸기(E)·지시문 규칙(D)의 효과 측정은 로봇이 빌 때 `hist_replay.py run D,E,DE` 로 따로 재고, 결과를 사용자에게 보고한 뒤 넣을지 정한다.

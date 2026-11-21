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

    def invoke(self, messages, local_messages=None):   # 진짜 LlmBackendManager.invoke 와 같은 꼴
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
    monkeypatch.setenv("VICA_ANSWER_LOCAL", "on")      # 3.5 — 로컬에 대답 칸을 줄 때만 종류별로 좁힌다
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


def test_answer_that_is_not_text_does_not_fail_the_utterance(on):
    """외부 API 가 answer 에 null·숫자를 넣어도 발화가 통째로 실패하지 않는다(검토 M2) — 빈 값으로 읽는다."""
    from src.realtime_intent import RealtimeResult
    assert parser._AnswerDraft(intent="unknown", answer=None).answer == ""
    assert parser._AnswerDraft(intent="unknown", answer=3).answer == ""
    result = RealtimeResult(draft={"intent": "unknown", "answer": None}, heard_text="뭐라고", latency_sec=1.0)
    intent, _heard, _dt, _info = parser._deliver_realtime_result(result, DESTS, None, question=q())
    assert (intent.intent, intent.answer) == ("unknown", "")


# ----- 3.5 — 로컬(Ollama)은 질문 줄·대답 칸 없이 규칙 대답만 -------------------------------------
from src.llm_backend import ProbeResult


class _Rec:
    """호출마다 (누가, 출력 클래스, messages)를 적고 draft 를 돌려주는 가짜 LLM."""

    def __init__(self, calls, who, fail=None):
        self.calls, self.who, self.fail = calls, who, fail

    def with_structured_output(self, schema, **kw):
        outer = self
        return type("R", (), {"invoke": lambda s, messages: outer._answer(schema, messages)})()

    def _answer(self, schema, messages):
        self.calls.append((self.who, schema, messages))
        if self.fail:
            raise self.fail
        kw = {"intent": "unknown", "reply": "음"}
        if "answer" in schema.model_fields:
            kw["answer"] = "unclear"
        return schema(**kw)


@pytest.fixture
def backends(monkeypatch):
    parser.reset_backend_manager()
    calls, cfg = [], {"cloud_fail": None}

    class FakeChat:
        def __init__(self, **kw):
            pass

        def with_structured_output(self, schema, **kw):
            return _Rec(calls, "local").with_structured_output(schema)

    def cloud_llm(model, **kw):
        rec = _Rec(calls, "cloud", cfg["cloud_fail"])
        return type("C", (), {"invoke": lambda s, m: rec._answer(parser._draft_cls(), m)})()

    monkeypatch.setattr(parser, "ChatOllama", FakeChat)
    monkeypatch.setattr(parser, "FALLBACK_MODEL", "midm2-mini")
    monkeypatch.setattr(parser, "_get_cloud_llm", cloud_llm)
    monkeypatch.setattr(parser, "_cloud_probe", lambda: ProbeResult.DEAD)
    monkeypatch.setenv("VICA_ANSWER", "on")
    yield calls, cfg
    parser.reset_backend_manager()


def _sys(messages):
    return messages[0].content


def _ollama_provider(monkeypatch, calls):
    monkeypatch.setattr(parser, "PROVIDER", "ollama")
    monkeypatch.setattr(parser, "_get_cloud_llm", lambda model, **kw: parser._ollama_structured(
        type("O", (), {"with_structured_output": lambda s, schema, **k: _Rec(calls, "ollama").with_structured_output(schema)})()))
    parser.reset_backend_manager()


def test_local_off_ollama_has_no_question_line_and_no_answer_field(monkeypatch, backends):
    calls, _ = backends
    _ollama_provider(monkeypatch, calls)
    parser.parse_intent("그냥", DESTS, question=q("yesno", "다시 출발할까요?"))
    who, schema, messages = calls[-1]
    assert who == "ollama" and schema is parser._IntentDraft and "answer" not in schema.model_fields
    assert "로봇이 물은 질문" not in _sys(messages)
    monkeypatch.delenv("VICA_ANSWER")        # 대답 칸을 끈 지시문과 글자까지 같다
    assert _sys(messages) == parser._build_system_prompt(DESTS)


@pytest.mark.parametrize("text,kind,want", [("네", "confirm", ("yes", "")), ("아니요", "yesno", ("no", "")),
                                            ("오 분", "minutes", ("minutes", "5"))])
def test_local_off_still_fills_the_rule_answer(monkeypatch, backends, text, kind, want):
    calls, _ = backends
    _ollama_provider(monkeypatch, calls)
    out = parser.parse_intent(text, DESTS, question=q(kind, "질문?"))
    assert all(c[1] is parser._IntentDraft for c in calls)    # 네/아니요는 지름길이라 LLM 을 안 부를 수도 있다
    assert (out.answer, out.answer_value, out.question_id) == (*want, "q1")


def test_local_off_without_a_rule_answer_leaves_it_empty(monkeypatch, backends):
    calls, _ = backends
    _ollama_provider(monkeypatch, calls)
    out = parser.parse_intent("안내소 가자", DESTS, question=q("confirm"))
    assert (out.answer, out.answer_value, out.question_id) == ("", "", "q1")


def test_local_on_ollama_keeps_question_line_and_narrowed_schema(monkeypatch, backends):
    calls, _ = backends
    monkeypatch.setenv("VICA_ANSWER_LOCAL", "on")
    _ollama_provider(monkeypatch, calls)
    parser.parse_intent("그냥", DESTS, question=q("confirm"))
    _, schema, messages = calls[-1]
    assert "answer" in schema.model_fields and af.question_line(q("confirm")) in _sys(messages)


def test_local_off_fallback_from_openai_uses_plain_local_call(monkeypatch, backends):
    calls, cfg = backends
    monkeypatch.setattr(parser, "PROVIDER", "openai")
    cfg["cloud_fail"] = RuntimeError("cloud down")
    out = parser.parse_intent("오 분", DESTS, question=q("minutes", "몇 분 기다릴까요?"))
    assert [c[0] for c in calls] == ["cloud", "local"]
    assert "로봇이 물은 질문" in _sys(calls[0][2]) and "answer" in calls[0][1].model_fields   # 외부 API 는 그대로
    _, schema, messages = calls[1]
    assert schema is parser._IntentDraft and "로봇이 물은 질문" not in _sys(messages)
    assert (out.answer, out.answer_value, out.question_id) == ("minutes", "5", "q1")


def test_local_on_fallback_from_openai_keeps_question_line(monkeypatch, backends):
    calls, cfg = backends
    monkeypatch.setenv("VICA_ANSWER_LOCAL", "on")
    monkeypatch.setattr(parser, "PROVIDER", "openai")
    cfg["cloud_fail"] = RuntimeError("cloud down")
    parser.parse_intent("그냥", DESTS, question=q("confirm"))
    assert [c[0] for c in calls] == ["cloud", "local"]
    assert af.question_line(q("confirm")) in _sys(calls[1][2]) and "answer" in calls[1][1].model_fields


def test_local_off_already_local_state_uses_plain_call(monkeypatch, backends):
    calls, _ = backends
    monkeypatch.setattr(parser, "PROVIDER", "openai")
    parser.get_backend_manager().go_local("Realtime 연결 오류")
    parser.parse_intent("그냥", DESTS, question=q("confirm"))
    assert [c[0] for c in calls] == ["local"]
    assert calls[0][1] is parser._IntentDraft and "로봇이 물은 질문" not in _sys(calls[0][2])


def test_local_off_cloud_answer_unchanged(monkeypatch, backends):
    calls, _ = backends
    monkeypatch.setattr(parser, "PROVIDER", "openai")
    parser.parse_intent("그냥", DESTS, question=q("confirm"))
    assert [c[0] for c in calls] == ["cloud"] and af.question_line(q("confirm")) in _sys(calls[0][2])

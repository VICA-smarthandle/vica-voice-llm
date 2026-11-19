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


def test_answer_that_is_not_text_does_not_fail_the_utterance(on):
    """외부 API 가 answer 에 null·숫자를 넣어도 발화가 통째로 실패하지 않는다(검토 M2) — 빈 값으로 읽는다."""
    from src.realtime_intent import RealtimeResult
    assert parser._AnswerDraft(intent="unknown", answer=None).answer == ""
    assert parser._AnswerDraft(intent="unknown", answer=3).answer == ""
    result = RealtimeResult(draft={"intent": "unknown", "answer": None}, heard_text="뭐라고", latency_sec=1.0)
    intent, _heard, _dt, _info = parser._deliver_realtime_result(result, DESTS, None, question=q())
    assert (intent.intent, intent.answer) == ("unknown", "")

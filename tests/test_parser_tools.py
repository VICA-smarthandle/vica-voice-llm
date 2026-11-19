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


def test_cloud_backend_is_picked_by_the_switch(monkeypatch):
    """스위치가 꺼지면 지금의 구조화 출력, openai 이고 켜지면 도구 출력 — 두 자리가 같은 조건."""
    picked = []
    monkeypatch.setattr(parser, "_get_structured_llm", lambda model, **kw: picked.append("structured") or object())
    monkeypatch.setattr(parser, "_get_tool_llm", lambda model, **kw: picked.append("tools") or object())
    monkeypatch.setattr(parser, "PROVIDER", "openai")
    monkeypatch.delenv("VICA_TOOLS_TEXT", raising=False)
    parser._get_cloud_llm("m")
    monkeypatch.setenv("VICA_TOOLS_TEXT", "on")
    parser._get_cloud_llm("m")
    monkeypatch.setattr(parser, "PROVIDER", "ollama")
    parser._get_cloud_llm("m")
    assert picked == ["structured", "tools", "structured"]


def test_tool_llm_binds_both_tools_and_requires_one(monkeypatch, tmp_path):
    """도구 둘(set_intent·reserve_facility)을 주고 반드시 하나를 부르게 한다. 응답은 해석 결과로 바뀐다."""
    import langchain_openai
    from langchain_core.runnables import RunnableLambda

    monkeypatch.setenv("VICA_RESERVATION_DIR", str(tmp_path))   # 운영 폴더의 시설 파일이 아니라 저장소 것
    bound = {}

    class FakeChat:
        def __init__(self, **kwargs):
            bound["init"] = kwargs

        def bind_tools(self, tools, **kwargs):
            bound["tools"], bound["kwargs"] = tools, kwargs
            return RunnableLambda(lambda messages: msg(("reserve_facility", {"facility": "상담실", "name": "이서연"})))

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChat)
    llm = parser._get_tool_llm("gpt-5.4-mini", timeout=6, max_retries=0)
    assert [t["function"]["name"] for t in bound["tools"]] == ["set_intent", "reserve_facility"]
    assert bound["kwargs"] == {"tool_choice": "required", "parallel_tool_calls": False}
    assert bound["init"] == {"model": "gpt-5.4-mini", "temperature": 0, "timeout": 6, "max_retries": 0}
    assert bound["tools"][1]["function"]["parameters"]["properties"]["facility"]["enum"] == ["세미나실", "상담실"]
    assert llm.invoke([]) == ReserveCall(facility="상담실", name="이서연")


def test_broken_facility_file_falls_back_to_structured_output(monkeypatch, tmp_path):
    """시설 파일이 깨져 도구 출력을 못 만들면 클라우드를 잃지 않고 지금의 구조화 출력으로 돈다."""
    (tmp_path / "facilities.yaml").write_text("facilities: [oops", encoding="utf-8")
    monkeypatch.setenv("VICA_RESERVATION_DIR", str(tmp_path))
    monkeypatch.setenv("VICA_TOOLS_TEXT", "on")
    monkeypatch.setattr(parser, "PROVIDER", "openai")
    structured = object()
    monkeypatch.setattr(parser, "_get_structured_llm", lambda model, **kw: structured)
    logged = []
    monkeypatch.setattr(parser, "_log", lambda level, text: logged.append((level, text)))
    assert parser._get_cloud_llm("gpt-5.4-mini") is structured
    assert logged and logged[0][0] == "error" and "도구 출력" in logged[0][1]


def _tool_llm_answering(monkeypatch, tmp_path, answer):
    """_get_tool_llm 을 가짜 ChatOpenAI 로 만든다. answer(messages) 가 모델 응답을 준다."""
    import langchain_openai
    from langchain_core.runnables import RunnableLambda

    class FakeChat:
        def __init__(self, **kwargs):
            pass

        def bind_tools(self, tools, **kwargs):
            return RunnableLambda(answer)

    monkeypatch.setenv("VICA_RESERVATION_DIR", str(tmp_path))
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChat)
    monkeypatch.setattr(parser, "_log", lambda level, text: None)
    return parser._get_tool_llm("gpt-5.4-mini")


def test_malformed_tool_answer_is_asked_once_more(monkeypatch, tmp_path):
    """I-4: 도구 응답의 형식 오류 한 번으로 로컬 대피까지 가지 않게, 같은 발화를 한 번만 다시 부른다."""
    answers = [msg(("go_to", {"location": "화장실"})), msg(("set_intent", {"intent": "navigate", "destination_candidate": "화장실"}))]
    llm = _tool_llm_answering(monkeypatch, tmp_path, lambda messages: answers.pop(0))
    assert llm.invoke([]).intent == "navigate" and answers == []
    answers[:] = [msg(), msg(("set_intent", {"intent": "fly"})), msg(("set_intent", {"intent": "unknown"}))]
    with pytest.raises(ValueError):
        llm.invoke([])
    assert len(answers) == 1                                  # 두 번까지만 부른다


def test_connection_failure_is_not_retried_here(monkeypatch, tmp_path):
    calls = []

    def down(messages):
        calls.append(1)
        raise ConnectionError("network down")

    llm = _tool_llm_answering(monkeypatch, tmp_path, down)
    with pytest.raises(ConnectionError):
        llm.invoke([])
    assert len(calls) == 1                                    # 연결 실패의 재시도·로컬 대피는 전환 담당 모듈 몫

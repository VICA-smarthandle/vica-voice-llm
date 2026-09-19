"""파서가 전환 담당 모듈를 거치는지, 명시 모델은 우회하는지 (LLM 없이 검증)."""
import pytest

from src import langchain_intent_parser as parser
from src.langchain_intent_parser import _IntentDraft, parse_intent
from src.llm_backend import BackendState, LlmBackendManager, ProbeResult
from src.replies import LLM_UNAVAILABLE
from src.schema import DestinationData

DEST = DestinationData(
    id="starlight_1f_restroom",
    name="별빛관 1층 화장실",
    confirm_prompt="별빛관 1층 화장실로 안내해드릴까요?",
)
DRAFT = _IntentDraft(intent="navigate", destination_candidate="별빛관 1층 화장실")


class Boom(Exception):
    pass


def _manager(cloud_fail: bool, local: bool = True) -> LlmBackendManager:
    def cloud(messages):
        if cloud_fail:
            raise Boom("cloud down")
        return DRAFT

    return LlmBackendManager(cloud, (lambda m: DRAFT) if local else None,
                             lambda: ProbeResult.DEAD, logger=lambda *_: None)


@pytest.fixture(autouse=True)
def _reset():
    parser.reset_backend_manager()
    yield
    parser.reset_backend_manager()


def test_parse_intent_uses_manager_and_falls_back(monkeypatch):
    mgr = _manager(cloud_fail=True)
    monkeypatch.setattr(parser, "get_backend_manager", lambda: mgr)
    intent = parse_intent("화장실로 안내해줘", [DEST])
    assert intent.intent == "navigate"
    assert intent.matched_destination_id == "starlight_1f_restroom"
    assert mgr.state is BackendState.LOCAL


def test_parse_intent_unavailable_when_both_fail(monkeypatch):
    mgr = _manager(cloud_fail=True, local=False)
    monkeypatch.setattr(parser, "get_backend_manager", lambda: mgr)
    intent = parse_intent("화장실로 안내해줘", [DEST])
    assert intent.intent == "unknown"
    assert intent.reply == LLM_UNAVAILABLE


def test_explicit_model_bypasses_manager(monkeypatch):
    called = []

    class Direct:
        def invoke(self, messages):
            called.append("direct")
            return DRAFT

    monkeypatch.setattr(parser, "_get_structured_llm", lambda model, **kw: Direct())
    monkeypatch.setattr(parser, "get_backend_manager",
                        lambda: (_ for _ in ()).throw(AssertionError("관리자를 부르면 안 된다")))
    intent = parse_intent("화장실로 안내해줘", [DEST], model="gemma4-e2b-text")
    assert called == ["direct"]
    assert intent.intent == "navigate"


def test_manager_is_built_once(monkeypatch):
    built = []
    calls = []

    class Direct:
        def invoke(self, messages):
            return DRAFT

    def stub(model, **kw):
        built.append(model)
        calls.append(kw)
        return Direct()

    monkeypatch.setattr(parser, "_get_structured_llm", stub)
    monkeypatch.setattr(parser, "FALLBACK_MODEL", "")
    a = parser.get_backend_manager()
    b = parser.get_backend_manager()
    assert a is b
    assert built == [parser.DEFAULT_MODEL]
    assert a.has_local is False  # 폴백 모델이 비면 로컬 없음 = 옛 동작
    assert calls[0]["timeout"] == 15
    assert calls[0]["max_retries"] == 1


def test_manager_has_local_when_fallback_set(monkeypatch):
    calls = []

    class Direct:
        def invoke(self, messages):
            return DRAFT

    def stub(model, **kw):
        calls.append(kw)
        return Direct()

    monkeypatch.setattr(parser, "_get_structured_llm", stub)
    monkeypatch.setattr(parser, "FALLBACK_MODEL", "gemma4-e2b-text")
    monkeypatch.setattr(parser, "_build_local_structured", lambda: Direct())
    mgr = parser.get_backend_manager()
    assert mgr.has_local is True
    assert mgr.state is BackendState.CLOUD
    assert calls[0]["timeout"] == parser.CLOUD_TIMEOUT_SEC
    assert calls[0]["max_retries"] == 0


def test_manager_falls_back_to_cloud_only_when_local_build_fails(monkeypatch):
    calls = []

    class Direct:
        def invoke(self, messages):
            return DRAFT

    def stub(model, **kw):
        calls.append(kw)
        return Direct()

    def boom_local():
        raise RuntimeError("no ollama")

    monkeypatch.setattr(parser, "FALLBACK_MODEL", "gemma4-e2b-text")
    monkeypatch.setattr(parser, "_build_local_structured", boom_local)
    monkeypatch.setattr(parser, "_get_structured_llm", stub)
    mgr = parser.get_backend_manager()
    assert mgr.has_local is False
    assert len(calls) == 1
    assert calls[0]["timeout"] == 15
    assert calls[0]["max_retries"] == 1
    assert parser.get_backend_manager() is mgr


class _RecordingStructured:
    def invoke(self, messages):
        return DRAFT


class _RecordingChatOllama:
    """ChatOllama 대역. 생성 인자를 기록하고 with_structured_output 은 그대로 통과."""

    calls: list = []

    def __init__(self, **kwargs):
        _RecordingChatOllama.calls.append(kwargs)

    def with_structured_output(self, schema):
        return _RecordingStructured()


def test_build_local_structured_sets_local_timeout(monkeypatch):
    """F1: 로컬 호출에 timeout 이 없으면 Ollama 가 멈출 때 영원히 기다린다."""
    _RecordingChatOllama.calls = []
    monkeypatch.setattr(parser, "ChatOllama", _RecordingChatOllama)
    monkeypatch.setattr(parser, "FALLBACK_MODEL", "gemma4-e2b-text")
    parser._build_local_structured()
    assert _RecordingChatOllama.calls[0]["client_kwargs"] == {"timeout": parser.LOCAL_TIMEOUT_SEC}
    assert _RecordingChatOllama.calls[0]["reasoning"] is False
    assert _RecordingChatOllama.calls[0]["keep_alive"] == -1
    assert _RecordingChatOllama.calls[0]["temperature"] == 0


def test_ollama_branch_applies_cloud_timeout(monkeypatch):
    """F3: ollama 분기도 openai 처럼 timeout 이 걸려야 클라우드가 멈춰도 무한 대기하지 않는다."""
    _RecordingChatOllama.calls = []
    monkeypatch.setattr(parser, "PROVIDER", "ollama")
    monkeypatch.setattr(parser, "ChatOllama", _RecordingChatOllama)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    parser._get_structured_llm("gemma4:cloud", timeout=6, max_retries=0)
    assert _RecordingChatOllama.calls[0]["client_kwargs"] == {"timeout": 6}

    monkeypatch.setenv("OLLAMA_API_KEY", "k")
    parser._get_structured_llm("gemma4:cloud", timeout=6, max_retries=0)
    assert _RecordingChatOllama.calls[1]["client_kwargs"] == {
        "timeout": 6,
        "headers": {"Authorization": "Bearer k"},
    }

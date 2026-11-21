"""출력 스키마 — not_for_robot·dismiss 값과 clarify 상황 칸(원칙 4·9·10·11, VICA_RECUT=on 일 때만)."""
import json

from src import langchain_intent_parser as parser
from src.realtime_intent import build_intent_tool


def test_p9_off_schema_is_unchanged():
    assert parser._draft_cls() is parser._IntentDraft
    assert "not_for_robot" not in json.dumps(parser._IntentDraft.model_json_schema())
    assert "clarify_kind" not in json.dumps(build_intent_tool())


def test_p9_on_schema_offers_new_values_and_fields(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    props = parser._draft_cls().model_json_schema()["properties"]
    assert {"clarify_kind", "clarify_options", "clarify_wanted"} <= set(props)
    assert props["clarify_kind"]["enum"] == ["", "several", "not_available", "no_destination"]
    assert "not_for_robot" in json.dumps(props["intent"])
    tool = build_intent_tool()["parameters"]["properties"]
    assert {"not_for_robot", "dismiss"} <= set(tool["intent"]["enum"])
    assert tool["clarify_kind"]["enum"] == ["", "several", "not_available", "no_destination"]
    assert build_intent_tool()["parameters"]["required"] == ["intent", "heard_text"]


def test_p4_recut_draft_tolerates_bad_values(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    d = parser._draft_cls()(intent="clarify", clarify_kind=None, clarify_options="남자 화장실", clarify_wanted=3)
    assert (d.clarify_kind, d.clarify_options, d.clarify_wanted) == ("", [], "")
    assert parser._draft_cls()(intent="clarify", clarify_kind="엉뚱").clarify_kind == "엉뚱"
    assert parser._draft_cls()(intent="dismiss", clarify_options=["a", 1, "b"]).clarify_options == ["a", "b"]


def test_p9_both_switches_give_both_sets_of_fields(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    monkeypatch.setenv("VICA_ANSWER", "on")
    assert {"answer", "clarify_kind"} <= set(parser._draft_cls().model_json_schema()["properties"])
    assert {"answer", "clarify_kind"} <= set(build_intent_tool()["parameters"]["properties"])


def test_p9_draft_cls_is_stable_per_combination(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    assert parser._draft_cls() is parser._draft_cls()
    assert parser._base_draft_cls() is parser._RecutDraft
    monkeypatch.setenv("VICA_ANSWER", "on")
    assert parser._draft_cls() is parser._RecutAnswerDraft


def test_p9_intent_accepts_new_values_regardless_of_switch():
    from src.schema import VicaIntent
    assert VicaIntent(intent="not_for_robot").intent == "not_for_robot"
    assert VicaIntent(intent="dismiss").intent == "dismiss"
    import pytest
    with pytest.raises(Exception):
        parser._IntentDraft(intent="dismiss")


class _FakeOllama:
    def __init__(self):
        self.schemas = []

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        return type("R", (), {"invoke": staticmethod(lambda messages: schema)})()


def _local_schema(kind=""):
    runnable = parser._ollama_structured(_FakeOllama())
    token = parser._ANSWER_KIND.set(kind)
    try:
        return runnable.invoke([])
    finally:
        parser._ANSWER_KIND.reset(token)


def test_p9_local_schema_follows_recut_switch(monkeypatch):
    # 꺼짐: 로컬이 받는 클래스에 새 칸이 없다
    llm = _FakeOllama()
    parser._ollama_structured(llm)
    assert llm.schemas == [parser._IntentDraft]
    assert "clarify_kind" not in llm.schemas[0].model_fields
    # 재단만 켬
    monkeypatch.setenv("VICA_RECUT", "on")
    llm = _FakeOllama()
    parser._ollama_structured(llm)
    assert "clarify_kind" in llm.schemas[0].model_fields
    # 둘 다 켬: narrowed 바탕이 재단 클래스 — 새 칸과 answer 가 모두 있다
    monkeypatch.setenv("VICA_ANSWER", "on")
    picked = _local_schema("minutes")
    assert {"clarify_kind", "answer"} <= set(picked.model_fields)
    assert _local_schema("") is parser._RecutDraft
    from src import answer_field
    n = answer_field.narrowed(parser._RecutDraft, "x")
    assert {"clarify_kind", "answer"} <= set(n.model_fields)

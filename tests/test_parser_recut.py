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


# ---- Task 2.4 뒤처리 -------------------------------------------------------------------------------------------
from src.replies import ASK_DESTINATION, RETRY_PROMPT
from src.schema import DestinationData

_D = [
    DestinationData(id="r407", name="407호", confirm_prompt="407호로 안내해드릴까요?"),
    DestinationData(id="r408", name="비공개 408호", is_approachable=False, unavailable_reason="들어갈 수 없는 곳이에요."),
    DestinationData(id="men", name="남자 화장실", aliases=["화장실"], confirm_prompt="남자 화장실로 안내해드릴까요?"),
    DestinationData(id="women", name="여자 화장실", aliases=["화장실"], confirm_prompt="여자 화장실로 안내해드릴까요?"),
]


def draft(**kw):
    return parser._draft_cls()(**kw)


def finalize(d, user_text=""):
    return parser._finalize(d, _D, user_text=user_text)


def audio(d, heard=""):
    return parser.deliver_audio_draft(d, _D, heard=heard)


def test_p7_not_for_robot_and_dismiss_come_back_bare(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    for kind in ("not_for_robot", "dismiss"):
        for run in (finalize, audio):
            out = run(draft(intent=kind, reply="아무 말", destination_candidate="화장실", confidence=0.7))
            assert (out.intent, out.reply, out.need_confirm, out.matched_destination_id) == (kind, "", False, None)
            assert out.destination_candidate is None and out.confidence == 0.7


def test_p4_several_uses_the_names_the_model_listed(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    d = draft(intent="clarify", clarify_kind="several", clarify_options=["남자 화장실", "여자 화장실", "없는 곳"], reply="화장실요?")
    for run in (finalize, audio):
        assert run(d).reply == "남자 화장실과 여자 화장실 중 어디로 갈까요?"


def test_p11_place_not_in_list(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    a = finalize(draft(intent="clarify", clarify_kind="not_available", clarify_wanted="스무디 파는 곳", reply="목록 중 어디로 갈까요?"))
    b = finalize(draft(intent="navigate", destination_candidate="스무디 가게"))
    c = audio(draft(intent="navigate", destination_candidate="스무디 가게"))
    assert a.reply == "제가 아는 목적지 중에서 스무디 파는 곳은 없는데, 어디로 안내해드릴까요?"
    assert b.intent == "clarify" and b.reply == "제가 아는 목적지 중에서 스무디 가게는 없는데, 어디로 안내해드릴까요?"
    assert c.intent == "clarify" and c.reply == b.reply


def test_p4_no_destination_asks_where(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    assert finalize(draft(intent="clarify", clarify_kind="no_destination", reply="네?")).reply == ASK_DESTINATION


def test_p18_private_place_is_not_listed_as_an_option(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    out = finalize(draft(intent="clarify", clarify_kind="several", clarify_options=["407호", "비공개 408호"]))
    assert out.reply == ASK_DESTINATION


def test_p8_unknown_asks_again_on_both_paths_but_wake_stays(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    assert audio(draft(intent="unknown", reply=""), heard="으어").reply == RETRY_PROMPT
    assert audio(draft(intent="unknown", reply="네, 말씀하세요."), heard="비카야").reply == "네?"
    assert audio(draft(intent="unknown", reply="네?"), heard="으어").reply == "네?"
    assert finalize(draft(intent="unknown", reply="제가 잘 모르겠어요")).reply == RETRY_PROMPT


def test_p8_other_intents_are_untouched(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    assert finalize(draft(intent="question", reply="여기는 1층입니다.")).reply == "여기는 1층입니다."
    assert settle_noop()


def settle_noop():
    from src import intent_recut
    from src.schema import VicaIntent
    r = VicaIntent(intent="unknown", reply="x", safety_flag="emergency")
    return intent_recut.settle(r, draft(intent="unknown"), _D, "") is r


def test_off_keeps_the_models_own_clarify_sentence(monkeypatch):
    monkeypatch.delenv("VICA_RECUT", raising=False)
    assert finalize(parser._IntentDraft(intent="clarify", reply="어느 방으로 갈까요?")).reply == "어느 방으로 갈까요?"
    assert audio(parser._IntentDraft(intent="unknown", reply="")).reply == ""
    from src import intent_recut
    assert intent_recut.early(parser._IntentDraft(intent="question")) is None


# ---- Task 2.4 수정 1차: 코드가 만든 고르기 문장 보존·LLM 이 지은 고르기 꼴은 틀에서 다시 -------------------------------
CHOOSE_TOILETS = "남자 화장실과 여자 화장실 중 어디로 갈까요?"


def test_p4_code_made_choice_sentence_is_kept(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    out = finalize(draft(intent="navigate", destination_candidate="화장실"))     # _ask_which 가 만든다
    assert out.intent == "clarify" and out.reply == CHOOSE_TOILETS


def test_p4_code_made_choice_survives_a_thin_options_list(monkeypatch):
    # 모델이 several 이라 했지만 쓸 수 있는 이름이 하나뿐 — 같은 발화에 코드가 만든 후보 질문을 덮지 않는다
    monkeypatch.setenv("VICA_RECUT", "on")
    d = dict(intent="navigate", destination_candidate="화장실", clarify_kind="several", clarify_options=["화장실"])
    assert finalize(draft(**d)).reply == CHOOSE_TOILETS
    assert audio(draft(**d)).reply == CHOOSE_TOILETS


def test_p12_p18_models_own_choice_shape_goes_through_the_frame(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    d = draft(intent="clarify", clarify_kind="", reply="407호와 비공개 408호 중 어디로 갈까요?")
    assert finalize(d).reply == ASK_DESTINATION


def test_p8_llm_unavailable_sentence_is_not_overwritten(monkeypatch):
    from src import intent_recut
    from src.replies import LLM_UNAVAILABLE
    from src.schema import VicaIntent
    monkeypatch.setenv("VICA_RECUT", "on")
    r = VicaIntent(intent="unknown", reply=LLM_UNAVAILABLE)
    assert intent_recut.settle(r, draft(intent="unknown"), _D, "비카야").reply == LLM_UNAVAILABLE

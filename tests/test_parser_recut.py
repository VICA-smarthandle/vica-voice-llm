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
    assert parser._base_draft_cls() is parser._IntentDraft  # 원칙 16(2.8): 로컬 바탕은 재단과 무관하게 _IntentDraft
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


def test_p9_local_schema_ignores_recut_switch(monkeypatch):
    # 꺼짐: 로컬이 받는 클래스에 새 칸이 없다
    llm = _FakeOllama()
    parser._ollama_structured(llm)
    assert llm.schemas == [parser._IntentDraft]
    assert "clarify_kind" not in llm.schemas[0].model_fields
    # 재단만 켬 — 원칙 16(2.8): 로컬 스키마에는 재단 칸이 없다
    monkeypatch.setenv("VICA_RECUT", "on")
    llm = _FakeOllama()
    parser._ollama_structured(llm)
    assert llm.schemas == [parser._IntentDraft]
    # 둘 다 켬: narrowed 바탕도 _IntentDraft — answer 만 있고 재단 칸은 없다
    monkeypatch.setenv("VICA_ANSWER", "on")
    monkeypatch.setenv("VICA_ANSWER_LOCAL", "on")    # 3.5 — 로컬 대답 칸이 켜졌을 때의 좁힌 스키마
    picked = _local_schema("minutes")
    assert "answer" in picked.model_fields and "clarify_kind" not in picked.model_fields
    assert _local_schema("") is parser._IntentDraft
    # 클라우드 쪽 바탕 클래스는 그대로 재단 클래스다
    assert parser._draft_cls() is parser._RecutAnswerDraft


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


# ---- Task 2.4 수정 2차: 갈 수 없는 곳이 든 고르기 문장은 코드가 만든 것이어도 그대로 두지 않는다(원칙 18) ----------------
def test_p18_audio_choice_naming_a_private_place_is_rebuilt(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    out = audio(draft(intent="clarify", clarify_kind="", reply="407호나 비공개 408호 중에 어디로 갈까요?"))
    assert "비공개 408호" not in out.reply and out.reply == ASK_DESTINATION


def test_p4_audio_choice_between_approachable_places_is_kept(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    out = audio(draft(intent="clarify", reply="남자 화장실이나 여자 화장실 중에 어디로 갈까요?"))   # 코드가 정해진 문장으로 바꾼다
    assert out.reply == CHOOSE_TOILETS


def test_off_audio_choice_naming_a_private_place_is_as_before(monkeypatch):
    monkeypatch.delenv("VICA_RECUT", raising=False)
    out = audio(parser._IntentDraft(intent="clarify", reply="407호나 비공개 408호 중에 어디로 갈까요?"))
    assert out.reply == "407호와 비공개 408호 중 어디로 갈까요?"


# ---- Task 2.4 수정 3차: 모델이 options 를 못 채워도 문장에 이름을 제대로 적었으면 고르기 틀로 -----------------------------
def test_p4_choice_sentence_without_options_is_rebuilt_from_listed_names(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    for reply in ("남자 화장실과 여자 화장실 중 어디로 갈까요?", "남자 화장실이나 여자 화장실 중에 어디로 갈까요?"):
        out = finalize(draft(intent="clarify", clarify_kind="", clarify_options=[], reply=reply))
        assert out.reply == CHOOSE_TOILETS


# ---- Task 2.8 로컬(Ollama)이 답하는 호출에는 재단을 싣지 않는다 (원칙 16, 2단계 측정 P3) ------------------------------
import pytest

from src.llm_backend import ProbeResult
from src.schema import DestinationData as _Dest

_TOILETS = [_Dest(id="men", name="남자 화장실", confirm_prompt="남자 화장실로 안내해드릴까요?")]
_ASKED = "어느 방으로 갈까요?"


class _Recorder:
    """호출마다 (출력 클래스, messages)를 적고 draft 를 돌려주는 가짜 LLM 하나."""

    def __init__(self, calls, who, fail=None):
        self.calls, self.who, self.fail = calls, who, fail

    def with_structured_output(self, schema, **kw):
        outer = self
        return type("R", (), {"invoke": lambda s, messages: outer._answer(schema, messages)})()

    invoke = None

    def _answer(self, schema, messages):
        self.calls.append((self.who, schema, messages))
        if self.fail:
            raise self.fail
        kw = {"intent": "clarify", "reply": _ASKED}
        if "clarify_kind" in schema.model_fields:
            kw["clarify_kind"] = "no_destination"
        return schema(**kw)


@pytest.fixture
def backends(monkeypatch):
    """진짜 LlmBackendManager + 가짜 cloud(OpenAI 자리)·가짜 Ollama(ChatOllama 자리). calls 에 누가 불렸는지 쌓인다."""
    parser.reset_backend_manager()
    calls = []
    cfg = {"cloud_fail": None}

    class FakeChat:
        def __init__(self, **kw):
            pass

        def with_structured_output(self, schema, **kw):
            return _Recorder(calls, "local").with_structured_output(schema)

    def cloud_llm(model, **kw):
        rec = _Recorder(calls, "cloud", cfg["cloud_fail"])
        return type("C", (), {"invoke": lambda s, m: rec._answer(parser._draft_cls(), m)})()

    monkeypatch.setattr(parser, "ChatOllama", FakeChat)
    monkeypatch.setattr(parser, "FALLBACK_MODEL", "midm2-mini")
    monkeypatch.setattr(parser, "_get_cloud_llm", cloud_llm)
    monkeypatch.setattr(parser, "_cloud_probe", lambda: ProbeResult.DEAD)
    monkeypatch.delenv("VICA_ANSWER", raising=False)
    yield calls, cfg
    parser.reset_backend_manager()


def _ask(text="그냥"):
    return parser.parse_intent(text, _TOILETS)


def _system(messages):
    return messages[0].content


def test_p16_ollama_provider_answers_as_if_recut_were_off(monkeypatch, backends):
    calls, _ = backends
    monkeypatch.setenv("VICA_RECUT", "on")
    monkeypatch.setattr(parser, "PROVIDER", "ollama")
    # 공급자가 ollama 면 "클라우드 자리"도 _ollama_structured 가 만든 호출체다
    monkeypatch.setattr(parser, "_get_cloud_llm", lambda model, **kw: parser._ollama_structured(
        type("O", (), {"with_structured_output": lambda s, schema, **k: _Recorder(calls, "ollama-cloud").with_structured_output(schema)})()))
    parser.reset_backend_manager()
    out = _ask()
    who, schema, messages = calls[-1]
    assert who == "ollama-cloud" and schema is parser._IntentDraft
    assert "not_for_robot" not in _system(messages) and "[판단 순서]" not in _system(messages)
    assert out.reply == _ASKED


def test_p16_fallback_to_local_answers_as_if_recut_were_off(monkeypatch, backends):
    calls, cfg = backends
    monkeypatch.setenv("VICA_RECUT", "on")
    monkeypatch.setattr(parser, "PROVIDER", "openai")
    cfg["cloud_fail"] = RuntimeError("cloud down")
    out = _ask()
    assert [c[0] for c in calls] == ["cloud", "local"]
    assert "[판단 순서]" in _system(calls[0][2])
    who, schema, messages = calls[1]
    assert schema is parser._IntentDraft and "clarify_kind" not in schema.model_fields
    assert "not_for_robot" not in _system(messages) and "[판단 순서]" not in _system(messages)
    assert out.reply == _ASKED


def test_p16_already_local_answers_as_if_recut_were_off(monkeypatch, backends):
    calls, _ = backends
    monkeypatch.setenv("VICA_RECUT", "on")
    monkeypatch.setattr(parser, "PROVIDER", "openai")
    parser.get_backend_manager().go_local("Realtime 연결 오류")
    out = _ask()
    assert [c[0] for c in calls] == ["local"]
    assert "[판단 순서]" not in _system(calls[0][2]) and out.reply == _ASKED


def test_p16_cloud_answer_still_uses_recut(monkeypatch, backends):
    calls, _ = backends
    monkeypatch.setenv("VICA_RECUT", "on")
    monkeypatch.setattr(parser, "PROVIDER", "openai")
    out = _ask()
    assert [c[0] for c in calls] == ["cloud"]
    assert "[판단 순서]" in _system(calls[0][2])
    assert out.reply == ASK_DESTINATION


@pytest.mark.parametrize("cloud_fail", [None, RuntimeError("cloud down")])
def test_p16_off_is_unchanged_for_both_backends(monkeypatch, backends, cloud_fail):
    calls, cfg = backends
    monkeypatch.delenv("VICA_RECUT", raising=False)
    monkeypatch.setattr(parser, "PROVIDER", "openai")
    cfg["cloud_fail"] = cloud_fail
    out = _ask()
    assert out.reply == _ASKED
    for _, schema, messages in calls:
        assert "clarify_kind" not in schema.model_fields and "[판단 순서]" not in _system(messages)
    # 꺼졌을 때 로컬에 가는 지시문은 cloud 에 간 것과 같다(local_messages 를 따로 만들지 않는다)
    if cloud_fail:
        assert _system(calls[1][2]) == _system(calls[0][2])


def test_p16_prompt_off_text_is_identical_with_explicit_recut_false(monkeypatch):
    monkeypatch.delenv("VICA_RECUT", raising=False)
    plain = parser._build_system_prompt(_TOILETS, local=True)
    monkeypatch.setenv("VICA_RECUT", "on")
    assert parser._build_system_prompt(_TOILETS, local=True, recut=False) == plain
    assert parser._build_system_prompt(_TOILETS, local=True) != plain

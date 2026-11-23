"""정해진 대답 둘 — "다시 말해줘"(repeat)·"어디 갈 수 있어?"(places) (설계 6.4, 원칙 28·29, Task 4.2)."""
from __future__ import annotations

import importlib
import json
import os
import sys
import types

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src import langchain_intent_parser as parser
from src.question_record import OpenQuestion
from src.realtime_intent import RealtimeResult, build_intent_tool
from src.replies import RETRY_PROMPT
from src.schema import DestinationData, RobotState, VicaIntent

LINE_REPEAT = ('- repeat: 로봇이 방금 한 말을 다시 해 달라고 한다("다시 말해줘", "뭐라고?", "못 들었어", "방금 뭐라고 했어?"). reply 는 빈 문자열 — 다시 할 말은 시스템이 고른다(직접 옮겨 말하지 않는다).')
LINE_PLACES = ('- places: 어디로 갈 수 있는지 묻는다("어디 갈 수 있어?", "갈 수 있는 곳 알려줘"). 이 물음은 question 으로 답하지 않고 반드시 places 다 — reply 는 빈 문자열, 문장은 시스템이 만든다(목적지 이름을 직접 늘어놓거나 "알려드릴게요"라고 하지 않는다). 무엇을 할 수 있는지 물으면 places 가 아니라 question 이고 about 은 abilities 다.')

TOILET = DestinationData(id="wc", name="화장실", confirm_prompt="화장실로 안내해드릴까요?")
SOCIAL = DestinationData(id="social", name="사회복지", confirm_prompt="사회복지로 안내해드릴까요?")
PRIVATE = DestinationData(id="r408", name="비공개 408호", is_approachable=False, unavailable_reason="들어갈 수 없는 곳이에요.")
R407 = DestinationData(id="r407", name="407호", confirm_prompt="407호로 안내해드릴까요?")
FOOD = DestinationData(id="food", name="식당", confirm_prompt="식당으로 안내해드릴까요?")
DESTS = [TOILET, SOCIAL, PRIVATE, R407]
DESTS_WITH_PRIVATE = DESTS
CLOSED = [PRIVATE, DestinationData(id="x", name="공사 중인 방", is_approachable=False)]


def fa():
    """시험 대상 모듈 — 아직 없을 때(RED) 이 함수를 부르는 시험만 실패하게 늦게 가져온다."""
    return importlib.import_module("src.fixed_answers")


def talk_on(monkeypatch):
    """4단계는 2·3단계 위에서만 켜진다(수정 1차 규칙 0) — 재단·대답 칸·정해진 대답 셋을 함께 켠다."""
    monkeypatch.setenv("VICA_RECUT", "on")
    monkeypatch.setenv("VICA_ANSWER", "on")
    monkeypatch.setenv("VICA_TALK", "on")


def test_p29_places_sentence_names_at_most_three():
    from src.replies import places_sentence
    assert places_sentence(["화장실", "사회복지", "407호"]) == "화장실, 사회복지, 407호로 안내할 수 있습니다."
    assert places_sentence(["화장실", "사회복지", "407호", "식당"]) == "화장실, 사회복지, 407호 등으로 안내할 수 있습니다."
    assert places_sentence(["화장실", "식당"]) == "화장실, 식당으로 안내할 수 있습니다."
    assert places_sentence(["화장실"]) == "화장실로 안내할 수 있습니다."
    assert places_sentence([]) == ""


def test_p18_places_never_names_a_place_the_robot_cannot_go():
    names = fa().places_names(DESTS_WITH_PRIVATE)
    assert "비공개 408호" not in names
    assert names == ["화장실", "사회복지", "407호"]
    assert fa().places_names(CLOSED) == []


def test_p28_repeat_says_the_open_question_or_the_last_sentence():
    last_said, settle_repeat = fa().last_said, fa().settle_repeat
    recent = [(1.0, "화장실에 도착했습니다."), (2.0, "네?")]
    assert last_said(recent) == "화장실에 도착했습니다."
    asked = OpenQuestion("q", "confirm", "화장실로 안내해드릴까요?", ("t",), 0.0)
    assert settle_repeat(asked, recent).intent == "repeat"
    wake = OpenQuestion("q", "open", "네?", (), 0.0)
    out = settle_repeat(wake, recent)
    assert (out.intent, out.reply) == ("question", "화장실에 도착했습니다.")
    assert settle_repeat(None, []).intent == "unknown"


NOT_AVAILABLE_ASK = "제가 아는 목적지 중에서 스무디 가게는 없는데, 어디로 안내해드릴까요?"


def _bare(out):
    return (out.intent, out.reply, out.ask_text, out.ask_kind, tuple(out.ask_option_ids), out.answer, out.question_id)


def test_p28_rule1_no_open_question_last_sentence_decides():
    settle_repeat = fa().settle_repeat
    out = settle_repeat(None, [(1.0, "화장실 안내를 시작합니다.")])
    assert (out.intent, out.reply, out.ask_text) == ("question", "화장실 안내를 시작합니다.", "")
    assert _bare(settle_repeat(None, [(1.0, "다시 출발할까요?")])) == ("repeat", "", "다시 출발할까요?", "open", (), "", "")
    assert _bare(settle_repeat(None, [(1.0, NOT_AVAILABLE_ASK)])) == ("repeat", "", NOT_AVAILABLE_ASK, "open", (), "", "")
    out = settle_repeat(None, [(1.0, "네?"), (2.0, " 네? ")])
    assert (out.intent, out.reply) == ("unknown", RETRY_PROMPT)


def test_p28_rule2_after_wake_greeting_only_the_last_sentence_counts():
    settle_repeat, wake = fa().settle_repeat, OpenQuestion("q0", "open", "네?", (), 0.0)
    recent = [(1.0, "화장실로 안내해드릴까요?"), (2.0, "네?")]
    assert _bare(settle_repeat(wake, recent)) == ("repeat", "", "화장실로 안내해드릴까요?", "open", (), "", "")
    recent = [(1.0, "다시 출발할까요?"), (2.0, "화장실 안내를 시작합니다."), (3.0, "네?")]
    out = settle_repeat(wake, recent)
    assert (out.intent, out.reply, out.ask_text) == ("question", "화장실 안내를 시작합니다.", "")
    assert _bare(settle_repeat(wake, [(2.0, "네?")])) == ("repeat", "", "", "", (), "", "")


def test_p28_n2_a_refused_question_is_not_asked_again():
    wake = OpenQuestion("q0", "open", "네?", (), 0.0)
    recent = [(1.0, "화장실로 안내해드릴까요?"), (4.0, "안내 요청이 취소되었습니다."), (8.0, "네?")]
    out = fa().settle_repeat(wake, recent)
    assert (out.intent, out.reply, out.ask_text) == ("question", "안내 요청이 취소되었습니다.", "")


@pytest.mark.parametrize("kind,text,ids", [
    ("confirm", "화장실로 안내해드릴까요?", ("wc",)),
    ("yesno", "다시 출발할까요?", ()),
    ("yesno_inverted", "안내가 필요 없으신가요?", ()),
    ("minutes", "몇 분쯤 걸리실까요?", ()),
    ("open", NOT_AVAILABLE_ASK, ()),
    ("open", "네, 어디로 모실까요?", ()),
])
def test_p28_rule3_any_other_open_question_goes_to_the_mission_with_its_text(kind, text, ids):
    q = OpenQuestion("q1", kind, text, ids, 0.0)
    assert _bare(fa().settle_repeat(q, [(1.0, "엉뚱한 말이에요.")])) == ("repeat", "", text, kind, ids, "", "")


def test_p28_rule3_handed_choice_is_handed_again_by_the_voice():
    q = OpenQuestion("q2", "choice", "화장실과 사회복지 중 어디로 갈까요?", ("wc", "social"), 0.0)
    out = fa().settle_repeat(q, [(1.0, "화장실과 사회복지 중 어디로 갈까요?"), (2.0, "엉뚱한 말이에요.")])
    assert (out.intent, out.reply, out.ask_text) == ("question", q.text, "")


def test_p28_last_said_skips_the_wait_beacon():
    from src.mission_phrases import WAIT_BEACON
    assert fa().last_said([(1.0, "화장실에 도착했습니다."), (2.0, WAIT_BEACON), (3.0, "네?")]) == "화장실에 도착했습니다."
    out = fa().settle_repeat(None, [(1.0, WAIT_BEACON)])
    assert out.intent == "unknown"


def test_off_schema_and_prompts_have_no_repeat_or_places():
    assert "places" not in parser._build_system_prompt(DESTS) and "places" not in json.dumps(build_intent_tool())


def test_conftest_keeps_talk_off():
    assert os.environ.get("VICA_TALK") == "off"


def test_p28_switch_needs_recut_and_answer(monkeypatch):
    monkeypatch.setenv("VICA_TALK", "on")
    for recut, answer in (("off", "off"), ("off", "on"), ("on", "off")):
        monkeypatch.setenv("VICA_RECUT", recut)
        monkeypatch.setenv("VICA_ANSWER", answer)
        assert fa().on() is False
    monkeypatch.setenv("VICA_RECUT", "on")
    monkeypatch.setenv("VICA_ANSWER", "on")
    assert fa().on() is True
    monkeypatch.setenv("VICA_TALK", "off")
    assert fa().on() is False


BUNDLE_NOTE = "다섯 개(VICA_RECUT·VICA_ANSWER + 미션 answer_field·reask_ladder·wake_revive)는 한 묶음으로 켠다"


@pytest.mark.parametrize("recut,answer,talk,off_keys", [
    ("off", "off", "off", None), ("on", "off", "off", None), ("on", "on", "off", None), ("on", "on", "on", None),
    ("off", "on", "off", ("VICA_ANSWER=on", "VICA_RECUT")),
    ("on", "off", "on", ("VICA_TALK=on", "VICA_ANSWER")),
    ("off", "off", "on", ("VICA_TALK=on", "VICA_RECUT", "VICA_ANSWER")),
    ("off", "on", "on", ("VICA_ANSWER=on", "VICA_TALK=on", "VICA_RECUT")),
])
def test_i4_switch_warning_names_the_mismatch(monkeypatch, recut, answer, talk, off_keys):
    """최종 검토 I4: 음성 스위치 조합이 어긋나면 기동 때 경고 한 줄(동작은 막지 않는다) — VICA_ANSWER=on 인데 VICA_RECUT 꺼짐 /"""
    monkeypatch.setenv("VICA_RECUT", recut)
    monkeypatch.setenv("VICA_ANSWER", answer)
    monkeypatch.setenv("VICA_TALK", talk)
    line = fa().switch_warning()
    if off_keys is None:
        assert line == ""
        return
    assert "\n" not in line and BUNDLE_NOTE in line
    for key in off_keys:
        assert key in line, key


def test_i4_switch_warning_ignores_the_local_answer_switch(monkeypatch):
    monkeypatch.setenv("VICA_ANSWER_LOCAL", "on")
    assert fa().switch_warning() == ""


def test_i4_node_logs_the_switch_warning_once_at_start():
    """의도 노드는 기동(__init__) 때 그 글을 경고 로그로만 낸다 — 빈 글자면 아무것도 내지 않는다."""
    import ast
    from tests.test_one_voice_wiring import _branch_tests, _calls, _method
    init = _method("__init__")
    (made,) = _calls(init, "fixed_answers.switch_warning")
    warned = [c for c in _calls(init, "self.get_logger().warning") if "switch_warn" in ast.unparse(c)]
    assert len(warned) == 1 and made.lineno < warned[0].lineno
    assert any(in_body and "switch_warn" in test for test, in_body in _branch_tests(init, warned[0]))


@pytest.mark.parametrize("env", [
    {},
    {"VICA_RECUT": "on"},
    {"VICA_RECUT": "on", "VICA_ANSWER": "on"},
    {"VICA_TALK": "on"},
    {"VICA_TALK": "on", "VICA_ANSWER": "on"},
    {"VICA_TALK": "on", "VICA_RECUT": "on"},
])
def test_off_combinations_offer_neither_value(monkeypatch, env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    q = OpenQuestion("q", "confirm", "화장실로 안내해드릴까요?", ("wc",), 0.0)
    for text in (parser._build_system_prompt(DESTS, question=q), parser.build_audio_prompt(DESTS),
                 json.dumps(parser._draft_cls().model_json_schema(), ensure_ascii=False),
                 json.dumps(build_intent_tool(), ensure_ascii=False)):
        assert "repeat" not in text and "places" not in text
    with pytest.raises(Exception):
        parser._draft_cls()(intent="places")


def test_p28_p29_on_schema_and_tool_offer_both_values(monkeypatch):
    talk_on(monkeypatch)
    enum = parser._draft_cls().model_json_schema()["properties"]["intent"]["enum"]
    assert {"repeat", "places", "not_for_robot", "dismiss"} <= set(enum)
    assert parser._draft_cls() is parser._draft_cls()
    assert "answer" in parser._draft_cls().model_fields
    tool = build_intent_tool()["parameters"]
    assert {"repeat", "places", "not_for_robot", "dismiss"} <= set(tool["properties"]["intent"]["enum"])
    assert tool["required"][:2] == ["intent", "heard_text"]
    assert parser._draft_cls()(intent="repeat").intent == "repeat"


def test_p28_p29_on_both_prompts_carry_the_two_lines(monkeypatch):
    talk_on(monkeypatch)
    for prompt in (parser._build_system_prompt(DESTS), parser.build_audio_prompt(DESTS)):
        assert LINE_REPEAT in prompt.split("\n") and LINE_PLACES in prompt.split("\n")


def test_mission_gets_repeat_but_never_places():
    assert VicaIntent(intent="repeat").intent == "repeat"
    with pytest.raises(Exception):
        VicaIntent(intent="places")


class _Manager:
    """parse_intent 의 백엔드 자리 — 받은 messages 를 적고 정해 둔 초안을 돌려준다."""
    has_local = False

    def __init__(self, **draft):
        self.draft, self.seen = draft, []

    def invoke(self, messages, local_messages=None):
        self.seen.append(messages)
        return parser._draft_cls()(**self.draft)


def _text(monkeypatch, dests=DESTS, said="어디 갈 수 있어?", **draft):
    manager = _Manager(**draft)
    monkeypatch.setattr(parser, "PROVIDER", "openai")
    monkeypatch.setattr(parser, "get_backend_manager", lambda: manager)
    return parser.parse_intent(said, dests), manager


def test_p29_text_path_places_is_a_code_sentence(monkeypatch):
    talk_on(monkeypatch)
    out, manager = _text(monkeypatch, intent="places", reply="화장실이랑 식당이 있어요.", confidence=0.9)
    assert (out.intent, out.reply, out.need_confirm) == ("question", "화장실, 사회복지, 407호로 안내할 수 있습니다.", False)
    assert out.matched_destination_id is None and out.destination_candidate is None
    assert LINE_PLACES in manager.seen[0][0].content


def test_p28_text_path_repeat_goes_out_bare(monkeypatch):
    talk_on(monkeypatch)
    out, _ = _text(monkeypatch, said="뭐라고?", intent="repeat", reply="다시 말씀드리면…", destination_candidate="화장실")
    assert (out.intent, out.reply, out.need_confirm, out.matched_destination_id) == ("repeat", "", False, None)
    assert out.destination_candidate is None


def test_p29_no_reachable_place_asks_again_on_the_text_path(monkeypatch):
    talk_on(monkeypatch)
    out, _ = _text(monkeypatch, dests=CLOSED, intent="places", reply="")
    assert (out.intent, out.reply) == ("unknown", RETRY_PROMPT)
    out, _ = _text(monkeypatch, dests=CLOSED, intent="places", reply="지금은 안내할 수 있는 곳이 없어요.")
    assert (out.intent, out.reply) == ("question", "지금은 안내할 수 있는 곳이 없어요.")


class _Realtime:
    def __init__(self, draft, heard):
        self.draft, self.heard, self.instructions = draft, heard, ""

    def ask(self, pcm, history, instructions, **kw):
        self.instructions = instructions
        return RealtimeResult(draft=self.draft, heard_text=self.heard, latency_sec=0.1)


def _audio(monkeypatch, draft, heard, dests=DESTS, question=None):
    client = _Realtime(draft, heard)
    monkeypatch.setattr(parser, "get_realtime_client", lambda: client)
    intent, _, _, _ = parser.parse_intent_audio(b"\x00\x00", dests, question=question)
    return intent, client


def test_p29_audio_path_places_is_a_code_sentence(monkeypatch):
    talk_on(monkeypatch)
    out, client = _audio(monkeypatch, {"intent": "places", "reply": "여러 곳이 있어요.", "heard_text": "어디 갈 수 있어"},
                         "어디 갈 수 있어")
    assert (out.intent, out.reply) == ("question", "화장실, 사회복지, 407호로 안내할 수 있습니다.")
    assert LINE_PLACES in client.instructions and LINE_REPEAT in client.instructions


def test_p28_audio_path_repeat_goes_out_bare(monkeypatch):
    talk_on(monkeypatch)
    q = OpenQuestion("q", "confirm", "화장실로 안내해드릴까요?", ("wc",), 0.0)
    out, _ = _audio(monkeypatch, {"intent": "repeat", "reply": "", "heard_text": "뭐라고", "answer": "other"}, "뭐라고",
                    question=q)
    assert (out.intent, out.reply, out.need_confirm) == ("repeat", "", False)


def test_p29_no_reachable_place_asks_again_on_the_audio_path(monkeypatch):
    talk_on(monkeypatch)
    out, _ = _audio(monkeypatch, {"intent": "places", "reply": ""}, "어디 갈 수 있어", dests=CLOSED)
    assert (out.intent, out.reply) == ("unknown", RETRY_PROMPT)


from tests.test_parser_recut import backends  # noqa: E402,F401  (진짜 LlmBackendManager + 가짜 cloud·Ollama)

_ONE = [DestinationData(id="men", name="남자 화장실", confirm_prompt="남자 화장실로 안내해드릴까요?")]


def test_p16_local_call_carries_no_talk_lines_or_values(monkeypatch, backends):  # noqa: F811
    calls, cfg = backends
    talk_on(monkeypatch)
    monkeypatch.setattr(parser, "PROVIDER", "openai")
    cfg["cloud_fail"] = RuntimeError("cloud down")
    parser.parse_intent("어디 갈 수 있어?", _ONE)
    (cloud, _, cloud_msgs), (local, schema, local_msgs) = calls
    assert (cloud, local) == ("cloud", "local")
    assert LINE_PLACES in cloud_msgs[0].content
    assert schema is parser._IntentDraft
    assert LINE_PLACES not in local_msgs[0].content and LINE_REPEAT not in local_msgs[0].content
    plain = local_msgs[0].content
    monkeypatch.setenv("VICA_TALK", "off")
    assert plain == parser._build_system_prompt(_ONE, recut=False)


def test_p16_local_answer_skips_the_new_post_processing(monkeypatch, backends):  # noqa: F811
    calls, cfg = backends
    talk_on(monkeypatch)
    monkeypatch.setattr(parser, "PROVIDER", "openai")
    seen = []
    real = fa().early
    monkeypatch.setattr(fa(), "early", lambda *a, **k: seen.append(a) or real(*a, **k))
    cfg["cloud_fail"] = RuntimeError("cloud down")
    parser.parse_intent("어디 갈 수 있어?", _ONE)
    assert seen == []
    cfg["cloud_fail"] = None
    parser.reset_backend_manager()
    parser.parse_intent("어디 갈 수 있어?", _ONE)
    assert len(seen) == 1


def test_with_text_tools_the_set_intent_tool_offers_both_values(monkeypatch):
    talk_on(monkeypatch)
    monkeypatch.setenv("VICA_TOOLS_TEXT", "on")
    params = parser._set_intent_tool()["function"]["parameters"]
    assert {"repeat", "places"} <= set(params["properties"]["intent"]["enum"])
    msg = types.SimpleNamespace(tool_calls=[{"name": "set_intent", "args": {"intent": "places", "reply": ""}}])
    assert parser._draft_from_tool_message(msg).intent == "places"


def test_with_audio_tools_both_tools_are_kept(monkeypatch):
    from src import realtime_intent
    talk_on(monkeypatch)
    monkeypatch.setenv("VICA_TOOLS_AUDIO", "on")
    monkeypatch.setattr(realtime_intent, "_facility_names", lambda: ["회의실"])
    tools, choice = realtime_intent.session_tools()
    assert [t["name"] for t in tools] == ["set_intent", "reserve_facility"] and choice == "required"
    assert {"repeat", "places"} <= set(tools[0]["parameters"]["properties"]["intent"]["enum"])
    prompt = parser.build_audio_prompt(DESTS)
    assert LINE_PLACES in prompt and "[시설 예약]" in prompt


def test_places_sentence_for_the_real_destinations_has_no_keyword():
    from src.destination_loader import load_destinations
    from src.emergency_filter import detect_emergency
    from src.replies import places_sentence
    sentence = places_sentence(fa().places_names(load_destinations()))
    assert sentence and detect_emergency(sentence) is None


class _Pub:
    def __init__(self):
        self.sent = []

    def publish(self, msg):
        keys = ("intent", "reply", "ask_text", "ask_kind", "ask_option_ids", "answer", "question_id", "speaker_verdict",
                "data")
        self.sent.append({k: getattr(msg, k, None) for k in keys})


@pytest.fixture
def node_mod(monkeypatch):
    """rclpy·std_msgs 자리를 가짜로 채우고 src.ros_node 를 가져온다(vica_interfaces 는 conftest 가 이미 가짜다)."""
    class _Msg:
        def __init__(self, data=None):
            self.data = data

    def module(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        return m

    for name, mod in {
        "rclpy": module("rclpy"),
        "rclpy.executors": module("rclpy.executors", ExternalShutdownException=Exception),
        "rclpy.node": module("rclpy.node", Node=type("Node", (), {})),
        "rclpy.qos": module("rclpy.qos", DurabilityPolicy=object, QoSProfile=object, ReliabilityPolicy=object),
        "std_msgs": module("std_msgs"),
        "std_msgs.msg": module("std_msgs.msg", Bool=_Msg, String=_Msg, UInt8MultiArray=_Msg),
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "src.ros_node", raising=False)
    yield importlib.import_module("src.ros_node")
    sys.modules.pop("src.ros_node", None)


def _node(question, recent, spk=None, history=()):
    from src.history import ConversationHistory
    log = types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, warn=lambda *a, **k: None)
    me = types.SimpleNamespace(
        _question=question, _robot_recent=list(recent), _destinations=DESTS, _history=ConversationHistory(),
        _spoken_history=False, _local=False, _hand_over=True,
        _spk_link=types.SimpleNamespace(take=lambda text, now: spk), _audio_spk=None, _last_robot_line=(0.0, ""),
        _robot_state=RobotState(), _spk_on=False, _followup_until=0.0,
        _intent_pub=_Pub(), _tts_pub=_Pub(), _listen_pub=_Pub(),
        _board=types.SimpleNamespace(on_intent=lambda i: None), get_logger=lambda: log)
    me._history.extend(list(history))
    return me


def _publish(node_mod, me, intent, text="뭐라고?"):
    node_mod.LlmIntentNode._publish_intent(me, intent, text)
    return me._intent_pub.sent, [m["data"] for m in me._tts_pub.sent]


def _row(m):
    return (m["intent"], m["reply"], m["ask_text"], m["ask_kind"], tuple(m["ask_option_ids"] or ()), m["answer"],
            m["question_id"])


REPEAT = VicaIntent(intent="repeat", need_confirm=False)
ASKED = OpenQuestion("q1", "confirm", "화장실로 안내해드릴까요?", ("wc",), 0.0)
WAKE = OpenQuestion("q2", "open", "네?", (), 0.0)
ASKED_HISTORY = (HumanMessage("화장실 가고 싶어"), AIMessage("화장실로 안내해드릴까요?"))


def test_p28_rule3_node_mission_question_goes_with_its_text_and_no_answer(node_mod, monkeypatch):
    talk_on(monkeypatch)
    me = _node(ASKED, [(1.0, ASKED.text)], spk={"verdict": "pass", "score": 0.8, "joined": 0.5}, history=ASKED_HISTORY)
    parsed = VicaIntent(intent="repeat", need_confirm=False, confidence=0.9, question_id="q1", answer="unclear")
    sent, said = _publish(node_mod, me, parsed)
    assert [_row(m) for m in sent] == [("repeat", "", ASKED.text, "confirm", ("wc",), "", "")]
    assert sent[0]["speaker_verdict"] == "pass"
    assert said == []


def test_p28_c1_node_repeat_to_the_mission_leaves_no_empty_robot_line(node_mod, monkeypatch):
    talk_on(monkeypatch)
    me = _node(ASKED, [(1.0, ASKED.text)], history=ASKED_HISTORY)
    _publish(node_mod, me, REPEAT)
    msgs = me._history.messages
    assert type(msgs[-1]) is HumanMessage and msgs[-1].content == "뭐라고?"
    assert [m.content for m in msgs if isinstance(m, AIMessage)] == ["화장실로 안내해드릴까요?"]
    um = parser._shortcut_intent("음", msgs, DESTS)
    assert um is not None and um.intent == "clarify"


def test_p28_rule2_node_question_under_wake_greeting_goes_to_the_mission(node_mod, monkeypatch):
    talk_on(monkeypatch)
    me = _node(WAKE, [(1.0, ASKED.text), (2.0, "네?")], history=ASKED_HISTORY)
    sent, said = _publish(node_mod, me, REPEAT)
    assert [_row(m) for m in sent] == [("repeat", "", ASKED.text, "open", (), "", "")]
    assert said == [] and isinstance(me._history.messages[-1], HumanMessage)


def test_p28_rule2_node_statement_under_wake_greeting_is_said_by_the_voice(node_mod, monkeypatch):
    talk_on(monkeypatch)
    me = _node(WAKE, [(1.0, "화장실에 도착했습니다."), (2.0, "네?")])
    sent, said = _publish(node_mod, me, REPEAT)
    assert [(m["intent"], m["reply"]) for m in sent] == [("question", "화장실에 도착했습니다.")]
    assert len(said) == 1 and said[0].endswith(":화장실에 도착했습니다.")
    assert [type(m) for m in me._history.messages] == [HumanMessage, AIMessage]
    assert me._history.messages[-1].content == "화장실에 도착했습니다."


def test_p28_rule2_node_nothing_under_wake_greeting_goes_bare_to_the_mission(node_mod, monkeypatch):
    talk_on(monkeypatch)
    me = _node(WAKE, [(2.0, "네?")])
    sent, said = _publish(node_mod, me, REPEAT)
    assert [_row(m) for m in sent] == [("repeat", "", "", "", (), "", "")]
    assert said == []


@pytest.mark.parametrize("q", [
    OpenQuestion("q3", "open", NOT_AVAILABLE_ASK, (), 0.0),
    OpenQuestion("q4", "open", "네, 어디로 모실까요?", (), 0.0),
    OpenQuestion("q5", "yesno", "다시 출발할까요?", (), 0.0),
])
def test_p28_rule3_node_open_question_goes_to_the_mission_with_its_text(node_mod, monkeypatch, q):
    talk_on(monkeypatch)
    me = _node(q, [(1.0, q.text)])
    sent, said = _publish(node_mod, me, REPEAT)
    assert [_row(m) for m in sent] == [("repeat", "", q.text, q.kind, (), "", "")]
    assert said == []


def test_p28_rule3_node_handed_choice_question_is_handed_again_as_a_choice(node_mod, monkeypatch):
    talk_on(monkeypatch)
    choose = "화장실과 사회복지 중 어디로 갈까요?"
    me = _node(OpenQuestion("q6", "choice", choose, ("wc", "social"), 0.0), [(1.0, choose), (2.0, "엉뚱한 말이에요.")])
    sent, said = _publish(node_mod, me, REPEAT)
    assert [(m["intent"], m["ask_text"], m["ask_kind"], tuple(m["ask_option_ids"])) for m in sent] == [
        ("question", choose, "choice", ("wc", "social"))]
    assert said == []


def test_p28_m1_node_no_open_question_last_question_goes_to_the_mission(node_mod, monkeypatch):
    talk_on(monkeypatch)
    me = _node(None, [(1.0, "다시 출발할까요?")], history=(HumanMessage("다시 가자"), AIMessage("다시 출발할까요?")))
    sent, said = _publish(node_mod, me, REPEAT)
    assert [_row(m) for m in sent] == [("repeat", "", "다시 출발할까요?", "open", (), "", "")]
    assert said == [] and isinstance(me._history.messages[-1], HumanMessage)


def test_p28_n2_node_refused_question_is_not_asked_again(node_mod, monkeypatch):
    talk_on(monkeypatch)
    me = _node(WAKE, [(1.0, ASKED.text), (4.0, "안내 요청이 취소되었습니다."), (8.0, "네?")])
    sent, said = _publish(node_mod, me, REPEAT)
    assert [(m["intent"], m["reply"], m["ask_text"]) for m in sent] == [("question", "안내 요청이 취소되었습니다.", "")]
    assert len(said) == 1 and said[0].endswith(":안내 요청이 취소되었습니다.")


def test_p28_p8_node_nothing_to_repeat_asks_again(node_mod, monkeypatch):
    talk_on(monkeypatch)
    me = _node(None, [])
    sent, said = _publish(node_mod, me, REPEAT)
    assert [m["intent"] for m in sent] == ["unknown"]
    assert RETRY_PROMPT in (sent[0]["reply"], sent[0]["ask_text"], *[s.split(":", 1)[-1] for s in said])


def test_p28_repeat_ask_fields_survive_the_one_voice_and_quiet_steps():
    from src.mission_question import quiet_for_mission
    from src.one_voice import hand_over_question
    out = VicaIntent(intent="repeat", need_confirm=False, ask_text=ASKED.text, ask_kind="confirm", ask_option_ids=["wc"])
    assert hand_over_question(out, DESTS) == out
    for state in ("", "confirming", "awaiting_user", "waiting", "asking_next", "paused"):
        assert quiet_for_mission(out, state, ASKED.text, 1.0, [d.name for d in DESTS]) == out


def test_p28_node_wiring_settles_repeat_before_publishing():
    import ast
    from tests.test_one_voice_wiring import _calls, _method
    fn = _method("_publish_intent")
    body = ast.unparse(fn)
    (call,) = _calls(fn, "fixed_answers.settle_repeat")
    assert [ast.unparse(a) for a in call.args] == ["question", "self._robot_recent"]
    assert body.index("NOT_FOR_ROBOT") < body.index("settle_repeat") < body.index("_intent_pub.publish")

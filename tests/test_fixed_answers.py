"""정해진 대답 둘 — "다시 말해줘"(repeat)·"어디 갈 수 있어?"(places) (설계 6.4, 원칙 28·29, Task 4.2).

`.env` VICA_TALK=on 이 재단(VICA_RECUT=on)과 함께 켜졌을 때만 LLM 이 두 값을 고를 수 있고, 그 대답의 문장은 코드가 만든다.
로컬(Ollama)이 답하는 호출에는 싣지 않는다(원칙 16). 의도 노드(ros_node)는 rclpy 없이 못 띄워 ROS 모듈 자리를 가짜로 채워
_publish_intent 하나만 부른다.
"""
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

LINE_REPEAT = '- repeat: 로봇이 방금 한 말을 다시 해 달라고 한다("다시 말해줘", "뭐라고?", "못 들었어"). reply 는 빈 문자열.'
LINE_PLACES = ('- places: 어디로 갈 수 있는지, 무엇을 할 수 있는지 묻는다("어디 갈 수 있어?", "뭐 할 수 있어?"). '
               'reply 는 빈 문자열 — 문장은 시스템이 만든다.')

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


def talk_on(monkeypatch, answer=False):
    monkeypatch.setenv("VICA_RECUT", "on")
    monkeypatch.setenv("VICA_TALK", "on")
    if answer:
        monkeypatch.setenv("VICA_ANSWER", "on")


# ---- 설명서 Step 1 의 넷 ------------------------------------------------------------------------------------------
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
    assert names == ["화장실", "사회복지", "407호"]          # 목록 순서
    assert fa().places_names(CLOSED) == []


def test_p28_repeat_says_the_open_question_or_the_last_sentence():
    last_said, settle_repeat = fa().last_said, fa().settle_repeat
    recent = [(1.0, "화장실에 도착했습니다."), (2.0, "네?")]
    assert last_said(recent) == "화장실에 도착했습니다."          # 평서문도 다시 한다, "네?"는 건너뛴다
    asked = OpenQuestion("q", "confirm", "화장실로 안내해드릴까요?", ("t",), 0.0)
    assert settle_repeat(asked, recent).intent == "repeat"         # 미션이 그 질문을 다시 말한다
    wake = OpenQuestion("q", "open", "네?", (), 0.0)
    out = settle_repeat(wake, recent)
    assert (out.intent, out.reply) == ("question", "화장실에 도착했습니다.")
    assert settle_repeat(None, []).intent == "unknown"


def test_p28_p9_repeat_goes_to_the_mission_only_for_questions_it_holds():
    # 미션은 자기가 물은 종류의 질문만 쥔다(미션 Task 4.3). kind open 은 "네?"뿐 아니라 음성이 넘긴 되묻기도 있다 — 미션이 쥐고
    # 있지 않아 repeat 에 말이 없으니 음성이 마지막 말을 다시 한다(controller 결정, 11-21). choice 도 음성이 넘긴 고르기 질문뿐이다
    # (미션은 choice 를 스스로 묻지 않는다 — mission_logic 의 _ask(…, "choice") 0건) — 같은 길.
    settle_repeat = fa().settle_repeat
    recent = [(1.0, "제가 아는 목적지 중에서 스무디 가게는 없는데, 어디로 안내해드릴까요?")]
    for kind in ("confirm", "yesno", "yesno_inverted", "minutes"):
        out = settle_repeat(OpenQuestion("q", kind, "질문?", (), 0.0), recent)
        assert (out.intent, out.reply, out.answer, out.question_id) == ("repeat", "", "", "")
    for kind in ("open", "choice"):
        out = settle_repeat(OpenQuestion("q", kind, recent[0][1], (), 0.0), recent)
        assert (out.intent, out.reply, out.need_confirm) == ("question", recent[0][1], False)
    assert settle_repeat(None, recent).reply == recent[0][1]
    assert settle_repeat(None, [(1.0, "네?"), (2.0, " 네? ")]).reply == RETRY_PROMPT


def test_off_schema_and_prompts_have_no_repeat_or_places():
    assert "places" not in parser._build_system_prompt(DESTS) and "places" not in json.dumps(build_intent_tool())


# ---- 스위치 ------------------------------------------------------------------------------------------------------
def test_conftest_keeps_talk_off():
    assert os.environ.get("VICA_TALK") == "off"


def test_p28_switch_needs_recut(monkeypatch):
    monkeypatch.setenv("VICA_TALK", "on")
    monkeypatch.setenv("VICA_RECUT", "off")
    assert fa().on() is False                                    # 4단계는 2단계 위에서만 켜진다
    monkeypatch.setenv("VICA_RECUT", "on")
    assert fa().on() is True
    monkeypatch.setenv("VICA_TALK", "off")
    assert fa().on() is False


@pytest.mark.parametrize("env", [
    {},                                                          # ① 전부 꺼짐
    {"VICA_RECUT": "on"},                                        # ② 재단만
    {"VICA_RECUT": "on", "VICA_ANSWER": "on"},                   # ③ 재단 + 대답 칸
    {"VICA_TALK": "on"},                                         # 재단 없이 VICA_TALK 만 — ①과 같아야 한다
    {"VICA_TALK": "on", "VICA_ANSWER": "on"},
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


# ---- 켰을 때: 값·지시문 두 줄 -----------------------------------------------------------------------------------
@pytest.mark.parametrize("answer", [False, True])
def test_p28_p29_on_schema_and_tool_offer_both_values(monkeypatch, answer):
    talk_on(monkeypatch, answer)
    enum = parser._draft_cls().model_json_schema()["properties"]["intent"]["enum"]
    assert {"repeat", "places", "not_for_robot", "dismiss"} <= set(enum)
    assert parser._draft_cls() is parser._draft_cls()             # 조합마다 같은 클래스
    assert ("answer" in parser._draft_cls().model_fields) is answer
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


# ---- 글자 경로 끝에서 끝까지(가짜 LLM) ------------------------------------------------------------------------------
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
    monkeypatch.setattr(parser, "PROVIDER", "openai")     # 외부 API 가 답하는 글자 경로(공급자 ollama 면 Ollama 가 답한다 — 원칙 16)
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
    # 갈 수 있는 곳이 0개 — 문장을 만들지 않는다. LLM 이 reply 를 비웠으면(지시대로) 되묻는다(unknown → RETRY_PROMPT).
    talk_on(monkeypatch)
    out, _ = _text(monkeypatch, dests=CLOSED, intent="places", reply="")
    assert (out.intent, out.reply) == ("unknown", RETRY_PROMPT)
    out, _ = _text(monkeypatch, dests=CLOSED, intent="places", reply="지금은 안내할 수 있는 곳이 없어요.")
    assert (out.intent, out.reply) == ("question", "지금은 안내할 수 있는 곳이 없어요.")   # 지금의 question 길


# ---- 소리 경로(가짜 Realtime) -----------------------------------------------------------------------------------
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
    talk_on(monkeypatch, answer=True)
    q = OpenQuestion("q", "confirm", "화장실로 안내해드릴까요?", ("wc",), 0.0)
    out, _ = _audio(monkeypatch, {"intent": "repeat", "reply": "", "heard_text": "뭐라고", "answer": "other"}, "뭐라고",
                    question=q)
    assert (out.intent, out.reply, out.need_confirm) == ("repeat", "", False)


def test_p29_no_reachable_place_asks_again_on_the_audio_path(monkeypatch):
    talk_on(monkeypatch)
    out, _ = _audio(monkeypatch, {"intent": "places", "reply": ""}, "어디 갈 수 있어", dests=CLOSED)
    assert (out.intent, out.reply) == ("unknown", RETRY_PROMPT)    # 글자 경로와 같다 — 소리 경로의 빈 question 은 침묵이 된다


# ---- 로컬(Ollama)이 답하는 호출에는 싣지 않는다(원칙 16) ----------------------------------------------------------
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
    assert plain == parser._build_system_prompt(_ONE, recut=False)   # 재단만 켠 때의 로컬 지시문과 글자까지 같다


def test_p16_local_answer_skips_the_new_post_processing(monkeypatch, backends):  # noqa: F811
    calls, cfg = backends
    talk_on(monkeypatch)
    monkeypatch.setattr(parser, "PROVIDER", "openai")
    seen = []
    real = fa().early
    monkeypatch.setattr(fa(), "early", lambda *a, **k: seen.append(a) or real(*a, **k))
    cfg["cloud_fail"] = RuntimeError("cloud down")
    parser.parse_intent("어디 갈 수 있어?", _ONE)
    assert seen == []                                            # 로컬이 답했다 — 새 뒤처리를 태우지 않는다
    cfg["cloud_fail"] = None
    parser.reset_backend_manager()
    parser.parse_intent("어디 갈 수 있어?", _ONE)
    assert len(seen) == 1                                        # 클라우드가 답하면 탄다


# ---- 예약 도구와 함께 켰을 때 -----------------------------------------------------------------------------------
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


# ---- 말할 문장에 긴급어가 없다(자가 트리거, test_spoken_text 와 같은 검사) -------------------------------------------
def test_places_sentence_for_the_real_destinations_has_no_keyword():
    from src.destination_loader import load_destinations
    from src.emergency_filter import detect_emergency
    from src.replies import places_sentence
    sentence = places_sentence(fa().places_names(load_destinations()))
    assert sentence and detect_emergency(sentence) is None


# ---- 의도 노드 _publish_intent 의 repeat 갈래 ------------------------------------------------------------------
class _Pub:
    def __init__(self):
        self.sent = []

    def publish(self, msg):
        keys = ("intent", "reply", "ask_text", "ask_kind", "answer", "question_id", "speaker_verdict", "data")
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


def _node(question, recent, spk=None):
    from src.history import ConversationHistory
    log = types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, warn=lambda *a, **k: None)
    me = types.SimpleNamespace(
        _question=question, _robot_recent=list(recent), _destinations=DESTS, _history=ConversationHistory(),
        _spoken_history=False, _local=False, _hand_over=True,
        _spk_link=types.SimpleNamespace(take=lambda text, now: spk), _audio_spk=None, _last_robot_line=(0.0, ""),
        _robot_state=RobotState(), _spk_on=False, _followup_until=0.0,
        _intent_pub=_Pub(), _tts_pub=_Pub(), _listen_pub=_Pub(),
        _board=types.SimpleNamespace(on_intent=lambda i: None), get_logger=lambda: log)
    return me


def _publish(node_mod, me, intent, text="뭐라고?"):
    node_mod.LlmIntentNode._publish_intent(me, intent, text)
    return me._intent_pub.sent, [m["data"] for m in me._tts_pub.sent]


ASKED = OpenQuestion("q1", "confirm", "화장실로 안내해드릴까요?", ("wc",), 0.0)
WAKE = OpenQuestion("q2", "open", "네?", (), 0.0)


def test_p28_p9_node_mission_question_open_sends_repeat_and_voice_stays_silent(node_mod, monkeypatch):
    talk_on(monkeypatch, answer=True)
    me = _node(ASKED, [(1.0, "화장실로 안내해드릴까요?")], spk={"verdict": "pass", "score": 0.8, "joined": 0.5})
    # 해석기가 대답 칸을 채워 왔어도(LLM 이 unclear) 미션이 다른 의도로 바꿔 읽지 않게 칸을 비운다
    parsed = VicaIntent(intent="repeat", need_confirm=False, confidence=0.9, question_id="q1", answer="unclear")
    sent, said = _publish(node_mod, me, parsed)
    assert [(m["intent"], m["reply"], m["ask_text"], m["answer"], m["question_id"]) for m in sent] == [
        ("repeat", "", "", "", "")]
    assert sent[0]["speaker_verdict"] == "pass"                  # 화자 판정은 그대로 실린다
    assert said == []                                            # 한 목소리 — 미션이 그 질문을 다시 말한다


def test_p28_node_after_wake_greeting_voice_says_the_last_sentence_again(node_mod, monkeypatch):
    talk_on(monkeypatch)
    me = _node(WAKE, [(1.0, "화장실에 도착했습니다."), (2.0, "네?")])
    sent, said = _publish(node_mod, me, VicaIntent(intent="repeat", need_confirm=False))
    assert [(m["intent"], m["reply"]) for m in sent] == [("question", "화장실에 도착했습니다.")]
    assert len(said) == 1 and said[0].endswith(":화장실에 도착했습니다.")
    # 대화 기록(글자 모드): 사용자 줄과 다시 한 말 — 다음 "다시 말해줘"의 last_said 도 같은 말이다(tts_done 으로 들어온다)
    assert [type(m) for m in me._history.messages] == [HumanMessage, AIMessage]
    assert me._history.messages[-1].content == "화장실에 도착했습니다."


def test_p28_node_handed_open_question_is_said_again_through_the_mission(node_mod, monkeypatch):
    # 음성이 넘긴 되묻기(kind open)는 미션이 쥐고 있지 않다 — 음성이 마지막 말을 다시 하고, 질문 꼴이라 지금 길(hand_over_question)
    # 대로 미션에 넘어가 다시 열린 질문이 된다. 음성은 직접 말하지 않는다(한 목소리).
    talk_on(monkeypatch)
    asked_again = "제가 아는 목적지 중에서 스무디 가게는 없는데, 어디로 안내해드릴까요?"
    me = _node(OpenQuestion("q3", "open", asked_again, (), 0.0), [(1.0, asked_again)])
    sent, said = _publish(node_mod, me, VicaIntent(intent="repeat", need_confirm=False))
    assert [(m["intent"], m["reply"], m["ask_text"], m["ask_kind"]) for m in sent] == [
        ("question", "", asked_again, "open")]
    assert said == []


def test_p28_node_handed_choice_question_is_handed_again_as_a_choice(node_mod, monkeypatch):
    # 음성이 넘긴 고르기 질문 — 미션은 쥐고 있지 않다(CONFIRMING 중이었다면 repeat 에 확인 질문을 다시 말해 엉뚱한 질문이 된다).
    talk_on(monkeypatch)
    choose = "화장실과 사회복지 중 어디로 갈까요?"
    me = _node(OpenQuestion("q4", "choice", choose, ("wc", "social"), 0.0), [(1.0, choose), (2.0, "네?")])
    sent, said = _publish(node_mod, me, VicaIntent(intent="repeat", need_confirm=False))
    assert [(m["intent"], m["ask_text"], m["ask_kind"]) for m in sent] == [("question", choose, "choice")]
    assert said == []


def test_p28_p8_node_nothing_to_repeat_asks_again(node_mod, monkeypatch):
    talk_on(monkeypatch)
    for question, recent in ((None, []), (WAKE, [(2.0, "네?")])):
        me = _node(question, recent)
        sent, said = _publish(node_mod, me, VicaIntent(intent="repeat", need_confirm=False))
        assert [m["intent"] for m in sent] == ["unknown"]
        assert RETRY_PROMPT in (sent[0]["reply"], sent[0]["ask_text"], *[s.split(":", 1)[-1] for s in said])


def test_p28_node_wiring_settles_repeat_before_publishing():
    import ast
    from tests.test_one_voice_wiring import _calls, _method
    fn = _method("_publish_intent")
    body = ast.unparse(fn)
    (call,) = _calls(fn, "fixed_answers.settle_repeat")
    assert [ast.unparse(a) for a in call.args] == ["question", "self._robot_recent"]
    assert body.index("NOT_FOR_ROBOT") < body.index("settle_repeat") < body.index("_intent_pub.publish")

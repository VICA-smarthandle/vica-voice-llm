"""10-24 미션 대역 시험(12:43~12:50, ⑬ midm2-mini)에서 나온 두 가지 — 사용자 "1번이랑 3번 고쳐주세요"."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src import langchain_intent_parser as parser
from src import local_rules
from src.langchain_intent_parser import _IntentDraft, parse_intent, parse_intent_audio
from src.realtime_intent import RealtimeResult
from src.schema import DestinationData, RobotState

SOCIAL = DestinationData(id="social", name="사회복지", aliases=["사회복지", "장애인", "서류", "장애인 서류", "복지"],
                         confirm_prompt="사회복지로 안내해드릴까요?")
MEN = DestinationData(id="men", name="남자 화장실",
                      aliases=["남자 화장실", "남자", "화장실", "남자화장실", "대변", "소변", "배아파", "아파"],
                      confirm_prompt="남자 화장실로 안내해드릴까요?")
WOMEN = DestinationData(id="women", name="여자 화장실",
                        aliases=["여자 화장실", "여자", "화장실", "여자화장실", "배아파", "아파", "대변", "소변"],
                        confirm_prompt="여자 화장실로 안내해드릴까요?")
DESTS = [SOCIAL, MEN, WOMEN]
CHOOSE = "남자 화장실과 여자 화장실 중 어디로 갈까요?"
NOT_HEARD = {"intent": "unknown", "reply": "잘 듣지 못했습니다. 다시 말씀해 주세요.", "confidence": 0.2}


def finalize(user_text, draft, dialog):
    return parser._finalize(_IntentDraft(**draft), DESTS, user_text=user_text, local=True,
                            robot_state=RobotState(dialog_state=dialog))


class TestUnknownWithSharedAliasAsksWhich:
    @pytest.mark.parametrize("dialog", ["waiting_release", "waiting", "idle", "asking_next"])
    def test_log_1249_unknown_bae_apa_becomes_choice(self, dialog):
        out = finalize("나 배 아파.", NOT_HEARD, dialog)
        assert out.intent == "clarify" and out.reply == CHOOSE and out.matched_destination_id == ""

    def test_not_during_approach_question(self):
        out = finalize("나 배 아파.", NOT_HEARD, "awaiting_user")
        assert out.intent == "unknown"

    @pytest.mark.parametrize("said", ["오늘 날씨 좋네", "남자 화장실 어디야", "서류"])
    def test_other_unknowns_are_left(self, said):
        out = finalize(said, NOT_HEARD, "idle")
        assert out.intent == "unknown"

    def test_audio_mode_unknown_is_left_to_the_model(self, monkeypatch):
        class FakeRT:
            def ask(self, pcm, history, instructions):
                return RealtimeResult(draft={"intent": "unknown", "reply": "", "confidence": 0.3},
                                      heard_text="나 배 아파", latency_sec=0.7, usage={})
        monkeypatch.setattr(parser, "get_realtime_client", lambda: FakeRT())
        intent, _, _, _ = parse_intent_audio(b"\x00\x00" * 1600, DESTS, robot_state=RobotState(dialog_state="idle"))
        assert intent.intent == "unknown" and intent.reply == ""


class _Manager:
    has_local = True

    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return _IntentDraft(intent="clarify", reply="무슨 말씀이신가요?", confidence=0.2)


@pytest.fixture
def llm(monkeypatch):
    m = _Manager()
    monkeypatch.setattr(parser, "get_backend_manager", lambda: m)
    return m


ASK_WC = "남자 화장실에 도착했습니다. 다녀오시는 동안 여기서 기다릴까요?"


@pytest.mark.parametrize("local", [True, False])
@pytest.mark.parametrize("said", ["하니.", "하니", " 하니? "])
def test_log_1244_hani_to_a_mission_question_is_deny(llm, monkeypatch, said, local):
    monkeypatch.setattr(local_rules, "enabled", lambda: local)
    out = parse_intent(said, DESTS, history=[AIMessage(ASK_WC)], robot_state=RobotState(dialog_state="asking_next"))
    assert out.intent == "deny" and out.reply == "" and llm.calls == 0


def test_hani_while_confirming_is_deny(llm, monkeypatch):
    monkeypatch.setattr(local_rules, "enabled", lambda: True)
    history = [HumanMessage("나 장애인 서류 내야 돼."), AIMessage(SOCIAL.confirm_prompt)]
    out = parse_intent("하니.", DESTS, history=history, robot_state=RobotState(dialog_state="confirming"))
    assert out.intent == "deny" and llm.calls == 0


@pytest.mark.parametrize("dialog", ["idle", "waiting", "navigating"])
def test_hani_when_nobody_asked_goes_to_the_model(llm, monkeypatch, dialog):
    monkeypatch.setattr(local_rules, "enabled", lambda: True)
    parse_intent("하니.", DESTS, history=[], robot_state=RobotState(dialog_state=dialog))
    assert llm.calls == 1


def test_hani_inside_a_sentence_is_not_a_no(llm, monkeypatch):
    monkeypatch.setattr(local_rules, "enabled", lambda: True)
    parse_intent("하니까 좀 이따 갈게", DESTS, history=[AIMessage(ASK_WC)],
                 robot_state=RobotState(dialog_state="asking_next"))
    assert llm.calls == 1


def test_yes_then_hani_while_confirming_does_not_depart(llm, monkeypatch):
    monkeypatch.setattr(local_rules, "enabled", lambda: True)
    history = [HumanMessage("나 장애인 서류 내야 돼."), AIMessage(SOCIAL.confirm_prompt)]
    out = parse_intent("네. 하니.", DESTS, history=history, robot_state=RobotState(dialog_state="confirming"))
    assert out.intent == "deny" and llm.calls == 0

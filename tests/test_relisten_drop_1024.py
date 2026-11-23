"""재청취 기각(ros_node 2-1)이 진짜 요청을 지우던 두 경우(2026-10-24 음성만 시험, 사용자 "재청취 기각이 많아요")."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src import langchain_intent_parser as parser
from src.langchain_intent_parser import _IntentDraft, is_relisten_noise, is_scripted_reply, parse_intent
from src.replies import WAKE_GREETING, choose_between
from src.schema import DestinationData, RobotState, VicaIntent

SOCIAL = DestinationData(id="social", name="사회복지", aliases=["사회복지", "장애인", "서류", "장애인 서류", "복지"],
                         confirm_prompt="사회복지로 안내해드릴까요?")
MEN = DestinationData(id="men", name="남자 화장실",
                      aliases=["남자 화장실", "남자", "화장실", "남자화장실", "대변", "소변", "배아파", "아파"],
                      confirm_prompt="남자 화장실로 안내해드릴까요?")
WOMEN = DestinationData(id="women", name="여자 화장실",
                        aliases=["여자 화장실", "여자", "화장실", "여자화장실", "배아파", "아파", "대변", "소변"],
                        confirm_prompt="여자 화장실로 안내해드릴까요?")
R407 = DestinationData(id="r407", name="407호", aliases=["407호"], confirm_prompt="407호로 안내해드릴까요?")
DESTS = [SOCIAL, MEN, WOMEN, R407]
CHOOSE = "남자 화장실과 여자 화장실 중 어디로 갈까요?"


def clarify(reply):
    return VicaIntent(intent="clarify", reply=reply, need_confirm=False)


class TestWakeGreetingWindowIsACallWindow:
    def test_choice_after_wake_greeting_is_kept(self):
        assert is_relisten_noise(clarify(CHOOSE), DESTS, in_window=True, last_robot_text=WAKE_GREETING) is False

    def test_chatter_after_wake_greeting_is_kept_like_a_direct_call(self):
        chatter = VicaIntent(intent="unknown", reply="무슨 말씀이신지 잘 모르겠어요.")
        assert is_relisten_noise(chatter, DESTS, in_window=True, last_robot_text=" 네? ") is False

    def test_chatter_after_a_question_is_still_dropped(self):
        made_up = clarify("어느 곳을 말씀하시는 건가요?")
        assert is_relisten_noise(made_up, DESTS, in_window=True, last_robot_text=SOCIAL.confirm_prompt) is True

    def test_outside_window_is_never_dropped(self):
        assert is_relisten_noise(clarify("어느 곳을 말씀하시는 건가요?"), DESTS, in_window=False,
                                 last_robot_text=SOCIAL.confirm_prompt) is False

    def test_emergency_is_never_dropped(self):
        urgent = VicaIntent(intent="unknown", reply="", safety_flag="emergency")
        assert is_relisten_noise(urgent, DESTS, in_window=True, last_robot_text=SOCIAL.confirm_prompt) is False


class TestChoiceQuestionIsScripted:
    @pytest.mark.parametrize("names", [["남자 화장실", "여자 화장실"], ["여자 화장실", "남자 화장실"],
                                       ["사회복지", "407호"], ["407호", "사회복지"],
                                       ["남자 화장실", "여자 화장실", "407호"]])
    def test_fixed_choice_sentence_is_scripted(self, names):
        assert is_scripted_reply(choose_between(names), DESTS) is True

    def test_choice_after_a_question_is_kept(self):
        assert is_relisten_noise(clarify(CHOOSE), DESTS, in_window=True,
                                 last_robot_text=SOCIAL.confirm_prompt) is False

    @pytest.mark.parametrize("reply", ["식당과 여자 화장실 중 어디로 갈까요?",
                                       "남자 화장실 중 어디로 갈까요?",
                                       "남자 화장실과 여자 화장실 중 어디로 안내해드릴까요?"])
    def test_other_sentences_are_not_scripted(self, reply):
        assert is_scripted_reply(reply, DESTS) is False


class _Manager:
    has_local = True

    def invoke(self, messages):
        return _IntentDraft(intent="navigate", destination_candidate="남자 화장실", need_confirm=True,
                            reply="남자 화장실로 안내해드릴까요?", confidence=0.8)


def test_log_1217_bae_apa_after_rescued_call_is_published(monkeypatch):
    monkeypatch.setattr(parser, "get_backend_manager", lambda: _Manager())
    history = [HumanMessage("나 장애인 거류 내야 돼."), AIMessage(SOCIAL.confirm_prompt),
               HumanMessage("비카야"), AIMessage(WAKE_GREETING)]
    out = parse_intent("나 배 아파.", DESTS, history=history, robot_state=RobotState(dialog_state="idle"))
    assert out.intent == "clarify" and out.reply == CHOOSE
    assert is_relisten_noise(out, DESTS, in_window=True, last_robot_text=WAKE_GREETING) is False


def test_node_uses_the_pure_decision_with_last_robot_text():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "src" / "ros_node.py").read_text(encoding="utf-8")
    body = src[src.index("    def _publish_intent"):src.index("    def _on_listen_state_for_resume")]
    block = body[body.index("# 2-1)"):body.index("# 2-2)")]
    assert "is_relisten_noise(" in block and "last_robot_text=" in block

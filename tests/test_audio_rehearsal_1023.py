"""소리 모드 10-23 리허설(run105)에서 나온 셋 — 확인 질문 이름, 대답 단계의 짧은 긍정·부정, 고르기 질문 문장."""
import pytest

from src import langchain_intent_parser as parser
from src.langchain_intent_parser import parse_intent_audio
from src.realtime_intent import RealtimeResult
from src.schema import DestinationData, RobotState

SOCIAL = DestinationData(id="social", name="사회복지", aliases=["사회복지", "장애인", "서류", "장애인 서류", "복지"],
                         confirm_prompt="사회복지로 안내해드릴까요?")
R408 = DestinationData(id="r408", name="408호", aliases=["408호", "408"], confirm_prompt="408호로 안내해드릴까요?")
MEN = DestinationData(id="men", name="남자 화장실",
                      aliases=["남자 화장실", "남자", "화장실", "남자화장실", "대변", "소변", "배아파", "아파"],
                      confirm_prompt="남자 화장실로 안내해드릴까요?")
WOMEN = DestinationData(id="women", name="여자 화장실",
                        aliases=["여자 화장실", "여자", "화장실", "여자화장실", "배아파", "아파", "대변", "소변"],
                        confirm_prompt="여자 화장실로 안내해드릴까요?")
DESTS = [SOCIAL, R408, MEN, WOMEN]
PCM = b"\x00\x00" * 1600
CHOOSE = "남자 화장실과 여자 화장실 중 어디로 갈까요?"


class FakeRT:
    def __init__(self, draft, heard):
        self.draft, self.heard = draft, heard

    def ask(self, pcm, history, instructions):
        return RealtimeResult(draft=dict(self.draft), heard_text=self.heard, latency_sec=0.7, usage={})


@pytest.fixture
def rt(monkeypatch):
    def use(draft, heard, dialog="idle", active=""):
        monkeypatch.setattr(parser, "get_realtime_client", lambda: FakeRT(draft, heard))
        return parse_intent_audio(PCM, DESTS, robot_state=RobotState(dialog_state=dialog, active_destination=active))
    return use


def nav(name, reply, need_confirm=True):
    return {"intent": "navigate", "destination_candidate": name, "need_confirm": need_confirm, "reply": reply,
            "confidence": 0.85}


class TestConfirmQuestionNamesTheDestination:
    def test_other_words_become_the_list_phrase(self, rt):
        intent, _, _, _ = rt(nav("사회복지", "장애인 서류로 안내해드릴까요?"), "나 장애인 터류 내야돼")
        assert intent.intent == "navigate" and intent.matched_destination_id == "social"
        assert intent.need_confirm is True and intent.reply == SOCIAL.confirm_prompt

    def test_model_words_with_the_name_are_kept(self, rt):
        intent, _, _, _ = rt(nav("사회복지", "사회복지 창구로 모실까요?"), "서류 내야 해")
        assert intent.reply == "사회복지 창구로 모실까요?"

    def test_name_spacing_does_not_matter(self, rt):
        intent, _, _, _ = rt(nav("남자 화장실", "남자화장실로 안내해드릴까요?"), "남자 화장실로 가자")
        assert intent.reply == "남자화장실로 안내해드릴까요?"

    def test_switch_question_while_guiding_is_kept(self, rt):
        reply = "지금 남자 화장실로 가는 중이에요. 408호로 바꿀까요?"
        intent, _, _, _ = rt(nav("408호", reply), "408호로 가자", dialog="navigating", active="남자 화장실")
        assert intent.matched_destination_id == "r408" and intent.reply == reply

    def test_confirmed_navigate_stays_silent(self, rt):
        intent, _, _, _ = rt(nav("사회복지", "출발합니다", need_confirm=False), "네", dialog="confirming")
        assert intent.intent == "navigate" and intent.reply == ""


class TestShortAnswerInAnswerStates:
    QUESTION = {"intent": "question", "reply": "어디로 안내해드릴까요?", "need_confirm": False, "confidence": 0.6}

    @pytest.mark.parametrize("heard", ["그래", "그래.", "네", "응", "좋아"])
    def test_yes_to_approach_question_is_affirm(self, rt, heard):
        intent, _, _, info = rt(self.QUESTION, heard, dialog="awaiting_user")
        assert intent.intent == "affirm" and intent.reply == "" and intent.need_confirm is False
        assert "짧은답" in info["src"]

    def test_no_is_deny(self, rt):
        intent, _, _, _ = rt(self.QUESTION, "아니", dialog="asking_next")
        assert intent.intent == "deny" and intent.reply == ""

    @pytest.mark.parametrize("dialog", ["confirming", "asking_next", "asking_wait_time"])
    def test_all_answer_states(self, rt, dialog):
        intent, _, _, _ = rt(self.QUESTION, "그래", dialog=dialog)
        assert intent.intent == "affirm"

    def test_not_while_nobody_asked(self, rt):
        intent, _, _, info = rt(self.QUESTION, "그래", dialog="idle")
        assert intent.intent == "question" and info["src"] == "llm"

    def test_longer_sentence_stays_with_the_model(self, rt):
        intent, _, _, _ = rt(self.QUESTION, "그래, 너무 먼가?", dialog="awaiting_user")
        assert intent.intent == "question"

    def test_model_answer_intents_are_not_touched(self, rt):
        intent, _, _, _ = rt({"intent": "wait", "wait_minutes": 10, "reply": "", "confidence": 0.9}, "그래",
                             dialog="asking_wait_time")
        assert intent.intent == "wait" and intent.wait_minutes == 10


class TestChoiceQuestionWording:
    ODD = "남자 화장실로인지 여자 화장실로인지요, 남자 화장실과 여자 화장실 중 어디로 안내해드릴까요?"

    def test_guess_with_odd_words_gets_fixed_question(self, rt):
        intent, _, _, info = rt(nav("남자 화장실", self.ODD), "배아파요")
        assert intent.intent == "clarify" and intent.reply == CHOOSE and "고르기" in info["src"]

    def test_model_clarify_naming_both_gets_fixed_question(self, rt):
        intent, _, _, _ = rt({"intent": "clarify", "reply": "남자 화장실과 여자 화장실 중 어디로 안내해드릴까요?",
                              "need_confirm": False, "confidence": 0.8}, "나 배 아파", dialog="waiting")
        assert intent.intent == "clarify" and intent.reply == CHOOSE

    def test_other_clarify_question_is_kept(self, rt):
        intent, _, _, _ = rt({"intent": "clarify", "reply": "어디로 가고 싶으세요?", "need_confirm": False,
                              "confidence": 0.5}, "음 거기", dialog="idle")
        assert intent.reply == "어디로 가고 싶으세요?"

"""소리 모드 남녀 되묻기 — 여러 곳에 걸린 별칭만 말했는데 모델이 한쪽을 짐작하면 고르게 묻는다(2026-10-21 실기).

10-21 10:24 "화장실로 가자"에 Realtime 이 남자 화장실을 바로 제안했고, 10:37 주행 중 "나 화장실 가고 싶어"에는
"여자 화장실로 바꿀까요"라고 한쪽만 물었다. 글자 모드의 로컬 규칙(_shared_alias_tie)은 소리 모드를 거치지 않는다.
목적지는 시연 지도(map_1021_071412) 그대로다.
"""
import pytest

from src import langchain_intent_parser as parser
from src.langchain_intent_parser import parse_intent_audio
from src.realtime_intent import RealtimeResult
from src.schema import DestinationData, RobotState

SOCIAL = DestinationData(id="social", name="사회복지", aliases=["사회복지", "장애인", "서류", "장애인 서류", "복지"],
                         confirm_prompt="사회복지로 안내해드릴까요?")
R407 = DestinationData(id="r407", name="407호", aliases=["407호"], confirm_prompt="407호로 안내해드릴까요?")
MEN = DestinationData(id="men", name="남자 화장실",
                      aliases=["남자 화장실", "남자", "화장실", "남자화장실", "대변", "소변", "배아파", "아파"],
                      confirm_prompt="남자 화장실로 안내해드릴까요?")
WOMEN = DestinationData(id="women", name="여자 화장실",
                        aliases=["여자 화장실", "여자", "화장실", "여자화장실", "배아파", "아파", "대변", "소변"],
                        confirm_prompt="여자 화장실로 안내해드릴까요?")
DESTS = [SOCIAL, R407, MEN, WOMEN]
PCM = b"\x00\x00" * 1600
CHOOSE = "남자 화장실과 여자 화장실 중 어디로 갈까요?"


class FakeRT:
    def __init__(self, draft, heard):
        self.draft, self.heard = draft, heard

    def ask(self, pcm, history, instructions):
        return RealtimeResult(draft=dict(self.draft), heard_text=self.heard, latency_sec=0.7, usage={})


@pytest.fixture
def rt(monkeypatch):
    def use(draft, heard):
        monkeypatch.setattr(parser, "get_realtime_client", lambda: FakeRT(draft, heard))
    return use


def nav(name, need_confirm=True, reply=None):
    return {"intent": "navigate", "destination_candidate": name, "need_confirm": need_confirm,
            "reply": reply if reply is not None else f"{name}로 안내해드릴까요?", "confidence": 0.8}


def state(dialog="idle", active=""):
    return RobotState(dialog_state=dialog, active_destination=active)


class TestGuessBecomesChoice:
    def test_toilet_only_guess_becomes_choice(self, rt):
        rt(nav("남자 화장실"), "화장실로 가자")
        intent, _, _, info = parse_intent_audio(PCM, DESTS, robot_state=state())
        assert intent.intent == "clarify"
        assert intent.matched_destination_id == "" and intent.destination_candidate is None
        assert intent.reply == CHOOSE
        assert "고르기" in info["src"]

    def test_stomach_ache_guess_becomes_choice(self, rt):
        rt(nav("남자 화장실"), "나 배 아파")
        intent, _, _, _ = parse_intent_audio(PCM, DESTS, robot_state=state())
        assert intent.intent == "clarify" and intent.reply == CHOOSE

    def test_clarify_naming_one_side_while_driving(self, rt):
        rt({"intent": "clarify", "reply": "지금 사회복지로 가는 중이에요. 여자 화장실로 바꿀까요", "confidence": 0.9},
           "나 화장실 가고 싶어")
        intent, _, _, info = parse_intent_audio(PCM, DESTS, robot_state=state("navigating", "사회복지"))
        assert intent.intent == "clarify" and intent.reply == CHOOSE
        assert "고르기" in info["src"]


class TestLeftAlone:
    def test_full_name_is_kept(self, rt):
        rt(nav("남자 화장실"), "남자 화장실")
        intent, _, _, info = parse_intent_audio(PCM, DESTS, robot_state=state())
        assert intent.intent == "navigate" and intent.matched_destination_id == "men"
        assert info["src"] == "llm"

    def test_full_name_after_no_is_kept(self, rt):
        rt(nav("남자 화장실"), "아니, 남자 화장실.")
        intent, _, _, _ = parse_intent_audio(PCM, DESTS, robot_state=state("confirming"))
        assert intent.matched_destination_id == "men"

    def test_word_only_one_side_has_is_kept(self, rt):
        rt(nav("남자 화장실"), "남자 쪽으로 가자")
        intent, _, _, _ = parse_intent_audio(PCM, DESTS, robot_state=state())
        assert intent.matched_destination_id == "men"

    def test_accepting_the_asked_place_is_kept(self, rt):
        # "남자 화장실로 안내해드릴까요?"에 "응 화장실 가자" — 물어 둔 그곳의 승낙이다.
        rt(nav("남자 화장실", need_confirm=False, reply=""), "응 화장실 가자")
        intent, _, _, _ = parse_intent_audio(PCM, DESTS, robot_state=state("confirming"))
        assert intent.intent == "navigate" and intent.matched_destination_id == "men"

    def test_model_already_asks_both(self, rt):
        asked = "남자 화장실과 여자 화장실 중 어디로 안내해드릴까요?"
        rt({"intent": "clarify", "reply": asked, "confidence": 0.8}, "화장실 가자")
        intent, _, _, info = parse_intent_audio(PCM, DESTS, robot_state=state())
        assert intent.reply == asked and info["src"] == "llm"

    def test_unique_alias_destination_untouched(self, rt):
        rt(nav("사회복지"), "장애인 서류 내는 곳")
        intent, _, _, info = parse_intent_audio(PCM, DESTS, robot_state=state())
        assert intent.matched_destination_id == "social" and info["src"] == "llm"

    def test_clarify_during_approach_question_untouched(self, rt):
        # 접근 질문 중 되묻기는 미션이 "안내를 받으시겠어요?"로 다시 묻는다 — 이 규칙이 수락으로 바꾸지 않는다.
        rt({"intent": "clarify", "reply": "여자 화장실로 갈까요?", "confidence": 0.5}, "화장실")
        intent, _, _, _ = parse_intent_audio(PCM, DESTS, robot_state=state("awaiting_user"))
        assert intent.intent == "clarify" and intent.reply == "여자 화장실로 갈까요?"

    def test_navigate_guess_during_approach_becomes_plain_accept(self, rt):
        # 접근 질문 중 고르기는 _ask_which 규칙대로 목적지 없는 수락 — 돌아선 뒤 미션이 목적지를 묻는다.
        rt(nav("남자 화장실"), "응 화장실 가고 싶어")
        intent, _, _, _ = parse_intent_audio(PCM, DESTS, robot_state=state("awaiting_user"))
        assert intent.intent == "navigate" and intent.matched_destination_id == "" and intent.reply == ""

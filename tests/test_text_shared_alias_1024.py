"""글자 경로(⑬ 오프라인·⑫ 예비)에도 '여러 곳에 걸린 별칭만 말하면 고르기 질문' 규칙(2026-10-24 사용자)."""
import pytest

from src import langchain_intent_parser as parser
from src.langchain_intent_parser import _IntentDraft
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


def finalize(user_text, draft, dialog="waiting", local=True):
    return parser._finalize(_IntentDraft(**draft), DESTS, user_text=user_text, local=local,
                            robot_state=RobotState(dialog_state=dialog))


GUESS_MEN = {"intent": "navigate", "destination_candidate": "남자 화장실", "need_confirm": True,
             "reply": "남자 화장실로 안내해드릴까요?", "confidence": 0.8}


@pytest.mark.parametrize("local", [True, False])
@pytest.mark.parametrize("said", ["나 배 아파.", "나 배아파", "화장실 가고 싶어"])
def test_shared_alias_guess_becomes_choice(said, local):
    out = finalize(said, GUESS_MEN, local=local)
    assert out.intent == "clarify" and out.reply == CHOOSE
    assert out.matched_destination_id == "" and out.destination_candidate is None and out.need_confirm is False


def test_named_place_still_goes():
    out = finalize("남자 화장실로 가 줘", GUESS_MEN)
    assert out.intent == "navigate" and out.matched_destination_id == "men" and out.need_confirm is True


def test_social_documents_still_go_to_social():
    draft = {"intent": "navigate", "destination_candidate": "사회복지", "need_confirm": True,
             "reply": "사회복지로 안내해드릴까요?", "confidence": 0.85}
    out = finalize("나 장애인 서류 내야 해", draft, dialog="idle")
    assert out.intent == "navigate" and out.matched_destination_id == "social"


def test_approach_answer_is_not_turned_into_a_question():
    out = finalize("배 아파", GUESS_MEN, dialog="awaiting_user")
    assert out.intent == "navigate" and out.matched_destination_id == "" and out.reply == ""


from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from src.langchain_intent_parser import parse_intent  # noqa: E402


class _Manager:
    """LLM 을 부르면 기록만 남기고 unknown 을 돌려준다 — 규칙이 LLM 앞에서 답했는지 본다."""
    has_local = True

    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return _IntentDraft(intent="unknown", reply="", confidence=0.1)


@pytest.fixture
def llm(monkeypatch):
    m = _Manager()
    monkeypatch.setattr(parser, "get_backend_manager", lambda: m)
    return m


ASK_MIN = "몇 분쯤 걸리실까요?"
ARRIVE_WC = "남자 화장실에 도착했습니다. 남자 화장실은 왼쪽에 있습니다. 다녀오시는 동안 여기서 기다릴까요?"
ASK_END = "여기까지 안내를 마칠까요?"


@pytest.mark.parametrize("said,minutes", [("한 십 분 정도.", 10), ("삼십 분.", 30), ("오 분이면 돼", 5), ("반시간", 30)])
def test_wait_minutes_read_without_llm(llm, said, minutes):
    out = parse_intent(said, DESTS, history=[AIMessage(ASK_MIN)], robot_state=RobotState(dialog_state="asking_wait_time"))
    assert out.intent == "wait" and out.wait_minutes == minutes and out.reply == "" and llm.calls == 0


def test_wait_answer_without_minutes_goes_to_llm(llm):
    parse_intent("좀 걸려", DESTS, history=[AIMessage(ASK_MIN)], robot_state=RobotState(dialog_state="asking_wait_time"))
    assert llm.calls == 1


@pytest.mark.parametrize("said", ["필요 없어.", "필요없어요", "안 기다려도 돼", "됐어요."])
def test_no_need_to_wait_is_deny(llm, said):
    out = parse_intent(said, DESTS, history=[AIMessage(ARRIVE_WC)], robot_state=RobotState(dialog_state="asking_next"))
    assert out.intent == "deny" and out.reply == "" and llm.calls == 0


def test_no_need_after_finish_question_is_left_to_llm(llm):
    parse_intent("필요 없어", DESTS, history=[AIMessage(ARRIVE_WC), HumanMessage("아니"), AIMessage(ASK_END)],
                 robot_state=RobotState(dialog_state="asking_next"))
    assert llm.calls == 1


def test_no_need_when_nobody_asked_is_left_to_llm(llm):
    parse_intent("필요 없어", DESTS, history=[], robot_state=RobotState(dialog_state="idle"))
    assert llm.calls == 1

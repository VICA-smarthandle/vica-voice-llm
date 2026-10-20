"""두 곳 이상에 똑같이 걸리는 목적지 말은 한쪽을 고르지 않고 되묻는다(2026-10-11 사용자 결정 4번)."""
from src import langchain_intent_parser as parser
from src.destination_matcher import match_destination, tied_destinations
from src.langchain_intent_parser import _IntentDraft, parse_intent_audio
from src.realtime_intent import RealtimeResult
from src.replies import choose_between
from src.schema import DestinationData
from src.tts_queue import request_for_intent

DESTS = [
    DestinationData(id="m", name="남자 화장실", aliases=["남자 화장실", "남자", "화장실", "배 아파"]),
    DestinationData(id="w", name="여자 화장실", aliases=["여자 화장실", "화장실", "여자", "배 아파"]),
    DestinationData(id="r7", name="407호", aliases=["407호", "사무실"]),
    DestinationData(id="r8", name="408호", aliases=["408호", "사무실"], is_approachable=False),
]
PCM = b"\x00\x00" * 1600


class FakeRT:
    def __init__(self, draft):
        self.draft = draft

    def ask(self, pcm, history, instructions):
        return RealtimeResult(draft=dict(self.draft), heard_text="화장실 가고 싶어", latency_sec=0.5, usage={})


def test_ties():
    assert [d.id for d in tied_destinations("화장실", DESTS)] == ["m", "w"]
    assert [d.id for d in tied_destinations("배 아파", DESTS)] == ["m", "w"]
    assert [d.id for d in tied_destinations("남자 화장실", DESTS)] == ["m"]
    assert [d.id for d in tied_destinations("사무실", DESTS)] == ["r7"]
    assert match_destination("사무실", DESTS).id == "r7"


def test_choice_question_words():
    assert choose_between(["남자 화장실", "여자 화장실"]) == "남자 화장실과 여자 화장실 중 어디로 갈까요?"


def _audio(draft, monkeypatch):
    monkeypatch.setattr(parser, "get_realtime_client", lambda: FakeRT(draft))
    return parse_intent_audio(PCM, DESTS)[0]


def test_audio_tie_becomes_a_question_keeping_the_model_words(monkeypatch):
    reply = "남자 화장실과 여자 화장실 중 어디로 안내해드릴까요?"
    out = _audio({"intent": "navigate", "destination_candidate": "화장실", "need_confirm": True,
                  "reply": reply, "confidence": 0.9}, monkeypatch)
    assert out.intent == "clarify" and out.matched_destination_id == "" and out.need_confirm is False
    assert out.reply == reply and request_for_intent(out)


def test_audio_tie_with_a_guess_is_replaced(monkeypatch):
    out = _audio({"intent": "navigate", "destination_candidate": "화장실", "need_confirm": True,
                  "reply": "여자 화장실로 안내해드릴까요?", "confidence": 0.9}, monkeypatch)
    assert out.intent == "clarify" and out.reply == choose_between(["남자 화장실", "여자 화장실"])


def test_audio_exact_name_still_goes(monkeypatch):
    out = _audio({"intent": "navigate", "destination_candidate": "여자 화장실", "need_confirm": True,
                  "reply": "여자 화장실로 안내해드릴까요?", "confidence": 0.9}, monkeypatch)
    assert out.intent == "navigate" and out.matched_destination_id == "w"


def test_text_path_tie_becomes_a_question():
    out = parser._finalize(_IntentDraft(intent="navigate", destination_candidate="화장실"), DESTS)
    assert out.intent == "clarify" and out.matched_destination_id == ""
    assert out.reply == "남자 화장실과 여자 화장실 중 어디로 갈까요?"


def test_approach_answer_with_a_tie_stays_an_acceptance(monkeypatch):
    from src.schema import RobotState
    draft = _IntentDraft(intent="navigate", destination_candidate="화장실", need_confirm=True, reply="")
    for state in ("awaiting_user", "turning"):
        out = parser.deliver_audio_draft(draft, DESTS, state)
        assert out.intent == "navigate" and out.matched_destination_id == "" and out.reply == ""
        out = parser._finalize(_IntentDraft(intent="navigate", destination_candidate="화장실"), DESTS,
                               robot_state=RobotState(dialog_state=state))
        assert out.intent == "navigate" and out.matched_destination_id == ""


def test_choice_question_with_an_exact_name_sends_no_destination():
    reply = "남자 화장실과 여자 화장실 중 어디로 안내해드릴까요?"
    draft = _IntentDraft(intent="navigate", destination_candidate="남자 화장실", need_confirm=True, reply=reply)
    out = parser.deliver_audio_draft(draft, DESTS, "waiting")
    assert out.intent == "clarify" and out.matched_destination_id == "" and out.reply == reply
    assert request_for_intent(out)


def test_change_question_while_guiding_is_kept():
    reply = "지금 407호로 가는 중이에요. 남자 화장실로 바꿀까요?"
    draft = _IntentDraft(intent="navigate", destination_candidate="남자 화장실", need_confirm=True, reply=reply)
    out = parser.deliver_audio_draft(draft, DESTS, "navigating", "407호")
    assert out.intent == "navigate" and out.matched_destination_id == "m"


def test_single_place_confirm_is_kept():
    draft = _IntentDraft(intent="navigate", destination_candidate="여자 화장실", need_confirm=True,
                         reply="여자 화장실로 안내해드릴까요?")
    assert parser.deliver_audio_draft(draft, DESTS, "idle").matched_destination_id == "w"


def test_choice_question_during_the_approach_question_stays_an_acceptance():
    draft = _IntentDraft(intent="navigate", destination_candidate="남자 화장실", need_confirm=True,
                         reply="남자 화장실과 여자 화장실 중 어디로 안내해드릴까요?")
    out = parser.deliver_audio_draft(draft, DESTS, "awaiting_user")
    assert out.intent == "navigate" and out.matched_destination_id == "" and out.reply == ""

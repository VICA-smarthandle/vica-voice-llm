"""도착 후 대화 intent — wait / finish (LLM 판단 + 코드의 시간 추출)."""
import pytest

from src.langchain_intent_parser import _finalize, _IntentDraft, parse_wait_minutes
from src.schema import DestinationData

DEST = DestinationData(id="d1", name="화장실", aliases=["별빛관 화장실"])


class TestParseWaitMinutes:
    """한국어 시간 표현 -> 분. 없으면 None. 상한은 여기서 안 건다(Mission 몫)."""

    @pytest.mark.parametrize("text,minutes", [
        ("20분", 20), ("이십 분", 20), ("삼십분만", 30), ("10분만 기다려", 10),
        ("반시간", 30), ("5분", 5), ("한 시간", 60), ("두 시간", 120),
        ("300분", 300),
        ("십오 분", 15), ("이십오 분만", 25),
        ("5분에서 10분", 15), ("십 분에서 십오 분.", 23),
        ("10~20분", 30), ("한 5분에서 10분?", 15),
    ])
    def test_extracts(self, text, minutes):
        assert parse_wait_minutes(text) == minutes

    @pytest.mark.parametrize("text", ["기다려줘", "여기 있어", "", "글쎄"])
    def test_no_number(self, text):
        assert parse_wait_minutes(text) is None


class TestFinalizeWait:
    """LLM 이 wait 로 분류한 초안의 후처리."""

    def test_wait_with_time_extracted_from_text(self):
        draft = _IntentDraft(intent="wait", reply="네 기다릴게요", confidence=0.9)
        r = _finalize(draft, [DEST], user_text="20분만 기다려줘")
        assert r.intent == "wait"
        assert r.reply == ""
        assert r.wait_minutes == 20
        assert r.need_confirm is False

    def test_wait_without_time_defers(self):
        draft = _IntentDraft(intent="wait", reply="", confidence=0.8)
        r = _finalize(draft, [DEST], user_text="여기서 좀 기다려")
        assert r.intent == "wait"
        assert r.wait_minutes == -1


class TestFinalizeFinish:
    def test_finish_reply_is_silenced(self):
        draft = _IntentDraft(intent="finish", reply="네 안녕히 가세요", confidence=0.9)
        r = _finalize(draft, [DEST], user_text="이제 됐어 고마워")
        assert r.intent == "finish"
        assert r.reply == ""
        assert r.need_confirm is False


class TestWaitTimeMerge:
    """시간 병합: 코드가 원문의 단일 숫자를 우선, 없으면 LLM 제안 채택."""

    def test_code_single_number_wins_over_llm(self):
        draft = _IntentDraft(intent="wait", confidence=0.9, wait_minutes=35)
        r = _finalize(draft, [DEST], user_text="20분만 기다려")
        assert r.wait_minutes == 20

    def test_code_computes_range_itself(self):
        draft = _IntentDraft(intent="wait", confidence=0.9, wait_minutes=13)
        r = _finalize(draft, [DEST], user_text="십 분에서 십오 분.")
        assert r.wait_minutes == 23

    def test_llm_value_used_when_no_number_at_all(self):
        draft = _IntentDraft(intent="wait", confidence=0.9, wait_minutes=15)
        r = _finalize(draft, [DEST], user_text="좀 있다가 올게")
        assert r.wait_minutes == 15

    def test_neither_defers_to_followup(self):
        draft = _IntentDraft(intent="wait", confidence=0.8, wait_minutes=None)
        r = _finalize(draft, [DEST], user_text="좀 기다려줘")
        assert r.wait_minutes == -1

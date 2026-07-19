"""규칙 기반 확인 응답 처리 테스트 (LLM 호출 없이 검증 가능한 경로만).

배경: 소형 로컬 모델(exaone3.5:2.4b)이 "네" 를 듣고도 is_confirmation 을
채우지 않아 확인 질문이 무한 반복되는 문제가 실기에서 확인됨 (2026-07-19).
짧은 긍정/부정은 LLM 을 거치지 않고 parse_intent 가 코드로 결정한다.
"""
from langchain_core.messages import AIMessage, HumanMessage

from src.langchain_intent_parser import (
    _normalize_short_reply,
    _pending_confirm_destination,
    parse_intent,
)
from src.schema import DestinationData

DEST = DestinationData(
    id="engineering_4f_room_407_prof_yoon_jiyoung_office",
    name="윤지영 교수님 사무실",
    confirm_prompt="윤지영 교수님 사무실로 안내해드릴까요?",
)
HISTORY = [
    HumanMessage("407호로 가주세요"),
    AIMessage(DEST.confirm_prompt),
]


class TestNormalize:
    def test_strips_punctuation_and_space(self):
        assert _normalize_short_reply("네.") == "네"
        assert _normalize_short_reply(" 네, 맞아요! ") == "네맞아요"


class TestPendingDetection:
    def test_finds_destination_from_last_ai_message(self):
        assert _pending_confirm_destination(HISTORY, [DEST]) is DEST

    def test_none_when_history_empty_or_unrelated(self):
        assert _pending_confirm_destination(None, [DEST]) is None
        assert _pending_confirm_destination([], [DEST]) is None
        unrelated = [HumanMessage("안녕"), AIMessage("안녕하세요!")]
        assert _pending_confirm_destination(unrelated, [DEST]) is None


class TestParseIntentShortcut:
    def test_affirmative_confirms_without_llm(self):
        result = parse_intent("네.", [DEST], history=HISTORY)
        assert result.intent == "navigate"
        assert result.matched_destination_id == DEST.id
        assert result.need_confirm is False
        assert "안내를 시작" in result.reply

    def test_various_affirmatives(self):
        for word in ("응", "그래요", "맞아요", "좋아", "가줘"):
            result = parse_intent(word, [DEST], history=HISTORY)
            assert result.need_confirm is False, word
            assert result.matched_destination_id == DEST.id, word

    def test_negative_clarifies(self):
        result = parse_intent("아니요", [DEST], history=HISTORY)
        assert result.intent == "clarify"
        assert result.matched_destination_id is None
        assert result.need_confirm is False

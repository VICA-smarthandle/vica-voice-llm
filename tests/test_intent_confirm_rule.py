"""규칙 기반 확인 응답 처리 테스트 (LLM 호출 없이 검증 가능한 경로만)."""
from langchain_core.messages import AIMessage, HumanMessage

from src.langchain_intent_parser import (
    _finalize,
    _IntentDraft,
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

    def test_negative_denies(self):
        """거절은 deny 다 (2026-09-02, 종전 clarify)."""
        result = parse_intent("아니요", [DEST], history=HISTORY)
        assert result.intent == "deny"
        assert result.matched_destination_id is None
        assert result.need_confirm is False
        assert result.reply == ""


class TestFinalizeConfirmationGate:
    """is_confirmation 은 코드가 아는 pending 과 일치할 때만 믿는다."""

    def _draft(self):
        return _IntentDraft(
            intent="navigate",
            destination_candidate=DEST.name,
            is_confirmation=True,
            confidence=0.9,
            reply="",
        )

    def test_no_pending_still_requires_confirm(self):
        result = _finalize(self._draft(), [DEST], pending=None)
        assert result.need_confirm is True
        assert result.reply == DEST.confirm_prompt

    def test_pending_mismatch_still_requires_confirm(self):
        other = DestinationData(
            id="starlight_1f_restroom",
            name="별빛관 1층 화장실",
            confirm_prompt="별빛관 1층 화장실로 안내해드릴까요?",
        )
        result = _finalize(self._draft(), [DEST, other], pending=other)
        assert result.need_confirm is True
        assert result.matched_destination_id == DEST.id

    def test_pending_match_confirms(self):
        result = _finalize(self._draft(), [DEST], pending=DEST)
        assert result.need_confirm is False
        assert "안내를 시작" in result.reply


class TestFirstWordShortcut:
    """첫 단어 판정(2026-09-01) — "응 화장실로 가자"가 LLM 없이 확정된다."""

    def test_affirm_prefix_with_tail_confirms(self):
        result = parse_intent("응 윤지영 교수님 사무실로 가자", [DEST], history=HISTORY)
        assert result.intent == "navigate"
        assert result.matched_destination_id == DEST.id
        assert result.need_confirm is False

    def test_negative_prefix_with_tail_declines(self):
        result = parse_intent("아니 거기 말고 다른 데", [DEST], history=HISTORY)
        assert result.intent == "deny"

    def test_hesitation_alone_confirms_but_prefix_does_not(self):
        """'음'·'어어'는 whisper 가 적은 "응"이라 단독이면 승낙이다. 그러나"""
        alone = parse_intent("음.", [DEST], history=HISTORY)
        assert alone.intent == "navigate"
        assert alone.need_confirm is False

        prefixed = parse_intent("음 아니야 다른 데 갈래", [DEST], history=HISTORY,
                                model="__no_llm__")
        assert prefixed.intent != "navigate"

    def test_word_starting_with_affirm_char_is_not_tripped(self):
        """"어디로 가?"는 "어" 접두라도 지름길에 안 걸려야 한다 (첫 단어"""
        result = parse_intent("어디로 가야 하지?", [DEST], history=HISTORY,
                              model="__no_llm__")
        assert result.intent != "navigate"


class TestCorrectionBeatsFirstWord:
    """사람은 말하다 마음을 바꾼다 — 첫 단어만 보면 그 정정을 놓친다."""

    def test_yes_then_no_is_deny(self):
        result = parse_intent("네. 아니. 아니. 아니.", [DEST], history=HISTORY)
        assert result.intent == "deny"
        assert result.need_confirm is False

    def test_yes_then_no_two_words(self):
        result = parse_intent("네 아니요", [DEST], history=HISTORY)
        assert result.intent == "deny"

    def test_plain_affirmative_with_tail_still_confirms(self):
        """정정이 아닌 평범한 승낙은 그대로 확정된다 — 방어가 과하면 안 된다."""
        result = parse_intent("응 윤지영 교수님 사무실로 가자", [DEST],
                              history=HISTORY)
        assert result.intent == "navigate"
        assert result.need_confirm is False

    def test_negative_inside_a_word_is_not_a_correction(self):
        """'아니고'는 부정어 목록에 없다 — 토막 전체 일치라 오폭하지 않는다."""
        result = parse_intent("네 화장실 아니고 안내소로", [DEST],
                              history=HISTORY, model="__no_llm__")
        assert result.intent != "deny"

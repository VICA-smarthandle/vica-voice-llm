"""affirm/deny 발행 시험 — 사람 접근 질문의 짧은 답 (LLM 없이 검증 가능한 경로)."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.langchain_intent_parser import _finalize, _IntentDraft, parse_intent
from src.replies import CANCEL_CONFIRM
from src.schema import DestinationData

DEST = DestinationData(
    id="starlight_1f_restroom",
    name="별빛관 1층 화장실",
    confirm_prompt="별빛관 1층 화장실로 안내해드릴까요?",
)


class TestShortAnswerShortcut:
    """확인 대기가 없는 짧은 긍/부정은 LLM 없이 0초에 affirm/deny 가 된다."""

    @pytest.mark.parametrize("word", ["네", "응", "그래요", "좋아요"])
    def test_bare_affirmative_becomes_affirm(self, word):
        result = parse_intent(word, [DEST], history=[])
        assert result.intent == "affirm"
        assert result.reply == ""
        assert result.need_confirm is False
        assert not result.matched_destination_id
        assert result.confidence == 1.0

    @pytest.mark.parametrize("word", ["아니요", "아니", "싫어요"])
    def test_bare_negative_becomes_deny(self, word):
        result = parse_intent(word, [DEST], history=[])
        assert result.intent == "deny"
        assert result.reply == ""
        assert result.need_confirm is False

    def test_cancel_word_stays_cancel(self):
        """'취소'는 부정 목록에도 있지만 취소 직행이 먼저다 — deny 로 새면 안 된다."""
        result = parse_intent("취소", [DEST], history=[])
        assert result.intent == "cancel"
        assert result.reply == CANCEL_CONFIRM

    def test_pause_word_stays_pause(self):
        result = parse_intent("잠깐만", [DEST], history=[])
        assert result.intent == "pause"


class TestExistingConfirmsStillWin:
    """음성이 직접 던진 확인 질문의 답은 기존 경로가 먼저다 (회귀 방지)."""

    def test_destination_confirm_beats_affirm(self):
        history = [HumanMessage("화장실로 가줘"), AIMessage(DEST.confirm_prompt)]
        result = parse_intent("네", [DEST], history=history)
        assert result.intent == "navigate"
        assert result.matched_destination_id == DEST.id

    def test_command_confirm_beats_affirm(self):
        history = [HumanMessage("취소해줘"), AIMessage(CANCEL_CONFIRM)]
        result = parse_intent("네", [DEST], history=history)
        assert result.intent == "cancel"


class TestLlmFallbackFinalize:
    """애매한 답("어… 부탁드려요")은 LLM 이 affirm/deny 로 분류한다."""

    def test_llm_affirm_draft_is_silenced(self):
        draft = _IntentDraft(intent="affirm", reply="알겠습니다, 안내할게요!",
                             confidence=0.9)
        result = _finalize(draft, [DEST])
        assert result.intent == "affirm"
        assert result.reply == ""
        assert result.need_confirm is False
        assert not result.matched_destination_id

    def test_llm_deny_draft_is_silenced(self):
        draft = _IntentDraft(intent="deny", reply="네, 알겠습니다.", confidence=0.8)
        result = _finalize(draft, [DEST])
        assert result.intent == "deny"
        assert result.reply == ""


class TestInstantUtterance:
    """접수 신호("확인할게요") 생략 판정 — 0초 지름길이면 신호가 군더더기다"""

    @pytest.mark.parametrize("word", [
        "네", "그래", "응", "좋아요",
        "아니요", "싫어요",
        "취소", "잠깐만",
        " 그래. ",
    ])
    def test_shortcut_words_are_instant(self, word):
        from src.langchain_intent_parser import is_instant_utterance
        assert is_instant_utterance(word) is True

    @pytest.mark.parametrize("text", [
        "화장실로 가자",
        "그래 가자",
        "다시 가자",
        "",
    ])
    def test_sentences_need_ack(self, text):
        from src.langchain_intent_parser import is_instant_utterance
        assert is_instant_utterance(text) is False


class TestWakeWordInsideWindow:
    """청취 창 안에서 "비카야"를 부른 경우 — LLM 없이 즉시 "네?"로 받는다."""

    @pytest.mark.parametrize("word", ["비카야", "피카야", "비까야", " 비카야? "])
    def test_bare_wake_word_gets_greeting(self, word):
        from src.replies import WAKE_GREETING
        result = parse_intent(word, [DEST], history=[])
        assert result.intent == "unknown"
        assert result.reply == WAKE_GREETING
        assert result.need_confirm is False

    def test_wake_word_plus_command_goes_to_normal_path(self):
        """"비카야, 화장실로 가자"는 지름길이 아니다 — 명령이 우선."""
        from src.langchain_intent_parser import _WAKE_WORDS
        from src.handle_mode import normalize_short_reply
        assert normalize_short_reply("비카야 화장실로 가자") not in _WAKE_WORDS

    def test_wake_word_is_instant(self):
        """접수 신호("확인할게요") 없이 바로 "네?"가 나가야 한다."""
        from src.langchain_intent_parser import is_instant_utterance
        assert is_instant_utterance("비카야") is True


class TestFollowupFilterLetsAnswersThrough:
    """재청취 기각(ros_node 2-1)이 **할 말을 든 판정**까지 삼키던 결함."""

    def test_decline_is_not_droppable_kind(self):
        """거절은 이제 deny 라 필터의 폐기 대상(unknown/clarify)이 아니다."""
        history = [HumanMessage("화장실로 가줘"), AIMessage(DEST.confirm_prompt)]
        result = parse_intent("아니", [DEST], history=history)
        assert result.intent == "deny"
        assert result.intent not in ("unknown", "clarify")

    def test_wake_greeting_is_marked_as_shortcut_reply(self):
        """호출 응답은 unknown 이지만 SHORTCUT_REPLIES 표지로 통과한다."""
        from src.langchain_intent_parser import SHORTCUT_REPLIES
        result = parse_intent("피카야", [DEST], history=[])
        assert result.intent == "unknown"
        assert result.reply in SHORTCUT_REPLIES

    def test_llm_chatter_reply_is_not_marked(self):
        """LLM 이 지어낸 대꾸는 표지가 없어 종전대로 버려진다 — 이 경계가"""
        from src.langchain_intent_parser import SHORTCUT_REPLIES
        assert "무슨 말씀이신지 잘 모르겠어요." not in SHORTCUT_REPLIES
        assert "" not in SHORTCUT_REPLIES


class TestWakeWordVariants2026_09_02:
    """실기에서 관측된 '비카야' 오전사 변형. 창이 열린 동안 호출 감지기가"""

    @pytest.mark.parametrize("word", ["미카야", "리카야", "비켜야.", "비кая."])
    def test_observed_variants_get_greeting(self, word):
        from src.replies import WAKE_GREETING
        result = parse_intent(word, [DEST], history=[])
        assert result.reply == WAKE_GREETING

    def test_variant_with_tail_is_not_a_wake_word(self):
        """"비켜야 해요"는 호출이 아니다 — 전체 일치라 꼬리가 붙으면 빠진다."""
        from src.langchain_intent_parser import _WAKE_WORDS
        from src.handle_mode import normalize_short_reply
        assert normalize_short_reply("비켜야 해요") not in _WAKE_WORDS

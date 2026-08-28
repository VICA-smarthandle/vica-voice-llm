"""affirm/deny 발행 시험 — 사람 접근 질문의 짧은 답 (LLM 없이 검증 가능한 경로).

계약 정본: vica_ros2_ws/src/vica_interfaces/msg/VicaIntent.msg 의 affirm/deny 절.
음성은 "직전 질문이 무엇인지" 판정하지 않는다 — 짧은 긍/부정을 affirm/deny 로
발행만 하고, 소비/무시는 상태를 가진 Mission 이 정한다 (아무 때나 보내도 안전).
reply 는 반드시 빈 문자열이다 — 수락/거절 발화는 Mission 몫, 채우면 두 번 말한다.
"""
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
        assert result.reply == ""            # 발화는 Mission 몫 (이중 발화 금지)
        assert result.need_confirm is False
        # 파이썬 스키마의 "없음"은 None 이다 — ROS 변환(ros_convert)에서 "" 가 된다
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
    """애매한 답("어… 부탁드려요")은 LLM 이 affirm/deny 로 분류한다.

    LLM 실호출은 정책상 자동화하지 않으므로 초안 후처리(_finalize)만 검증한다.
    """

    def test_llm_affirm_draft_is_silenced(self):
        draft = _IntentDraft(intent="affirm", reply="알겠습니다, 안내할게요!",
                             confidence=0.9)
        result = _finalize(draft, [DEST])
        assert result.intent == "affirm"
        assert result.reply == ""            # LLM 이 채워도 코드가 비운다
        assert result.need_confirm is False
        assert not result.matched_destination_id

    def test_llm_deny_draft_is_silenced(self):
        draft = _IntentDraft(intent="deny", reply="네, 알겠습니다.", confidence=0.8)
        result = _finalize(draft, [DEST])
        assert result.intent == "deny"
        assert result.reply == ""


class TestInstantUtterance:
    """접수 신호("확인할게요") 생략 판정 — 0초 지름길이면 신호가 군더더기다
    (2026-08-28 사용자: "그래" 뒤 확인할게요는 별로)."""

    @pytest.mark.parametrize("word", [
        "네", "그래", "응", "좋아요",          # 긍정 지름길
        "아니요", "싫어요",                    # 부정 지름길
        "취소", "잠깐만",                      # 취소·일시정지 지름길
        " 그래. ",                             # 문장부호·공백 무시
    ])
    def test_shortcut_words_are_instant(self, word):
        from src.langchain_intent_parser import is_instant_utterance
        assert is_instant_utterance(word) is True

    @pytest.mark.parametrize("text", [
        "화장실로 가자",                       # LLM 행 — 생각 시간 필요
        "그래 가자",                           # 목적지 확정 문장 — LLM 행
        "다시 가자",
        "",
    ])
    def test_sentences_need_ack(self, text):
        from src.langchain_intent_parser import is_instant_utterance
        assert is_instant_utterance(text) is False

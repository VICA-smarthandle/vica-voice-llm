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


class TestWakeWordInsideWindow:
    """청취 창 안에서 "비카야"를 부른 경우 — LLM 없이 즉시 "네?"로 받는다.

    창이 열린 동안 호출 감지기는 잠들어 있어 "비카야"가 발화로 전사돼
    LLM 까지 갔다 왔다(2026-08-29 분석). 밖에서 부른 것과 똑같이 느껴지게
    지름길로 답한다. reply "네?"는 ?로 끝나 재청취 창이 다시 열린다.
    """

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
    """재청취 기각(ros_node 2-1)이 **할 말을 든 판정**까지 삼키던 결함.

    9/1~9/2 실기: 확인 질문에 "아니"라고 답한 6회가 전부 증발했고
    ('재청취 기각(무의미): 아니. intent=clarify'), 창 안에서 부른 '피카야'의
    "네?"도 같은 자리에서 삼켜졌다. 필터가 intent 이름만 보고 reply 를 보지
    않았기 때문이다. 여기서는 필터가 보는 **입력 조건**을 못박는다 — 필터
    자체는 노드에 있어 ROS 없이 직접 부를 수 없다.
    """

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
        assert result.intent == "unknown"          # 폐기 대상 종류지만
        assert result.reply in SHORTCUT_REPLIES    # 표지가 있어 살아난다

    def test_llm_chatter_reply_is_not_marked(self):
        """LLM 이 지어낸 대꾸는 표지가 없어 종전대로 버려진다 — 이 경계가
        무너지면 잡담마다 로봇이 대꾸해 멘트 최소주의가 깨진다."""
        from src.langchain_intent_parser import SHORTCUT_REPLIES
        assert "무슨 말씀이신지 잘 모르겠어요." not in SHORTCUT_REPLIES
        assert "" not in SHORTCUT_REPLIES

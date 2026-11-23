"""cancel/pause/resume 제어 intent 테스트 (LLM 호출 없이 검증 가능한 경로만)."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.langchain_intent_parser import (_finalize, _IntentDraft, _shortcut_intent, is_instant_utterance,
                                         parse_intent)
from src.replies import (
    CANCEL_CONFIRM,
    COMMAND_DECLINED,
    PAUSE_ACK,
    RESUME_CONFIRM,
)
from src.schema import DestinationData

DEST = DestinationData(
    id="starlight_1f_restroom",
    name="별빛관 1층 화장실",
    confirm_prompt="별빛관 1층 화장실로 안내해드릴까요?",
)


class TestCancelRule:
    def test_bare_cancel_asks_confirm_without_llm(self):
        result = parse_intent("취소해줘", [DEST], history=[])
        assert result.intent == "cancel"
        assert result.need_confirm is True
        assert result.reply == CANCEL_CONFIRM

    def test_cancel_during_destination_confirm_stays_negative(self):
        history = [HumanMessage("화장실로 가줘"), AIMessage(DEST.confirm_prompt)]
        result = parse_intent("취소", [DEST], history=history)
        assert result.intent == "deny"


class TestCommandConfirmFlow:
    def test_yes_after_cancel_question_confirms(self):
        history = [HumanMessage("취소해줘"), AIMessage(CANCEL_CONFIRM)]
        result = parse_intent("네", [DEST], history=history)
        assert result.intent == "cancel"
        assert result.need_confirm is False

    def test_no_after_cancel_question_goes_to_the_mission(self):
        """취소 확인의 "아니요"는 미션이 받아 "안내를 계속하겠습니다."라고 답한다(2026-10-08)."""
        history = [HumanMessage("취소해줘"), AIMessage(CANCEL_CONFIRM)]
        result = parse_intent("아니요", [DEST], history=history)
        assert result.intent == "deny"
        assert result.reply == ""
        assert result.need_confirm is False

    def test_no_after_resume_question_is_still_answered_here(self):
        """"다시 출발할까요?"는 음성이 쥔 질문이라 미션이 모른다 — 대답은 지금처럼 여기서."""
        history = [AIMessage(RESUME_CONFIRM)]
        result = parse_intent("아니요", [DEST], history=history)
        assert result.intent == "unknown"
        assert result.reply == COMMAND_DECLINED

    def test_yes_after_resume_question_confirms(self):
        history = [AIMessage(RESUME_CONFIRM)]
        result = parse_intent("응", [DEST], history=history)
        assert result.intent == "resume"
        assert result.need_confirm is False


class TestFinalizeCommandGate:
    def _draft(self, intent):
        return _IntentDraft(intent=intent, reply="")

    def test_llm_command_proposal_always_asks(self):
        for intent, phrase in (
            ("cancel", CANCEL_CONFIRM),
            ("resume", RESUME_CONFIRM),
        ):
            result = _finalize(self._draft(intent), [DEST])
            assert result.need_confirm is True, intent
            assert result.reply == phrase, intent

    def test_reconfirm_via_llm_when_command_pending(self):
        result = _finalize(self._draft("resume"), [DEST], pending_command="resume")
        assert result.need_confirm is False

    def test_other_command_while_pending_still_asks(self):
        result = _finalize(self._draft("cancel"), [DEST], pending_command="resume")
        assert result.need_confirm is True
        assert result.reply == CANCEL_CONFIRM


class TestPauseImmediate:
    """pause 는 서는 방향이라 되묻지 않는다 (2026-08-15 현장 시험 결정)."""

    def test_bare_pause_word_requests_immediately(self):
        result = parse_intent("잠깐만", [DEST], history=[])
        assert result.intent == "pause"
        assert result.need_confirm is False
        assert result.reply == PAUSE_ACK

    def test_llm_pause_proposal_needs_no_confirm(self):
        result = _finalize(_IntentDraft(intent="pause", reply=""), [DEST])
        assert result.need_confirm is False
        assert result.reply == PAUSE_ACK


class TestResumeConfirmGate:
    def test_resume_proposal_is_not_forwarded(self):
        """확인 필요 딱지가 붙은 resume 제안은 미션에 보내지 않는다."""
        from src.schema import VicaIntent, should_forward_intent

        proposal = VicaIntent(intent="resume", need_confirm=True,
                              reply=RESUME_CONFIRM, confidence=0.9)
        assert should_forward_intent(proposal) is False

    def test_confirmed_resume_is_forwarded(self):
        from src.schema import VicaIntent, should_forward_intent

        confirmed = VicaIntent(intent="resume", need_confirm=False,
                               reply="", confidence=1.0)
        assert should_forward_intent(confirmed) is True

    def test_cancel_proposal_still_forwarded(self):
        """취소는 미션 안에 확인 게이트가 있다(실주행 검증 경로) — 보낸다."""
        from src.schema import VicaIntent, should_forward_intent

        proposal = VicaIntent(intent="cancel", need_confirm=True,
                              reply=CANCEL_CONFIRM, confidence=0.9)
        assert should_forward_intent(proposal) is True

    def test_confirm_answer_reply_is_silent(self):
        """'네' 확정의 reply 는 빈 문자열 — 결과 발화('다시 출발합니다',"""
        history = [AIMessage(RESUME_CONFIRM)]
        result = parse_intent("네", [DEST], history=history)
        assert result.intent == "resume"
        assert result.need_confirm is False
        assert result.reply == ""


class TestPauseResumeTextRules:
    """일시정지·재개 말은 LLM 없이 읽는다(3-나-1, 설계 2026-11-18 4.2). 11-18 실기: 로컬 LLM 이 이 말들을 407호 이동으로 읽었다."""

    @pytest.mark.parametrize("said", ["잠깐", "잠깐.", "잠시", "잠깐만", "잠시만요"])
    def test_short_pause_words(self, said):
        result = _shortcut_intent(said, [], [DEST])
        assert (result.intent, result.need_confirm, result.reply) == ("pause", False, PAUSE_ACK)
        assert is_instant_utterance(said)

    @pytest.mark.parametrize("said", ["잠깐", "잠시"])
    def test_bare_pause_words_may_be_a_time_answer(self, said):
        assert _shortcut_intent(said, [], [DEST], dialog_state="asking_wait_time", local=True) is None
        assert _shortcut_intent("잠깐만", [], [DEST], dialog_state="asking_wait_time", local=True).intent == "pause"

    @pytest.mark.parametrize("said", [
        "다시 가자.", "다시 가", "다시 가줘", "다시 출발", "다시 출발.", "다시 출발해", "다시 출발해줘", "다시 출발하자",
        "출발", "출발해", "출발해줘", "출발하자", "계속 가자", "계속 가", "계속 가줘",
        "다시 가요", "다시 출발해요", "다시 출발해 주세요", "출발해요", "출발해 주세요", "계속 가요", "계속 가 주세요",
        "다시 가 주세요", "다시 가줘요", "다시 출발해줘요", "출발해줘요", "계속 가줘요",
    ])
    def test_resume_words_ask_before_moving(self, said):
        result = _shortcut_intent(said, [], [DEST], dialog_state="paused", local=True)
        assert (result.intent, result.need_confirm, result.reply) == ("resume", True, RESUME_CONFIRM)

    @pytest.mark.parametrize("dialog_state", ["", "idle", "navigating", "waiting", "asking_next", "paused_handle"])
    def test_resume_words_only_while_paused(self, dialog_state):
        for said in ("다시 가자.", "출발"):
            assert _shortcut_intent(said, [], [DEST], dialog_state=dialog_state, local=True) is None

    @pytest.mark.parametrize("said", ["다시 가자고 했잖아", "출발지가 어디야", "화장실로 다시 가자", "다시"])
    def test_longer_sentences_are_left_to_the_llm(self, said):
        assert _shortcut_intent(said, [], [DEST], dialog_state="paused", local=True) is None

    def test_resume_word_answers_the_open_resume_question(self):
        result = _shortcut_intent("다시 가자", [AIMessage(RESUME_CONFIRM)], [DEST], dialog_state="paused", local=True)
        assert (result.intent, result.need_confirm, result.reply) == ("resume", False, "")

    def test_resume_word_during_cancel_question_asks_anew(self):
        result = _shortcut_intent("다시 가자", [AIMessage(CANCEL_CONFIRM)], [DEST], dialog_state="paused", local=True)
        assert (result.intent, result.need_confirm, result.reply) == ("resume", True, RESUME_CONFIRM)

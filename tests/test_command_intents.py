"""cancel/pause/resume 제어 intent 테스트 (LLM 호출 없이 검증 가능한 경로만).

원칙: LLM 은 제어를 '제안'만 한다. 실행은 ① 코드가 확인 질문에 대한 "네"를
검증하고 ② Mission Manager 가 상태를 판정한 뒤다. 여기서는 ①을 검증한다.
확인 문구의 자가 트리거 여부는 tests/test_spoken_text.py 가 자동으로 훑는다.
"""
from langchain_core.messages import AIMessage, HumanMessage

from src.langchain_intent_parser import _finalize, _IntentDraft, parse_intent
from src.replies import (
    CANCEL_CONFIRM,
    COMMAND_DECLINED,
    PAUSE_CONFIRM,
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
        # 목적지 확인 중의 "취소"는 기존 NEGATIVES 경로(되묻기)가 우선이다.
        history = [HumanMessage("화장실로 가줘"), AIMessage(DEST.confirm_prompt)]
        result = parse_intent("취소", [DEST], history=history)
        assert result.intent == "clarify"


class TestCommandConfirmFlow:
    def test_yes_after_pause_question_confirms(self):
        history = [HumanMessage("잠깐 쉬었다 가자"), AIMessage(PAUSE_CONFIRM)]
        result = parse_intent("네", [DEST], history=history)
        assert result.intent == "pause"
        assert result.need_confirm is False

    def test_no_after_cancel_question_continues(self):
        history = [HumanMessage("취소해줘"), AIMessage(CANCEL_CONFIRM)]
        result = parse_intent("아니요", [DEST], history=history)
        assert result.intent == "unknown"
        assert result.reply == COMMAND_DECLINED
        assert result.need_confirm is False

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
            ("pause", PAUSE_CONFIRM),
            ("resume", RESUME_CONFIRM),
        ):
            result = _finalize(self._draft(intent), [DEST])
            assert result.need_confirm is True, intent
            assert result.reply == phrase, intent

    def test_reconfirm_via_llm_when_command_pending(self):
        result = _finalize(self._draft("pause"), [DEST], pending_command="pause")
        assert result.need_confirm is False

    def test_other_command_while_pending_still_asks(self):
        # "잠시 멈출까요?" 대기 중에 취소 제안이 오면 취소를 새로 되묻는다.
        result = _finalize(self._draft("cancel"), [DEST], pending_command="pause")
        assert result.need_confirm is True
        assert result.reply == CANCEL_CONFIRM

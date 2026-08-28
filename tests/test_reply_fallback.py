"""navigate reply 생략과 빈 reply 안전망 테스트 (LLM 호출 없음).

배경: navigate 의 확인 문구는 _finalize 가 confirm_prompt 로 갈아끼우므로 LLM 이
문장을 써 봐야 폐기된다. 그래서 프롬프트가 navigate 의 reply 를 비우게 지시해
생성 토큰을 줄인다 (Jetson 로컬 22tok/s 실측 기준 ~1초 이상 절감, 2026-08-15).
빈 reply 가 다른 intent 로 번지면 침묵이 되므로 _finalize 끝에서 고정 문구로
메운다 — 소리로만 상태를 아는 사용자에게 침묵은 최악이다.
"""
import pytest
from src.langchain_intent_parser import _finalize, _IntentDraft
from src.replies import ASK_DESTINATION, RETRY_PROMPT
from src.schema import DestinationData

DEST = DestinationData(
    id="starlight_1f_restroom",
    name="별빛관 1층 화장실",
    confirm_prompt="별빛관 1층 화장실로 안내해드릴까요?",
)


class TestNavigateEmptyReply:
    def test_empty_reply_becomes_confirm_prompt(self):
        draft = _IntentDraft(intent="navigate", destination_candidate=DEST.name, reply="")
        result = _finalize(draft, [DEST])
        assert result.need_confirm is True
        assert result.reply == DEST.confirm_prompt

    def test_unmatched_candidate_falls_back_to_ask(self):
        draft = _IntentDraft(intent="navigate", destination_candidate="없는 곳", reply="")
        result = _finalize(draft, [DEST])
        assert result.intent == "clarify"
        assert result.reply == ASK_DESTINATION


class TestEmptyReplySafetyNet:
    def test_clarify_empty_reply_asks_destination(self):
        draft = _IntentDraft(intent="clarify", reply="")
        result = _finalize(draft, [DEST])
        assert result.reply == ASK_DESTINATION

    def test_question_empty_reply_prompts_retry(self):
        draft = _IntentDraft(intent="question", reply="")
        result = _finalize(draft, [DEST])
        assert result.reply == RETRY_PROMPT

    def test_nonempty_reply_is_kept(self):
        draft = _IntentDraft(intent="question", reply="여기는 1층입니다.")
        result = _finalize(draft, [DEST])
        assert result.reply == "여기는 1층입니다."


class TestAckPool:
    """대기(접수) 멘트 풀 — 같은 말만 반복되면 지겹다(2026-08-28 사용자).

    무작위 선택은 ros_node 몫이고, 여기서는 풀 자체의 건강만 검사한다.
    """

    def test_pool_has_variety(self):
        from src.replies import ACK_LISTENING_POOL
        assert len(ACK_LISTENING_POOL) >= 3
        assert len(set(ACK_LISTENING_POOL)) == len(ACK_LISTENING_POOL)
        assert all(p.strip() for p in ACK_LISTENING_POOL)

    def test_pool_is_covered_by_emergency_scan(self):
        """all_phrases 가 묶음을 못 펴면 긴급어 검사망에서 빠진다."""
        from src.replies import ACK_LISTENING_POOL, all_phrases
        collected = set(all_phrases().values())
        for phrase in ACK_LISTENING_POOL:
            assert phrase in collected


class TestExpectsAnswer:
    """"방금 한 말이 질문인가" 판정 — 재청취 창을 여는 새 열쇠.

    LLM 이 자유 생성한 질문("무엇을 도와드릴까요?", intent=unknown)에
    창이 안 열려 사용자가 "비카야"를 다시 불러야 했다 (2026-08-28 실측).
    """

    @pytest.mark.parametrize("reply", [
        "무엇을 도와드릴까요?",
        "어디로 안내해 드릴까요?",
        "요청이 확인되지 않았습니다. 다시 말씀해 주세요.",
        "  어디로 갈까요?  ",
    ])
    def test_questions_expect_answer(self, reply):
        from src.replies import expects_answer
        assert expects_answer(reply) is True

    @pytest.mark.parametrize("reply", [
        "안내를 시작하겠습니다.",
        "죄송합니다. 지금은 요청을 처리할 수 없어요.",
        "",
    ])
    def test_statements_do_not(self, reply):
        from src.replies import expects_answer
        assert expects_answer(reply) is False

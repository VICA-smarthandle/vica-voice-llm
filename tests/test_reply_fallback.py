"""navigate reply 생략과 빈 reply 안전망 테스트 (LLM 호출 없음).

배경: navigate 의 확인 문구는 _finalize 가 confirm_prompt 로 갈아끼우므로 LLM 이
문장을 써 봐야 폐기된다. 그래서 프롬프트가 navigate 의 reply 를 비우게 지시해
생성 토큰을 줄인다 (Jetson 로컬 22tok/s 실측 기준 ~1초 이상 절감, 2026-08-15).
빈 reply 가 다른 intent 로 번지면 침묵이 되므로 _finalize 끝에서 고정 문구로
메운다 — 소리로만 상태를 아는 사용자에게 침묵은 최악이다.
"""
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

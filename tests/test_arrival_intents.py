"""도착 후 대화 intent — wait / finish (LLM 판단 + 코드의 시간 추출).

계약 정본: vica_ros2_ws vica_interfaces/msg/VicaIntent.msg 의 wait/finish 절.
발화 의도가 맥락에 좌우돼("그만 좀 물어봐" != 종료) 지름길을 두지 않는다 —
LLM 이 wait/finish 를 판단하고, 시간 숫자만 코드가 원문에서 뽑는다
(2026-08-30 사용자 결정). LLM 실호출은 자동화하지 않으므로 초안 후처리
(_finalize)만 검증한다 (test_affirm_deny 의 TestLlmFallbackFinalize 와 같은 패턴).
"""
import pytest

from src.langchain_intent_parser import _finalize, _IntentDraft, parse_wait_minutes
from src.schema import DestinationData

DEST = DestinationData(id="d1", name="화장실", aliases=["별빛관 화장실"])


class TestParseWaitMinutes:
    """한국어 시간 표현 -> 분. 없으면 None. 상한은 여기서 안 건다(Mission 몫)."""

    @pytest.mark.parametrize("text,minutes", [
        ("20분", 20), ("이십 분", 20), ("삼십분만", 30), ("10분만 기다려", 10),
        ("반시간", 30), ("5분", 5), ("한 시간", 60), ("두 시간", 120),
        ("300분", 300),   # 상한을 넘겨도 그대로 — 깎기는 Mission
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
        assert r.reply == ""              # 발화는 Mission 몫
        assert r.wait_minutes == 20
        assert r.need_confirm is False

    def test_wait_without_time_defers(self):
        draft = _IntentDraft(intent="wait", reply="", confidence=0.8)
        r = _finalize(draft, [DEST], user_text="여기서 좀 기다려")
        assert r.intent == "wait"
        assert r.wait_minutes == -1       # Mission 이 "몇 분쯤?" 후속 질문


class TestFinalizeFinish:
    def test_finish_reply_is_silenced(self):
        draft = _IntentDraft(intent="finish", reply="네 안녕히 가세요", confidence=0.9)
        r = _finalize(draft, [DEST], user_text="이제 됐어 고마워")
        assert r.intent == "finish"
        assert r.reply == ""              # 종료 발화는 Mission 몫
        assert r.need_confirm is False


class TestWaitTimeMerge:
    """시간 병합: 코드가 원문의 단일 숫자를 우선, 없으면 LLM 제안 채택.

    LLM 이 범위("5분에서 10분")를 상단×1.5=15 로 제안하는 것은 프롬프트
    규칙이라 여기서 검증 못 한다(실호출 없음) — 대신 그 값이 폴백으로
    쓰이는지, 원문 단일 숫자가 그것을 이기는지를 본다. 상한 강제는 Mission.
    """

    def test_code_single_number_wins_over_llm(self):
        # 사용자가 "20분"이라 명확히 말함 — LLM 이 딴 값(35)을 줘도 코드가 이긴다
        draft = _IntentDraft(intent="wait", confidence=0.9, wait_minutes=35)
        r = _finalize(draft, [DEST], user_text="20분만 기다려")
        assert r.wait_minutes == 20

    def test_llm_value_used_when_code_cannot_parse(self):
        # 범위 발화 — 코드는 단일 숫자를 못 뽑고, LLM 의 15(10×1.5)를 쓴다
        draft = _IntentDraft(intent="wait", confidence=0.9, wait_minutes=15)
        r = _finalize(draft, [DEST], user_text="한 5분에서 10분?")
        assert r.wait_minutes == 15

    def test_neither_defers_to_followup(self):
        draft = _IntentDraft(intent="wait", confidence=0.8, wait_minutes=None)
        r = _finalize(draft, [DEST], user_text="좀 기다려줘")
        assert r.wait_minutes == -1       # Mission 이 "몇 분쯤?" 후속 질문

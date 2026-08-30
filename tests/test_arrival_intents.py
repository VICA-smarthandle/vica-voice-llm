"""도착 후 대화 intent — wait / finish 파싱과 한국어 시간 추출 (LLM 없이).

계약 정본: vica_ros2_ws vica_interfaces/msg/VicaIntent.msg 의 wait/finish 절.
음성은 "어느 질문의 답인지" 판정하지 않는다 — wait/finish 로 발행만 하고,
소비는 상태를 가진 Mission(ARRIVED/ASKING_NEXT)이 정한다. reply 는 빈 문자열.
"""
import pytest

from src.langchain_intent_parser import parse_intent, parse_wait_minutes
from src.schema import DestinationData

DEST = DestinationData(id="d1", name="화장실", aliases=["별빛관 화장실"])


class TestParseWaitMinutes:
    """한국어 시간 표현 -> 분. 없으면 None. 상한은 여기서 걸지 않는다(Mission 몫)."""

    @pytest.mark.parametrize("text,minutes", [
        ("20분", 20), ("이십 분", 20), ("삼십분만", 30), ("10분만 기다려", 10),
        ("반시간", 30), ("5분", 5), ("한 시간", 60), ("두 시간", 120),
    ])
    def test_extracts(self, text, minutes):
        assert parse_wait_minutes(text) == minutes

    @pytest.mark.parametrize("text", ["기다려줘", "여기 있어", "", "글쎄"])
    def test_no_number(self, text):
        assert parse_wait_minutes(text) is None


class TestWaitFinishShortcut:
    @pytest.mark.parametrize("text,minutes", [
        ("기다려줘", -1), ("여기서 기다려", -1),
        ("20분만 기다려줘", 20), ("30분 있다 올게", 30),
    ])
    def test_wait(self, text, minutes):
        r = parse_intent(text, [DEST], history=[])
        assert r.intent == "wait"
        assert r.reply == ""              # 발화는 Mission 몫
        assert r.wait_minutes == minutes

    @pytest.mark.parametrize("text", ["이제 됐어", "그만할래", "안내 끝", "다 됐어요"])
    def test_finish(self, text):
        r = parse_intent(text, [DEST], history=[])
        assert r.intent == "finish"
        assert r.reply == ""

    def test_cancel_is_not_finish(self):
        """'취소'는 주행 중간용 — finish 로 새면 안 된다."""
        from src.replies import CANCEL_CONFIRM
        r = parse_intent("취소", [DEST], history=[])
        assert r.intent == "cancel"
        assert r.reply == CANCEL_CONFIRM

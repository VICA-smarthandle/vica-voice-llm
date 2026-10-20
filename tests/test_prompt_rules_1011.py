"""10-11 지시문 두 규칙(사용자 결정): 평서문은 "."로 끝내기, 여러 곳에 걸리는 말은 짐작하지 말고 다 들어 묻기."""
from src import langchain_intent_parser as parser
from src.schema import DestinationData

DESTS = [DestinationData(id="m", name="남자 화장실", aliases=["화장실", "남자"]),
         DestinationData(id="w", name="여자 화장실", aliases=["화장실", "여자"])]


def test_audio_prompt_rules():
    text = parser.build_audio_prompt(DESTS)
    assert "평서문이니 \".\"로 끝낸다" in text
    assert "접근 질문 중 question 답은 예외" in text
    assert "하나를 짐작해 고르지 말고 clarify" in text and "남자 화장실과 여자 화장실" in text


def test_text_prompt_rules():
    text = parser._build_system_prompt(DESTS)
    assert "되묻는 말만 \"?\"로 끝내고" in text
    assert "navigate 로 하나를 짐작해 고르지 말고 clarify" in text

"""10-11 지시문 두 규칙(사용자 결정): 평서문은 "."로 끝내기, 여러 곳에 걸리는 말은 짐작하지 말고 다 들어 묻기.

10-09~11 로그: 질문 대답 30개 중 18개가 "?"로 끝났다("성별은 추정할 수 없어요?") — 말끝이 올라가고 귀가 쓸데없이
열린다. "화장실 가고 싶어"(남자·여자 화장실 둘 다 별칭 '화장실')에 "여자 화장실로 안내해드릴까요?"·"남자
화장실인가요?"로 한쪽을 짐작했다.
"""
from src import langchain_intent_parser as parser
from src.schema import DestinationData

DESTS = [DestinationData(id="m", name="남자 화장실", aliases=["화장실", "남자"]),
         DestinationData(id="w", name="여자 화장실", aliases=["화장실", "여자"])]


def test_audio_prompt_rules():
    text = parser.build_audio_prompt(DESTS)
    assert "평서문이니 \".\"로 끝낸다" in text
    assert "접근 질문 중 question 답은 예외" in text          # 기존 예외 문구는 그대로
    assert "하나를 짐작해 고르지 말고 clarify" in text and "남자 화장실과 여자 화장실" in text


def test_text_prompt_rules():
    text = parser._build_system_prompt(DESTS)
    assert "되묻는 말만 \"?\"로 끝내고" in text
    assert "navigate 로 하나를 짐작해 고르지 말고 clarify" in text

"""대기 중에는 '가는 중'이라고 말하지 않는다 (2026-10-10 사용자 결정, 15:05 실기).

사회 복지창구 대기 장소에서 기다리던 중 "여자 화장실 가자"에 Realtime 이 주행 중 바꾸기 질문을 골라
"지금 사회 복지창구로 가는 중인데, 여자 화장실로 바꿀까요?"라고 했다. [지금 상황]은 '안내 중: 없음'·
'대화 단계: 대기 중'으로 맞게 왔고, 소리 경로는 모델 reply 를 그대로 말한다(글자 경로는 코드가
dialog_state 가 navigating 일 때만 이 질문을 만든다). 그래서 소리 지시문 두 곳에 반대쪽 금지와
대기 중 답 문장을 적는다.
"""
from __future__ import annotations

from src import langchain_intent_parser as parser
from src.schema import DestinationData

DEST = DestinationData(id="x", name="식당", confirm_prompt="식당으로 안내해드릴까요?")


def _rule(text: str, name: str, next_name: str) -> str:
    start = text.index(f"\n- {name}:")
    return text[start:text.index(f"\n- {next_name}:", start)]


def test_change_question_is_only_for_guiding():
    rule = _rule(parser.build_audio_prompt([DEST]), "navigate(주행 중 바꾸기)", "navigate(확정)")
    assert "'안내 중(주행)'일 때만 쓴다" in rule
    assert '"가는 중"이라고 말하지 않는다' in rule
    assert "'직전에 간 곳'과 '대기 장소'는 지금 가는 곳이 아니다" in rule


def test_waiting_destination_reply_is_the_confirm_question():
    rule = _rule(parser.build_audio_prompt([DEST]), "대기 중 목적지", "정정")
    assert "reply 는 목록의 확인 질문 하나다" in rule

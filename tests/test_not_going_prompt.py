"""대기 중에는 '가는 중'이라고 말하지 않는다 (2026-10-10 사용자 결정, 15:05 실기)."""
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

"""'대기해줘'는 기다리기(wait)다 (2026-10-10 사용자 결정, 14:52 주행).

홈으로 돌아가다 "비카야"로 세운 뒤 "그냥 대기해줘"·"여기서 대기해줘"를 Realtime 이 세 번 중 두 번
잠깐 서기(pause)로 내서 미션이 "지금은 안내 중이 아닙니다"로 거절했다(15:06·15:08). 미션은 세운 복귀
뒤에 기다리기·끝내기·다시 가기만 받고, 기다리기면 직전에 간 곳의 대기 장소로 간다. 안내 주행·일시정지
중의 기다리기는 미션이 잠깐 서기로 받으므로 '대기해줘'를 늘 wait 로 내도 안전하다.
"""
from __future__ import annotations

from src import langchain_intent_parser as parser
from src.schema import DestinationData

DEST = DestinationData(id="x", name="식당", confirm_prompt="식당으로 안내해드릴까요?")
WAIT_WORDS = '"대기해줘"·"여기서 기다려줘"'


def _rule(text: str, name: str, next_name: str) -> str:
    start = text.index(f"\n- {name}:")
    return text[start:text.index(f"\n- {next_name}:", start)]


def test_audio_wait_rule_names_the_stay_words_and_the_stopped_return():
    rule = _rule(parser.build_audio_prompt([DEST]), "wait", "finish")
    assert WAIT_WORDS in rule
    assert '"비카야"로 세운 뒤에도 wait' in rule


def test_audio_pause_rule_sends_the_stay_words_to_wait():
    rule = _rule(parser.build_audio_prompt([DEST]), "pause", "resume")
    assert f"{WAIT_WORDS}는 pause 가 아니라 wait 다" in rule


def test_text_prompt_has_the_same_rules():
    text = parser._build_system_prompt([DEST])
    wait = _rule(text, "wait", "finish")
    assert WAIT_WORDS in wait
    assert '"비카야"로 세운 뒤에도 wait' in wait
    assert f"{WAIT_WORDS}는 pause 가 아니라 wait 다" in _rule(text, "pause", "resume")

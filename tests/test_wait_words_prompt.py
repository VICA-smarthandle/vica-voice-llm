"""'대기해줘'는 기다리기(wait)다 (2026-10-10 사용자 결정, 14:52 주행)."""
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

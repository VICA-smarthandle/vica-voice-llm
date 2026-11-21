"""잡담 길(설계 6.5, 원칙 25·26, Task 4.4) — 비카 자신에 대한 내용(FACTS)·잡담 규칙(TALK_RULE)이 지시문에 들어가는 조건.

`fixed_answers.on()`(VICA_TALK·VICA_RECUT·VICA_ANSWER 모두 on)이고 재단을 싣는 지시문일 때만 글자·소리 지시문에 들어간다.
로컬(Ollama)이 답하는 호출(recut=False)·꺼진 조합의 글자는 그대로다.
"""
from __future__ import annotations

import itertools

import pytest

from src import langchain_intent_parser as parser
from src import vica_facts
from src.schema import DestinationData

DESTS = [DestinationData(id="wc", name="화장실", confirm_prompt="화장실로 안내해드릴까요?"),
         DestinationData(id="r407", name="407호", confirm_prompt="407호로 안내해드릴까요?")]

TEXT_Q_OLD = '- question: 이동이 아니라 정보 질문("지금 몇 층이야?").'
TEXT_Q_NEW = '- question: 이동이 아닌 정보 질문이나 로봇에게 건 가벼운 말("지금 몇 층이야?").'
AUDIO_OLD = "농담·추측·맞장구(\"~하겠죠\", \"~하시군요\")와 예시 늘어놓기"
AUDIO_NEW = "안내에 관한 말에는 농담·추측·맞장구(\"~하겠죠\", \"~하시군요\")와 예시 늘어놓기"

FACTS_TEXT = """[비카]
- 이름: 비카(VICA), VISION COMPANION을 줄인말이다.
- 만든사람 : 문지원,문여원,김민서,이태규
- 할 수 있는 일 : 건물 안 목적지 까지 안내하기, 도착한 곳에서 기다리기, 짐 옮기기, 물류배송하기, 경비로봇 되기
- 부를 때는 "비카야"라고 한다.
- 자기소개 : 안녕하세요? 저는 시각장애인 안내로봇 비카입니다.
- 할 수 있는 언어 : 한국어
- 비카 이행시: 비: 비척비척 비가 올때는 카: 카스 한잔 어떠세요?"""


def env(monkeypatch, talk="on", recut="on", answer="on"):
    for k, v in (("VICA_TALK", talk), ("VICA_RECUT", recut), ("VICA_ANSWER", answer)):
        monkeypatch.setenv(k, v)


def both():
    return (parser._build_system_prompt(DESTS), parser.build_audio_prompt(DESTS))


def test_p26_on_prompts_carry_the_facts_and_the_talk_rule(monkeypatch):
    env(monkeypatch)
    for prompt in both():
        assert vica_facts.FACTS in prompt and vica_facts.TALK_RULE in prompt


def test_p16_local_messages_do_not_carry_them(monkeypatch):
    env(monkeypatch)
    local = parser._build_system_prompt(DESTS, recut=False)
    assert vica_facts.FACTS not in local and vica_facts.TALK_RULE not in local and "VISION COMPANION" not in local


def test_off_prompts_are_unchanged(monkeypatch):
    env(monkeypatch, talk="off")
    for prompt in both():
        assert "VISION COMPANION" not in prompt and vica_facts.TALK_RULE not in prompt


@pytest.mark.parametrize("talk,recut,answer", [c for c in itertools.product(("on", "off"), repeat=3) if c != ("on",) * 3])
def test_p26_any_off_switch_keeps_prompts_without_the_talk_path(monkeypatch, talk, recut, answer):
    env(monkeypatch, talk, recut, answer)
    for prompt in both():
        assert vica_facts.FACTS not in prompt and vica_facts.TALK_RULE not in prompt


def test_p25_on_widens_text_question_and_narrows_audio_joke_ban(monkeypatch):
    env(monkeypatch)
    text, audio = both()
    assert TEXT_Q_NEW in text and TEXT_Q_OLD not in text
    assert AUDIO_NEW in audio and AUDIO_OLD not in audio.replace(AUDIO_NEW, "")


def test_p25_off_keeps_both_lines_as_they_were(monkeypatch):
    env(monkeypatch, talk="off")
    text, audio = both()
    assert TEXT_Q_OLD in text and TEXT_Q_NEW not in text
    assert AUDIO_OLD in audio and AUDIO_NEW not in audio


def test_facts_are_the_users_text_verbatim():
    assert vica_facts.FACTS.splitlines() == FACTS_TEXT.splitlines()


def test_talk_rule_text():
    assert vica_facts.TALK_RULE == (
        "안내와 상관없는 말이어도 로봇에게 한 말이면 question 으로 한 문장으로 답한다. 비카 자신에 대한 것은 [비카]의 내용으로, "
        "인사나 가벼운 말은 짧게 답한다. 사실이 필요한 것은 [비카]·목적지 목록·[지금 상황]에 있는 것만 말하고, 없으면 "
        "\"그건 제가 몰라요.\"라고만 한다. 하지 않을 일을 약속하지 않는다.")

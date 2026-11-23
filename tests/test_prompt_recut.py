"""지시문 두 벌(글자 경로·소리 경로)과 로컬 규칙의 재단(VICA_RECUT, 원칙 1·4·7·12·31)."""
import types

from src import intent_recut, langchain_intent_parser as parser, local_rules
from src.question_record import OpenQuestion
from src.realtime_intent import RealtimeResult
from src.schema import DestinationData

DESTS = [DestinationData(id="a", name="남자 화장실", aliases=["화장실"], confirm_prompt="남자 화장실로 안내해드릴까요?"),
         DestinationData(id="b", name="여자 화장실", aliases=["화장실"], confirm_prompt="여자 화장실로 안내해드릴까요?")]


def _q():
    return OpenQuestion("q", "confirm", "화장실로 안내해드릴까요?", ("t",), 0.0)


def test_off_prompts_are_byte_identical(monkeypatch):
    before = (parser._build_system_prompt(DESTS), parser.build_audio_prompt(DESTS))
    monkeypatch.setenv("VICA_RECUT", "off")
    assert (parser._build_system_prompt(DESTS), parser.build_audio_prompt(DESTS)) == before
    assert "not_for_robot" not in before[0] and "not_for_robot" not in before[1]


def test_off_local_rules_are_the_constant():
    assert local_rules.prompt_rules(False) == local_rules.LOCAL_PROMPT_RULES
    assert "not_for_robot" not in local_rules.prompt_rules(False)


def test_on_both_prompts_carry_the_same_principles(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    for prompt in (parser._build_system_prompt(DESTS), parser.build_audio_prompt(DESTS)):
        assert intent_recut.PROMPT_INTENTS in prompt and intent_recut.PROMPT_ORDER in prompt
        assert "clarify_kind=not_available" in prompt and "clarify_kind=several" in prompt


def test_p12_on_prompts_stop_asking_the_model_to_write_reask_sentences(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    text, audio = parser._build_system_prompt(DESTS), parser.build_audio_prompt(DESTS)
    assert "reply 에 되묻는 질문을 담아라" not in text
    assert "reply 에 되묻는 질문(반드시" not in audio and "학과사무실이 맞을까요" not in audio
    assert "잘 못 들었는데, 다시 말씀해 주시겠어요" not in audio


def test_on_local_rule_five_says_not_for_robot(monkeypatch):
    on = local_rules.prompt_rules(True)
    assert "intent 는 not_for_robot" in on and "intent 는 unknown" not in on
    monkeypatch.setenv("VICA_RECUT", "on")
    assert "intent 는 not_for_robot" in parser._build_system_prompt(DESTS, local=True)


def test_on_no_contradiction_left(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    text, audio = parser._build_system_prompt(DESTS), parser.build_audio_prompt(DESTS)
    assert "그 외(question/clarify/unknown)" not in text
    assert "question·clarify·unknown 공통" not in audio
    assert "clarify 로 몇 번인지 되묻는다" not in audio


def test_p7_clue_line_reaches_all_three_prompts(monkeypatch):
    monkeypatch.setenv("VICA_RECUT", "on")
    q = _q()
    line = intent_recut.clue_line(q)
    assert line in parser._build_system_prompt(DESTS, question=q)
    seen = {}

    class Client:
        def ask(self, pcm, history, instructions, **kw):
            seen["audio"] = instructions
            return RealtimeResult(draft={"intent": "unknown"}, heard_text="", latency_sec=0.1)

        def ask_text(self, text, history, instructions, **kw):
            seen["text"] = instructions
            return RealtimeResult(draft={"intent": "unknown"}, heard_text=text, latency_sec=0.1)

    monkeypatch.setattr(parser, "get_realtime_client", lambda: Client())
    parser.parse_intent_audio(b"\x00\x00", DESTS, question=q)
    parser.parse_intent_realtime_text("응", DESTS, question=q)
    assert line in seen["audio"] and line in seen["text"]


def test_off_clue_line_absent(monkeypatch):
    assert intent_recut.clue_line(_q()) not in parser._build_system_prompt(DESTS, question=_q())


def test_p8_unreadable_speech_is_unknown_not_dropped(monkeypatch):
    line = "외국어이거나 잘못 들린 듯해 뜻을 알 수 없는 말도 not_for_robot 이 아니다 — 로봇에게 한 말일 수 있으니 unknown 이다."
    assert intent_recut.PROMPT_ORDER.endswith(line)
    monkeypatch.setenv("VICA_RECUT", "on")
    assert line in parser._build_system_prompt(DESTS) and line in parser.build_audio_prompt(DESTS)
    monkeypatch.setenv("VICA_RECUT", "off")
    assert line not in parser._build_system_prompt(DESTS) and line not in parser.build_audio_prompt(DESTS)


def test_p18_same_place_said_a_little_differently_is_not_unavailable(monkeypatch):
    line = ("  not_available = 목록에 없는 곳을 찾는다. clarify_wanted 에 찾는 곳이나 물건의 이름만 사용자가 말한 낱말로 적는다(문장을 통째로 "
            "적지 않는다). 다른 곳인데 비슷하다고 그곳으로 바꾸지 않는다. 같은 곳을 조금 다르게 부른 것(띄어쓰기, 앞뒤에 붙은 말, 별칭)은 "
            "not_available 이 아니라 그 목적지로 navigate 다.")
    assert line in intent_recut.PROMPT_INTENTS.split("\n")
    assert "사용자가 말한 낱말 그대로 적는다. 비슷한 다른 곳으로 바꾸지 않는다." not in intent_recut.PROMPT_INTENTS
    monkeypatch.setenv("VICA_RECUT", "on")
    assert line in parser._build_system_prompt(DESTS) and line in parser.build_audio_prompt(DESTS)

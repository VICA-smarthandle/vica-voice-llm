"""[비카] 대답은 정해진 문장으로(Task N3, 원칙 25·22·10·9·16) — LLM 은 about 칸에 주제만 고르고 문장은 코드가 낸다.

2026-11-22 4단계 실기: LLM 이 지은 [비카] 대답은 supertonic 이 그 자리에서 합성해 억양·속도가 어색했고, 주신 글을 줄이거나
바꿔 말했다. 사용자 결정: "어디 갈 수 있어?"(places)와 같은 방식 — 여덟 문장은 사용자가 확정한 글자 그대로이고 CosyVoice 로
구워 둔다(assets/baked). `fixed_answers.on()`(VICA_TALK·VICA_RECUT·VICA_ANSWER 모두 on)일 때만 움직이고, 꺼졌으면
지시문·스키마·도구·동작이 BASE(afb838f)와 같다. 로컬(Ollama)이 답하는 호출에는 싣지 않는다(원칙 16).
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import types

import pytest

from src import langchain_intent_parser as parser
from src.question_record import OpenQuestion
from src.realtime_intent import RealtimeResult, build_intent_tool
from src.schema import DestinationData, VicaIntent
from tests.test_fixed_answers import _node, _publish, node_mod  # noqa: F401  (의도 노드 _publish_intent 를 가짜 ROS 로)

# 2026-11-22 사용자가 확정한 글자 그대로(설명서 task-N3 표) — 낱말·띄어쓰기·문장부호를 바꾸지 않는다.
SENTENCES = {
    "name": "제 이름은 비카입니다. 비전 컴패니언을 줄인 말이에요.",
    "makers": "저는 문지원님, 문여원님, 김민서님, 이태규님이 만들었습니다. 제 부모님들이에요.",
    "abilities": "건물 안 목적지까지 안내하기, 도착한 곳에서 대기하기, 물류배송하기, 짐 옮기기, 야간 순찰을 할 수 있습니다.",
    "call": "저를 부르실 때는 비카야라고 불러주세요.",
    "intro": "안녕하세요? 저는 시각장애인 안내로봇 비카입니다. 저와 함께 목적지까지 동행하실 수 있습니다.",
    "language": "저는 지금은 한국어만 할 수 있습니다.",
    "acrostic": "비, 비척비척 비가 올 때는. 카, 카스 한잔 어떠세요?",
    "cannot": "죄송해요. 아직은 제가 할 수 없습니다.",
}
ACROSTIC = SENTENCES["acrostic"]
ABOUT_RULE = (
    "[비카에 대한 물음]\n"
    "사용자가 비카 자신에 대해 묻거나, 비카가 모르는 것·할 수 없는 일을 말하면 intent 는 question, reply 는 빈 문자열로 두고 about 에 아래 값 하나를 적는다 — 문장은 시스템이 말한다(직접 지어 말하지 않는다).\n"
    '- name: 이름이나 이름의 뜻을 묻는다("이름이 뭐야?", "비카가 무슨 뜻이야?")\n'
    "- makers: 누가 만들었는지 묻는다\n"
    '- abilities: 무엇을 할 수 있는지 묻는다("뭐 할 수 있어?") — 어디로 갈 수 있는지 묻는 것은 places 다\n'
    "- call: 어떻게 부르면 되는지 묻는다\n"
    '- intro: 누구인지 묻거나 자기소개를 시킨다("넌 누구야?", "자기소개 해 봐")\n'
    '- language: 어떤 말을 할 수 있는지 묻는다("영어 할 줄 알아?")\n'
    "- acrostic: 비카 이행시를 시킨다\n"
    "- cannot: 비카가 모르는 것(사람·날짜·날씨·세상 일)을 묻거나, 안내 밖의 할 수 없는 일(노래·전화·검색 등)을 시킨다\n"
    "그 밖의 말은 about 을 빈 문자열로 둔다.")
TALK_RULE = (
    "로봇에게 부탁하거나 묻는 말은 안내와 상관없어도(잡담·추천) not_for_robot 이 아니다. about 에 해당하지 않는 인사나 가벼운 말, "
    "추천이나 의견을 묻는 말에는 question 으로 한 문장으로 짧게 답하고 reply 를 비우지 않는다. 사실이 필요한 것은 목적지 목록·"
    "[지금 상황]에 있는 것만 말한다. 할 수 없는 일을 할 수 있다고 하지 않고, 하지 않을 일을 약속하지 않는다.")
ABOUT_DESC = "비카 자신에 대한 물음이나 비카가 모르는 것·할 수 없는 일이면 그 주제. 그 밖은 빈 문자열"
REPLY_DESC = ("사용자에게 할 짧은 말. question 은 about 을 적지 않았으면 반드시 한 문장을 적는다. "
              "할 말이 없는 의도(affirm·deny·repeat·not_for_robot 등)는 빈 문자열")
MAKERS = ("문지원", "문여원", "김민서", "이태규")

DESTS = [DestinationData(id="wc", name="화장실", confirm_prompt="화장실로 안내해드릴까요?"),
         DestinationData(id="r407", name="407호", confirm_prompt="407호로 안내해드릴까요?")]
CONFIRM = OpenQuestion("q", "confirm", "화장실로 안내해드릴까요?", ("wc",), 0.0)
OFF = [c for c in itertools.product(("on", "off"), repeat=3) if c != ("on",) * 3]   # (TALK, RECUT, ANSWER)


def fa():
    return importlib.import_module("src.fixed_answers")


def switches(monkeypatch, talk="on", recut="on", answer="on"):
    for key, value in (("VICA_TALK", talk), ("VICA_RECUT", recut), ("VICA_ANSWER", answer)):
        monkeypatch.setenv(key, value)


# ---- 1. 문장의 집 -----------------------------------------------------------------------------------------------
def test_p25_about_sentences_are_the_users_text_verbatim():
    from src import vica_facts
    assert list(vica_facts.ABOUT_SENTENCES.items()) == list(SENTENCES.items())


def test_p25_the_two_prompt_texts_and_the_field_description_are_verbatim():
    from src import vica_facts
    assert vica_facts.ABOUT_RULE == ABOUT_RULE
    assert vica_facts.TALK_RULE == TALK_RULE
    assert vica_facts.ABOUT_DESCRIPTION == ABOUT_DESC


def test_p25_old_facts_are_gone():
    from src import vica_facts
    assert not hasattr(vica_facts, "FACTS")      # 11-21 에 받은 글 — 사용자가 위 여덟 문장으로 바꿨다


def test_about_sentences_have_no_emergency_keyword():
    # 자가 트리거(test_spoken_text 와 같은 검사) — 로봇이 말하는 문장에 긴급어가 있으면 자기 말을 긴급어로 듣는다.
    from src.emergency_filter import detect_emergency
    assert [s for s in SENTENCES.values() if detect_emergency(s)] == []


# ---- 4. 지시문 -----------------------------------------------------------------------------------------------------
def _prompts():
    return parser._build_system_prompt(DESTS, question=CONFIRM), parser.build_audio_prompt(DESTS)


def test_p25_on_both_prompts_carry_about_rule_then_talk_rule(monkeypatch):
    switches(monkeypatch)
    for prompt in _prompts():
        assert ABOUT_RULE + "\n" + TALK_RULE + "\n\n[목적지 목록]" in prompt


def test_p25_on_nothing_sent_out_names_the_makers_or_vision_companion(monkeypatch):
    """만든 사람 이름은 이제 코드의 문장에만 있다 — 지시문·스키마·도구 정의로 외부에 나가지 않는다."""
    switches(monkeypatch)
    sent = (*_prompts(), parser._build_system_prompt(DESTS, recut=False),
            json.dumps(parser._draft_cls().model_json_schema(), ensure_ascii=False),
            json.dumps(parser._set_intent_tool(), ensure_ascii=False),
            json.dumps(build_intent_tool(), ensure_ascii=False))
    for text in sent:
        assert [w for w in (*MAKERS, "VISION COMPANION", "비전 컴패니언") if w in text] == []


# 꺼진 조합(TALK 만 끔 포함)은 BASE afb838f 와 글자까지 같다 — json.dumps(…, ensure_ascii=False, sort_keys=True)·지시문의
# sha256 앞 16자. 글자 지시문은 열린 확인 질문(CONFIRM)을 싣는다(대답 칸이 켜지면 그 줄이 붙는다).
BASE_SHA = {
    ("on", "on", "off"): ("9075ae28dc46e327", "8336c5b13a02bad6", "532dcb18529c6375", "60e9cc51a6c85c5b"),
    ("on", "off", "on"): ("c6ef8a2308c5c58e", "e6bf82c9915294c8", "5948f9f7029dd752", "21a57297adef8c2e"),
    ("on", "off", "off"): ("a6faf98b26a8756e", "e6bf82c9915294c8", "d1f3680053495966", "e276e77ccfd4c08d"),
    ("off", "on", "on"): ("5700a1b35655c66e", "8336c5b13a02bad6", "80de6134da3945eb", "3b3e8aa6fd2b3b5c"),
    ("off", "on", "off"): ("9075ae28dc46e327", "8336c5b13a02bad6", "532dcb18529c6375", "60e9cc51a6c85c5b"),
    ("off", "off", "on"): ("c6ef8a2308c5c58e", "e6bf82c9915294c8", "5948f9f7029dd752", "21a57297adef8c2e"),
    ("off", "off", "off"): ("a6faf98b26a8756e", "e6bf82c9915294c8", "d1f3680053495966", "e276e77ccfd4c08d"),
}
BASE_LOCAL_SHA = "a6faf98b26a8756e"      # 로컬(recut=False) 지시문 — 스위치와 상관없이 같다(원칙 16)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@pytest.mark.parametrize("talk,recut,answer", OFF)
def test_p16_off_prompts_schema_and_tool_are_base(monkeypatch, talk, recut, answer):
    switches(monkeypatch, talk, recut, answer)
    text, audio = _prompts()
    got = (_sha(text), _sha(audio),
           _sha(json.dumps(parser._draft_cls().model_json_schema(), ensure_ascii=False, sort_keys=True)),
           _sha(json.dumps(build_intent_tool(), ensure_ascii=False, sort_keys=True)))
    assert got == BASE_SHA[(talk, recut, answer)]
    assert _sha(parser._build_system_prompt(DESTS, recut=False)) == BASE_LOCAL_SHA


def test_p16_on_local_prompt_and_schema_carry_no_about(monkeypatch):
    switches(monkeypatch)
    assert _sha(parser._build_system_prompt(DESTS, recut=False)) == BASE_LOCAL_SHA
    assert "about" not in parser._base_draft_cls().model_fields
    assert "about" not in json.dumps(parser._base_draft_cls().model_json_schema())


# ---- 2. 새 칸 about --------------------------------------------------------------------------------------------------
def test_p25_on_schema_offers_the_about_field(monkeypatch):
    switches(monkeypatch)
    for cls in (parser._TalkDraft, parser._TalkAnswerDraft):
        prop = cls.model_json_schema()["properties"]["about"]
        assert (prop["type"], prop["enum"], prop["description"]) == ("string", ["", *SENTENCES], ABOUT_DESC)
    assert parser._draft_cls() is parser._TalkAnswerDraft
    switches(monkeypatch, answer="off", recut="on", talk="on")
    assert "about" not in parser._draft_cls().model_fields       # 대답 칸이 꺼지면 on() 이 거짓 — 지금의 재단 스키마
    switches(monkeypatch)
    tool = parser._set_intent_tool()["function"]["parameters"]["properties"]["about"]   # 글자 경로 도구 호출(VICA_TOOLS_TEXT)
    assert tool["enum"] == ["", *SENTENCES]


@pytest.mark.parametrize("value,want", [("name", "name"), ("", ""), (None, ""), (3, ""), (["name"], ""), ("weather", "weather")])
def test_p25_about_field_reads_odd_values_without_failing(monkeypatch, value, want):
    """외부 API 는 형식을 강제하지 않는다 — null·숫자·목록에도 발화 하나가 통째로 실패하지 않는다(answer 칸과 같다).
    목록 밖의 글자는 그대로 받고 뒤처리(early)가 무시한다."""
    switches(monkeypatch)
    assert parser._draft_cls()(intent="question", about=value).about == want
    assert parser._draft_cls()(intent="question").about == ""


def test_p25_on_realtime_tool_requires_about_and_reply_says_unless_about(monkeypatch):
    switches(monkeypatch)
    params = build_intent_tool()["parameters"]
    assert params["properties"]["about"] == {"type": "string", "enum": ["", *SENTENCES], "description": ABOUT_DESC}
    assert params["required"] == ["intent", "heard_text", "answer", "reply", "about"]    # 선택 칸은 모델이 빼먹는다(Task N1)
    assert params["properties"]["reply"] == {"type": "string", "description": REPLY_DESC}


# ---- 3. 뒤처리 early ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("about", list(SENTENCES))
def test_p25_early_question_about_says_the_fixed_sentence(monkeypatch, about):
    switches(monkeypatch)
    draft = parser._draft_cls()(intent="question", about=about, reply="제가 지은 말이에요.", confidence=0.8,
                                need_confirm=True, destination_candidate="화장실")
    assert fa().early(draft, DESTS) == VicaIntent(intent="question", reply=SENTENCES[about], need_confirm=False,
                                                  confidence=0.8)


def test_p25_early_question_without_about_takes_the_present_path(monkeypatch):
    switches(monkeypatch)
    assert fa().early(parser._draft_cls()(intent="question", about="", reply="안녕하세요."), DESTS) is None


@pytest.mark.parametrize("kind", ["navigate", "unknown", "not_for_robot", "clarify", "affirm", "dismiss", "cancel"])
def test_p25_early_reads_about_only_on_question(monkeypatch, kind):
    switches(monkeypatch)
    draft = parser._draft_cls()(intent=kind, about="name", reply="", destination_candidate="화장실")
    assert fa().early(draft, DESTS) is None


def test_p28_p29_places_and_repeat_keep_their_own_answers_even_with_about(monkeypatch):
    switches(monkeypatch)
    places = fa().early(parser._draft_cls()(intent="places", about="name"), DESTS)
    assert (places.intent, places.reply) == ("question", "화장실, 407호로 안내할 수 있습니다.")
    repeat = fa().early(parser._draft_cls()(intent="repeat", about="cannot"), DESTS)
    assert (repeat.intent, repeat.reply) == ("repeat", "")


@pytest.mark.parametrize("about", ["weather", "Name", " name", "name ", "이름", "places", "FACTS"])
def test_p25_early_ignores_about_outside_the_list(monkeypatch, about):
    switches(monkeypatch)
    assert fa().early(parser._draft_cls()(intent="question", about=about, reply=""), DESTS) is None


@pytest.mark.parametrize("talk,recut,answer", OFF)
def test_p25_early_is_none_when_off(monkeypatch, talk, recut, answer):
    switches(monkeypatch, talk, recut, answer)
    draft = types.SimpleNamespace(intent="question", about="name", reply="", confidence=0.9)
    assert fa().early(draft, DESTS) is None


# ---- 글자·소리 두 경로 끝에서 끝까지(가짜 LLM) ------------------------------------------------------------------------
class _Manager:
    """parse_intent 의 백엔드 자리 — 받은 messages 를 적고 정해 둔 초안을 돌려준다."""
    has_local = False

    def __init__(self, **draft):
        self.draft, self.seen = draft, []

    def invoke(self, messages, local_messages=None):
        self.seen.append(messages)
        return parser._draft_cls()(**self.draft)


@pytest.mark.parametrize("about", list(SENTENCES))
def test_p25_text_path_about_is_the_fixed_sentence(monkeypatch, about):
    switches(monkeypatch)
    manager = _Manager(intent="question", about=about, reply="", confidence=0.9)
    monkeypatch.setattr(parser, "PROVIDER", "openai")     # 외부 API 가 답하는 글자 경로(공급자 ollama 면 Ollama 가 답한다 — 원칙 16)
    monkeypatch.setattr(parser, "get_backend_manager", lambda: manager)
    out = parser.parse_intent("이름이 뭐야?", DESTS)
    assert (out.intent, out.reply, out.need_confirm, out.matched_destination_id) == ("question", SENTENCES[about], False, None)
    assert ABOUT_RULE in manager.seen[0][0].content


def test_p25_text_path_drops_what_the_llm_wrote(monkeypatch):
    switches(monkeypatch)
    manager = _Manager(intent="question", about="makers", reply="문지원·문여원·김민서·이태규가 만들었어요.", confidence=0.9)
    monkeypatch.setattr(parser, "PROVIDER", "openai")
    monkeypatch.setattr(parser, "get_backend_manager", lambda: manager)
    assert parser.parse_intent("누가 만들었어?", DESTS).reply == SENTENCES["makers"]


class _Realtime:
    """소리 첫 호출(ask)만 받는 가짜 Realtime. timeout_sec 는 진짜 클라이언트 자리(빈 대답 재시도의 시간 예산이 읽는다)."""

    def __init__(self, draft, heard):
        self.draft, self.heard, self.instructions, self.timeout_sec = draft, heard, "", 8.0

    def ask(self, pcm, history, instructions, **kw):
        self.instructions = instructions
        return RealtimeResult(draft=dict(self.draft), heard_text=self.heard, latency_sec=0.7)


@pytest.mark.parametrize("about", list(SENTENCES))
def test_p25_audio_path_about_is_the_fixed_sentence_and_is_not_retried(monkeypatch, about):
    """5번 — N1 의 빈 대답 재시도·되묻기는 early 를 지난 뒤의 결과를 본다. about 이 채워진 question 은 걸리지 않는다."""
    switches(monkeypatch)
    retried = []

    def retry(*a, **k):
        retried.append(a)
        return VicaIntent(intent="question", reply="다른 말이에요."), "", 0.1, {"src": "llm"}

    monkeypatch.setattr(parser, "parse_intent_realtime_text", retry)
    client = _Realtime({"intent": "question", "reply": "", "about": about, "heard_text": "이름이 뭐야", "answer": ""},
                       "이름이 뭐야")
    monkeypatch.setattr(parser, "get_realtime_client", lambda: client)
    intent, heard, _dt, info = parser.parse_intent_audio(b"\x00\x00", DESTS)
    assert (intent.intent, intent.reply, intent.need_confirm, info["src"]) == ("question", SENTENCES[about], False, "llm")
    assert retried == []
    assert ABOUT_RULE in client.instructions and TALK_RULE in client.instructions


def test_p25_audio_path_under_a_confirm_question_still_says_the_sentence(monkeypatch):
    switches(monkeypatch)
    client = _Realtime({"intent": "question", "reply": "", "about": "name", "heard_text": "이름이 뭐야", "answer": "other"},
                       "이름이 뭐야")
    monkeypatch.setattr(parser, "get_realtime_client", lambda: client)
    intent, *_ = parser.parse_intent_audio(b"\x00\x00", DESTS, question=CONFIRM)
    assert (intent.intent, intent.reply, intent.question_id) == ("question", SENTENCES["name"], "q")


def test_p25_audio_path_empty_about_is_still_the_n1_path(monkeypatch):
    """about 이 빈 question 에 할 말도 비었으면 지금처럼 N1 이 한 번 더 묻는다 — 새 칸이 그 길을 막지 않는다."""
    switches(monkeypatch)
    retried = []
    monkeypatch.setattr(parser, "parse_intent_realtime_text",
                        lambda *a, **k: retried.append(a) or (VicaIntent(intent="question", reply="좋은 저녁이에요."),
                                                              "", 0.1, {"src": "llm"}))
    client = _Realtime({"intent": "question", "reply": "", "about": "", "heard_text": "고마워"}, "고마워")
    monkeypatch.setattr(parser, "get_realtime_client", lambda: client)
    intent, _h, _dt, info = parser.parse_intent_audio(b"\x00\x00", DESTS)
    assert (intent.reply, info["src"], len(retried)) == ("좋은 저녁이에요.", "llm+빈대답재시도", 1)


# ---- 6. 말한 뒤에 듣지 않는다 -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("about", list(SENTENCES))
def test_p9_about_sentences_do_not_wait_for_an_answer(monkeypatch, about):
    """여덟 문장은 알리는 말이다 — acrostic 은 "?"로 끝나고 intro 는 가운데 "?"가 있다. 질문으로 미션에 넘기지 않고(넘기면 15초 뒤
    "잘 듣지 못했습니다…"), 접근 질문 중에도 지우지 않는다."""
    from src.mission_question import quiet_for_mission
    from src.one_voice import hand_over_question
    from src.replies import APPROACH_QUESTION, expects_answer
    switches(monkeypatch)
    said = VicaIntent(intent="question", reply=SENTENCES[about], need_confirm=False)
    assert expects_answer(SENTENCES[about]) is False
    assert expects_answer(f"  {SENTENCES[about]} ") is False
    assert hand_over_question(said, DESTS) == said
    assert quiet_for_mission(said, "awaiting_user", APPROACH_QUESTION, 1.0, [d.name for d in DESTS]) == said


@pytest.mark.parametrize("talk,recut,answer", OFF)
def test_p16_off_sentence_shape_rule_is_as_before(monkeypatch, talk, recut, answer):
    """꺼지면 문장 꼴 판정(expects_answer)은 BASE 그대로 — "?"로 끝나면 대답을 기다린다."""
    from src.replies import expects_answer
    switches(monkeypatch, talk, recut, answer)
    assert expects_answer(ACROSTIC) is True and expects_answer(SENTENCES["intro"]) is False


def test_p9_a_question_like_other_text_still_waits(monkeypatch):
    from src.replies import expects_answer
    switches(monkeypatch)
    assert expects_answer("카스 한잔 어떠세요?") is True               # 정해진 문장과 글자까지 같을 때만 알리는 말
    assert expects_answer(ACROSTIC.replace("카스", "맥주")) is True


@pytest.mark.parametrize("hand_over", [True, False])
def test_p9_node_says_the_acrostic_itself_and_opens_no_answer_window(node_mod, monkeypatch, hand_over):  # noqa: F811
    """의도 노드 _publish_intent — 음성이 말하고(ask_text 없음), 옛 설치본(질문 넘김 꺼짐)에서도 듣기 요청을 내지 않는다."""
    switches(monkeypatch)
    me = _node(None, [])
    me._hand_over = hand_over
    sent, said = _publish(node_mod, me, VicaIntent(intent="question", reply=ACROSTIC, need_confirm=False),
                          text="비카 이행시 해 줘")
    assert [(m["intent"], m["reply"], m["ask_text"], m["ask_kind"]) for m in sent] == [("question", ACROSTIC, "", "")]
    assert len(said) == 1 and said[0].endswith(":" + ACROSTIC)
    assert me._listen_pub.sent == []


def test_p28_p9_repeat_of_the_acrostic_is_said_again_by_the_voice(monkeypatch):
    """"다시 말해줘" — 마지막 말이 이행시면 음성이 다시 말한다(질문으로 미션에 보내지 않는다)."""
    switches(monkeypatch)
    out = fa().settle_repeat(None, [(1.0, ACROSTIC)])
    assert (out.intent, out.reply, out.ask_text, out.ask_kind) == ("question", ACROSTIC, "", "")


# ---- 7. 굽기 묶음(scripts/bake_one_cv.py --about) — 모델 없이 고를 목록만 본다 ---------------------------------------------
def _bake_script():
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "scripts" / "bake_one_cv.py"
    spec = importlib.util.spec_from_file_location("bake_one_cv", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_p9_bake_about_batch_is_the_eight_sentences():
    assert _bake_script()._about_ments() == {f"vica_about_{topic}": text for topic, text in SENTENCES.items()}


def test_p9_bake_about_batch_skips_what_is_baked_with_the_same_words(monkeypatch, tmp_path):
    bake = _bake_script()
    monkeypatch.setattr(bake, "BAKED", tmp_path)
    (tmp_path / "vica_about_name.wav").write_bytes(b"RIFF")
    (tmp_path / "vica_about_call.wav").write_bytes(b"RIFF")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "vica_about_name.wav": SENTENCES["name"],                 # 같은 글자로 구움 — 건너뛴다
        "vica_about_call.wav": "저를 부를 때는 비카야라고 해 주세요.",   # 글자가 다르다 — 다시 굽는다
        "vica_about_makers.wav": SENTENCES["makers"],             # manifest 에만 있고 파일이 없다 — 굽는다
    }, ensure_ascii=False))
    todo = [name for name, _ in bake._todo(bake._about_ments(), force=False)]
    assert todo == [f"vica_about_{t}" for t in SENTENCES if t != "name"]
    assert len(bake._todo(bake._about_ments(), force=True)) == 8

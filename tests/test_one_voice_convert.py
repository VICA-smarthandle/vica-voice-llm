"""질문 넘김 칸(설계 2026-11-14 4.2·4.4, 3-가) — 스키마·메시지 변환."""
from __future__ import annotations

import src.ros_convert as rc
from src.langchain_intent_parser import _IntentDraft
from src.schema import VicaIntent

_OLD_SLOTS = ("intent", "destination_candidate", "matched_destination_id", "confidence", "need_confirm",
              "reply", "safety_flag", "wait_minutes", "speaker_verdict", "speaker_score", "speaker_score_joined")


class _OldMsg:
    """옛 설치본 — 실제 ROS 메시지처럼 생성 때 모든 칸을 채운다."""
    __slots__ = _OLD_SLOTS

    def __init__(self):
        for cls in type(self).__mro__:
            for name in getattr(cls, "__slots__", ()):
                setattr(self, name, None)


class _NewMsg(_OldMsg):
    __slots__ = ("ask_text", "ask_kind", "ask_option_ids")


def test_intent_to_msg_carries_ask_fields(monkeypatch):
    monkeypatch.setattr(rc, "VicaIntentMsg", _NewMsg)
    msg = rc.intent_to_msg(VicaIntent(intent="clarify", ask_text="어떤 서류를 내러 가시나요?", ask_kind="open",
                                      ask_option_ids=["a", "b"]))
    assert (msg.ask_text, msg.ask_kind, list(msg.ask_option_ids)) == ("어떤 서류를 내러 가시나요?", "open", ["a", "b"])


def test_intent_to_msg_skips_ask_fields_on_old_interface(monkeypatch):
    monkeypatch.setattr(rc, "VicaIntentMsg", _OldMsg)
    msg = rc.intent_to_msg(VicaIntent(intent="clarify", ask_text="어디로 가시나요?", ask_kind="open"))
    assert msg.intent == "clarify"


def test_supports_ask_fields_false_on_old_interface():
    assert rc.supports_ask_fields(_OldMsg()) is False
    assert rc.supports_ask_fields(_NewMsg()) is True


def test_llm_draft_has_no_ask_fields():
    """LLM 출력 형식은 그대로 — 질문 넘김 칸은 코드가 채운다."""
    assert not [name for name in _IntentDraft.model_fields if name.startswith("ask_")]


class _AnswerMsg(_NewMsg):
    __slots__ = ("question_id", "answer", "answer_value")


def test_intent_to_msg_carries_answer_fields(monkeypatch):
    monkeypatch.setattr(rc, "VicaIntentMsg", _AnswerMsg)
    msg = rc.intent_to_msg(VicaIntent(intent="affirm", question_id="q1-3", answer="destination", answer_value="toilet"))
    assert (msg.question_id, msg.answer, msg.answer_value) == ("q1-3", "destination", "toilet")


def test_intent_to_msg_skips_answer_fields_on_msg_without_them(monkeypatch):
    monkeypatch.setattr(rc, "VicaIntentMsg", _NewMsg)       # ask 칸은 있고 대답 칸은 없는 설치본
    msg = rc.intent_to_msg(VicaIntent(intent="affirm", question_id="q1-3", answer="yes"))
    assert msg.intent == "affirm" and not hasattr(msg, "question_id")
    monkeypatch.setattr(rc, "VicaIntentMsg", _OldMsg)
    assert rc.intent_to_msg(VicaIntent(intent="affirm", answer="yes")).intent == "affirm"

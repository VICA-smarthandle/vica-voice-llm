"""관리자 호출(도움 요청) — 음성 쪽 몫(설계 2026-10-15 7.1절, 10-16 결정)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src import mission_phrases as mp
from src import realtime_intent
from src.langchain_intent_parser import (
    _build_system_prompt,
    _finalize,
    _IntentDraft,
    build_audio_prompt,
    deliver_audio_draft,
)
from src.mission_question import quiet_for_mission
from src.schema import DestinationData, VicaIntent, should_forward_intent
from src.tts_queue import request_for_intent

MISSION_PKG = (Path(__file__).resolve().parents[2]
               / "vica_ros2_ws" / "src" / "vica_mission_manager")
DESTS = [DestinationData(id="r407", name="407호")]


@pytest.fixture(scope="module")
def ml():
    if not (MISSION_PKG / "vica_mission_manager" / "mission_logic.py").exists():
        pytest.skip("옆 저장소 vica_ros2_ws 가 없다 — 계약 대조는 두 저장소가 함께 있는 곳에서")
    sys.path.insert(0, str(MISSION_PKG))
    try:
        from vica_mission_manager import mission_logic
    finally:
        sys.path.remove(str(MISSION_PKG))
    return mission_logic


def _help(reply="직원을 불러 드릴까요?"):
    return VicaIntent(intent="help", reply=reply, safety_flag="normal")


def test_help_is_an_intent_for_both_paths():
    assert _help().intent == "help"
    assert "help" in realtime_intent._INTENTS


@pytest.mark.parametrize("state", ["navigating", "paused", "idle", "waiting", "returning",
                                   "confirming", "asking_next", "awaiting_user"])
def test_voice_never_speaks_the_help_reply(state):
    out = quiet_for_mission(_help(), state, "", 9.0)
    assert out.reply == "" and request_for_intent(out) is None
    assert out.intent == "help" and should_forward_intent(out)


@pytest.mark.parametrize("state, question", [
    ("navigating", "직원의 도움이 필요하신가요?"),
    ("paused", "도움 요청을 취소할까요?"),
])
def test_unknown_answer_to_a_help_question_is_left_to_the_mission(state, question):
    """미션이 못 알아들은 답에 같은 질문을 한 번 다시 묻는다 — LLM 대꾸까지 나가면 두 목소리다. 두"""
    unknown = VicaIntent(intent="unknown", reply="다시 한 번 말씀해 주시겠어요?", safety_flag="normal")
    assert quiet_for_mission(unknown, state, question, 3.0).reply == ""


@pytest.mark.parametrize("convert", ["sound", "text"])
def test_model_reply_is_dropped_by_contract(convert):
    draft = _IntentDraft(intent="help", reply="관리자를 불러 드릴게요.")
    out = (deliver_audio_draft(draft, DESTS) if convert == "sound"
           else _finalize(draft, DESTS))
    assert out.intent == "help" and out.reply == "" and out.need_confirm is False


@pytest.mark.parametrize("local", [False, True])
def test_text_prompt_defines_help(local):
    assert "- help:" in _build_system_prompt(DESTS, local=local)


def test_sound_prompt_defines_help():
    assert "- help:" in build_audio_prompt(DESTS)


def test_help_sentences_match_the_mission(ml):
    assert mp.HELP_SENTENCES == (
        ml.MSG_HELP_CONFIRM, ml.MSG_HELP_SENT, ml.MSG_HELP_COMING,
        ml.MSG_HELP_CANCEL_CONFIRM, ml.MSG_HELP_CANCELED, ml.MSG_HELP_UNAVAILABLE)


def test_help_sentences_are_prewarmed():
    warm = mp.standalone_prewarm([])
    for sentence in mp.HELP_SENTENCES:
        assert sentence in warm, sentence

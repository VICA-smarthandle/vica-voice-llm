"""대기 중 본인 확인의 음성 쪽 — 본인 질문 대답 읽기·LLM 침묵 (2026-11-11 21:21 on 실기 수리).

실기: 본인 질문 "아까 화장실에 가자고 하신 사용자님이신가요?"의 대답 "네"·"아니요"·"그래 맞아"를 LLM 이
navigate 로 읽어 미션이 버렸다(4번 중 3번). 되묻기·본인 질문 중 LLM 이 "네, 말씀하세요." 등을 말해
미션 질문과 두 목소리가 났다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src import mission_phrases as mp
from src.mission_question import identity_answer, quiet_for_mission, spk_gate_question
from src.schema import VicaIntent

IDENTITY = "아까 화장실에 가자고 하신 사용자님이신가요?"
MISSION_PKG = (Path(__file__).resolve().parents[2] / "vica_ros2_ws" / "src" / "vica_mission_manager")


def _nav(dest="df05", need_confirm=False):
    return VicaIntent(intent="navigate", destination_candidate="화장실", matched_destination_id=dest,
                      need_confirm=need_confirm, reply="")


# --- 어떤 질문이 열려 있나 ------------------------------------------------------

def test_gate_question_kinds():
    assert spk_gate_question(IDENTITY, 5.0, "waiting", True) == "identity"
    assert spk_gate_question(mp.SPK_REASK, 5.0, "waiting", True) == "reask"
    assert spk_gate_question(mp.SPK_REASK, 5.0, "waiting_release", True) == "reask"


def test_gate_question_closed():
    assert spk_gate_question(IDENTITY, 25.0, "waiting", True) == ""        # 미션 창 15 s 지남
    assert spk_gate_question(IDENTITY, 5.0, "waiting", False) == ""        # 화자 판정 on 아님
    assert spk_gate_question(IDENTITY, 5.0, "navigating", True) == ""      # 대기 아님
    assert spk_gate_question("네?", 1.0, "waiting", True) == ""


# --- 본인 질문 대답 -------------------------------------------------------------

@pytest.mark.parametrize("text", ["네", "네. 네.", "그래 맞아", "어 그래.", "응"])
def test_identity_yes_becomes_affirm(text):
    out = identity_answer(_nav(), text, "identity")
    assert (out.intent, out.matched_destination_id, out.need_confirm, out.reply) == ("affirm", None, False, "")


@pytest.mark.parametrize("text", ["아니요", "아뇨.", "아니.", "아니 방2로 가자"])
def test_identity_no_becomes_deny(text):
    assert identity_answer(_nav("8156", need_confirm=True), text, "identity").intent == "deny"


@pytest.mark.parametrize("text", ["화장실로 가자", "음", "", "이 뒤로는 잘 안 넘어가네"])
def test_identity_other_words_left_to_llm(text):
    intent = _nav()
    assert identity_answer(intent, text, "identity") is intent


def test_identity_answer_only_after_identity_question():
    intent = _nav()
    assert identity_answer(intent, "네", "reask") is intent
    assert identity_answer(intent, "네", "") is intent


def test_identity_answer_keeps_emergency():
    intent = VicaIntent(intent="unknown", safety_flag="emergency")
    assert identity_answer(intent, "아니", "identity") is intent


# --- 질문이 열린 동안 LLM 은 말하지 않는다 ---------------------------------------

@pytest.mark.parametrize("kind,reply", [("unknown", "네, 말씀하세요."), ("question", "어디 부분에서 막혔는지요?"),
                                        ("clarify", "방1은 목록에 없어요. 뭐가 맞나요?")])
def test_llm_quiet_while_gate_question_open(kind, reply):
    intent = VicaIntent(intent=kind, reply=reply)
    assert quiet_for_mission(intent, "waiting", "", 99.0, spk_on=True, spk_gate="reask").reply == ""
    assert quiet_for_mission(intent, "waiting", "", 99.0, spk_on=True, spk_gate="identity").reply == ""
    assert quiet_for_mission(intent, "waiting", "", 99.0, spk_on=True).reply == reply      # 질문 없으면 그대로


# --- 미션 문장과 같은 글자 -------------------------------------------------------

def test_gate_sentences_match_the_mission():
    if not (MISSION_PKG / "vica_mission_manager" / "mission_logic.py").exists():
        pytest.skip("옆 저장소 vica_ros2_ws 가 없다")
    sys.path.insert(0, str(MISSION_PKG))
    from vica_mission_manager import mission_logic as ml
    assert mp.SPK_REASK == ml.MSG_SPK_REASK
    assert mp.SPK_IDENTITY == ml.MSG_SPK_IDENTITY

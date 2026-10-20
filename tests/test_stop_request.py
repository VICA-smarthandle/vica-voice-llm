"""청취 창 안 멈춤 말 → 주행 중이면 일시정지 (2026-10-06 사용자 결정 (가))."""
import pytest

from src.emergency_filter import detect_stop_request, stop_pause_intent_for
from src.schema import should_forward_intent
from src.tts_queue import request_for_intent


@pytest.mark.parametrize("text, word", [
    ("잠깐 멈춰", "멈춰"),
    ("이동을 멈춰", "멈춰"),
    ("멈춰!", "멈춰"),
    ("일시 정지", "정지"),
    ("일시정지", "일시정지"),
    ("일시정지 해줘", "일시정지"),
    ("스톱", "스톱"),
])
def test_stop_words_are_found(text, word):
    assert detect_stop_request(text) == word


@pytest.mark.parametrize("text", ["화장실로 가자", "행정지원실 어디야", "네. 네. 네.", "", "잠깐만"])
def test_other_speech_is_not_a_stop(text):
    assert detect_stop_request(text) is None


@pytest.mark.parametrize("state", ["navigating", "paused_handle"])
def test_stop_while_guiding_becomes_a_pause_proposal(state):
    intent = stop_pause_intent_for("잠깐 멈춰", state)
    assert intent is not None
    assert intent.intent == "pause"
    assert intent.need_confirm is False
    assert intent.safety_flag == "emergency"
    assert should_forward_intent(intent)
    assert request_for_intent(intent) is None


@pytest.mark.parametrize("state", ["idle", "confirming", "approaching", "returning", "estopped", ""])
def test_stop_outside_guidance_keeps_old_behaviour(state):
    """주행이 아니면 바꾸지 않는다 — 미션이 일시정지를 거절하며 '지금은 안내 중이"""
    assert stop_pause_intent_for("잠깐 멈춰", state) is None


def test_non_stop_speech_while_guiding_is_left_to_the_llm():
    assert stop_pause_intent_for("화장실 아직 멀었어?", "navigating") is None

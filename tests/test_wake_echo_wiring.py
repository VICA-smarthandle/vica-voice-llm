"""호출 응답 메아리 떼기 배선(3-나-1) — ros_node 소스 구조로 확인한다."""
from __future__ import annotations

import ast

from tests.test_one_voice_wiring import _calls, _method


def test_user_text_is_stripped_before_any_rule():
    fn = _method("_on_user_text")
    (strip,) = _calls(fn, "strip_wake_echo")
    assert ast.unparse(strip.args[1]) == "self._question"
    assert ast.unparse(strip.args[2]) == "self._robot_state.dialog_state"
    for later in ("stop_pause_intent_for", "detect_emergency", "parse_intent"):
        assert strip.lineno < min(c.lineno for c in _calls(fn, later)), later


def test_speaker_link_uses_the_raw_text():
    """화자 판정은 호출어 노드가 낸 원문 글자로 찾는다 — 뗀 글자로 찾으면 판정이 끊긴다."""
    fn = _method("_on_user_text")
    (strip,) = _calls(fn, "strip_wake_echo")
    raw = ast.unparse(strip.args[0])
    publishes = _calls(fn, "self._publish_intent")
    assert len(publishes) == 2
    for call in publishes:
        kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
        assert kw.get("link_text") == raw
    (take,) = _calls(_method("_publish_intent"), "self._spk_link.take")
    assert ast.unparse(take.args[0]) == "link_text or text"


def test_audio_path_is_untouched():
    assert not _calls(_method("_on_user_audio"), "strip_wake_echo")

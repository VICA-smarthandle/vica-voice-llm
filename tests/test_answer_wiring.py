"""대답 칸 배선(3-나-2 1단계) — ros_node 소스 구조로 확인한다. 1단계는 기록만 한다."""
from __future__ import annotations

import ast
from pathlib import Path

from tests.test_one_voice_wiring import _calls, _method

SRC = Path(__file__).resolve().parents[1] / "src"


def _kw(call: ast.Call) -> dict:
    return {k.arg: ast.unparse(k.value) for k in call.keywords}


def test_text_path_passes_the_open_question():
    (call,) = _calls(_method("_on_user_text"), "parse_intent")
    assert _kw(call).get("question") == "self._question"


def test_audio_path_passes_the_question_it_started_with():
    fn = _method("_on_user_audio")
    (call,) = _calls(fn, "parse_intent_audio")
    assert _kw(call).get("question") == "question"
    snapshot = [u for u in _calls(fn, "self._audio_turn.update") if "question" in _kw(u)]
    assert snapshot and snapshot[0].lineno < call.lineno      # 그림자 갈래가 같은 질문으로 규칙을 본다


def test_time_based_awaiting_line_is_replaced_only_when_on():
    assert "answer_field.on()" in ast.unparse(_method("_situation_block"))


def test_answer_is_logged_after_publishing():
    fn = _method("_publish_intent")
    (log,) = _calls(fn, "answer_field.log_line")
    assert log.lineno > max(c.lineno for c in _calls(fn, "self._intent_pub.publish"))


def test_transcript_rule_is_only_logged():
    fn = _method("_log_transcript_rule")
    assert _calls(fn, "answer_field.short_answer")
    assert not _calls(fn, "self._intent_pub.publish") and not _calls(fn, "self._publish_intent")
    assert _calls(_method("_on_user_text"), "self._log_transcript_rule")


def test_nothing_new_goes_to_the_mission_yet():
    assert "answer" not in (SRC / "ros_convert.py").read_text()
    for name in ("mission_question.py", "one_voice.py"):
        assert "answer_field" not in (SRC / name).read_text()

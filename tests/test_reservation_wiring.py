"""예약 흐름 노드 배선(설계 2026-11-18 5.3·5.5) — ros_node 는 rclpy 없이 못 띄워 소스 구조로 확인한다."""
from __future__ import annotations

import ast

from tests.test_one_voice_wiring import SRC, _branch_tests, _calls, _method


def test_model_result_goes_through_handle_result():
    assert _calls(_method("_on_user_text"), "handle_result")
    body = ast.unparse(_method("_on_user_text"))
    assert body.index("parse_intent(") < body.index("handle_result(")


def test_situation_block_carries_the_draft_line():
    assert "situation_line(" in ast.unparse(_method("_situation_block"))


def test_scripted_skips_the_three_filters():
    node = _method("_publish_intent")
    assert "scripted" in [a.arg for a in node.args.args + node.args.kwonlyargs]
    for dotted in ("is_relisten_noise", "self._apply_misheard_rule", "quiet_for_mission"):
        call = _calls(node, dotted)[0]
        assert any("scripted" in test for test, _ in _branch_tests(node, call)), dotted


def test_event_is_published_before_the_sentence():
    body = ast.unparse(_method("_reservation_say"))
    assert body.index("self._reservation_pub.publish(") < body.index("self._publish_intent(")
    assert "scripted=True" in body


def test_flow_is_built_only_when_a_switch_is_on():
    src = SRC.read_text()
    assert "text_tools_on()" in src and "audio_tools_on()" in src and "/vica/reservation/event" in src
    init = _method("__init__")
    build = _calls(init, "ReservationFlow")[0]
    tests = [test for test, in_body in _branch_tests(init, build) if in_body]
    assert any("text_tools_on()" in t and "audio_tools_on()" in t for t in tests)
    pub = [c for c in _calls(init, "self.create_publisher") if "/vica/reservation/event" in ast.unparse(c)][0]
    assert any("text_tools_on()" in t for t, in_body in _branch_tests(init, pub) if in_body)


def test_question_record_and_cleared_context_reach_the_flow():
    assert "on_question_record(" in ast.unparse(_method("_on_question"))
    assert SRC.read_text().count("self._reservation.drop()") >= 2


def test_shadow_compare_survives_a_reserve_call():
    assert "ReserveCall" in ast.unparse(_method("_shadow_text"))


def test_broken_facility_file_does_not_kill_the_node():
    """시설 파일이 깨져도 음성 노드는 뜬다 — 흐름을 만드는 자리가 try 안에 있다."""
    init = _method("__init__")
    build = _calls(init, "ReservationFlow")[0]
    tries = [n for n in ast.walk(init) if isinstance(n, ast.Try) and any(build is c for b in n.body for c in ast.walk(b))]
    assert tries and "self._reservation = None" in ast.unparse(tries[0].handlers[0])

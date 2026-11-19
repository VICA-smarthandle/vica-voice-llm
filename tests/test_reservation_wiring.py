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


# ----- 소리 경로(작업 9) --------------------------------------------------
def test_audio_reservation_is_under_the_text_first_guard():
    body = ast.unparse(_method("_on_user_audio"))
    guard = body.index("self._last_text_publish_t > turn_started", body.index("detect_emergency("))
    assert guard < body.index("handle_result(")
    assert _calls(_method("_on_user_audio"), "self._reservation_say")


def test_audio_path_is_not_bypassed_for_reservation_answers():
    body = ast.unparse(_method("_on_user_audio"))
    assert "asked_text" not in body[:body.index("parse_intent_audio(")]      # 예약 질문이라고 소리 경로를 비키지 않는다


def test_audio_log_and_turn_record_survive_a_reserve_call():
    """[RT] 로그 줄과 그림자 비교가 읽는 turn['intent'] 는 ReserveCall 을 의도처럼 읽지 않는다."""
    node = _method("_on_user_audio")
    body = ast.unparse(node)
    assert "intent=reserve_facility" in body
    # 의도 칸을 읽는 [RT] 줄은 ReserveCall 이 아닌 갈래에만 있다
    rt_logs = [c for c in _calls(node, "self.get_logger().info") if "nc={" in ast.unparse(c)]
    assert rt_logs and all(any("ReserveCall" in test and not in_body for test, in_body in _branch_tests(node, c))
                           for c in rt_logs)
    assert "audio_tools_active()" in body


def test_flow_needs_the_question_record_and_hand_over():
    """질문 넘김·질문 기록이 없는 옛 vica_interfaces 에서는 흐름을 만들지 않는다 — 버릴 때를 알 수 없다(검토 I-1)."""
    init = _method("__init__")
    build = _calls(init, "ReservationFlow")[0]
    tests = " ".join(test for test, in_body in _branch_tests(init, build) if in_body)
    assert "VicaQuestionMsg is not None" in tests and "self._hand_over" in tests
    assert "hand_over=" not in ast.unparse(_method("_reservation_say"))

"""한 목소리(3-가) 노드 배선 — ros_node 는 rclpy 없이 못 띄워 소스 구조로 확인한다(2단계와 같은 방식)."""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "ros_node.py"
TREE = ast.parse(SRC.read_text())


def _method(name: str) -> ast.FunctionDef:
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(name)


def _calls(node: ast.AST, dotted: str) -> list[ast.Call]:
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == dotted]


def _parents(root: ast.AST) -> dict:
    return {child: parent for parent in ast.walk(root) for child in ast.iter_child_nodes(parent)}


def _branch_tests(root: ast.AST, target: ast.AST) -> list[tuple[str, bool]]:
    """target 을 감싼 if·조건식마다 (조건 글자, 참 갈래인가)."""
    parents, out, node = _parents(root), [], target
    while node in parents:
        parent = parents[node]
        if isinstance(parent, (ast.If, ast.IfExp)) and node is not parent.test:
            in_body = node in (parent.body if isinstance(parent, ast.If) else [parent.body])
            out.append((ast.unparse(parent.test), in_body))
        node = parent
    return out


def _only_without_hand_over(root: ast.AST, call: ast.Call) -> bool:
    for test, in_body in _branch_tests(root, call):
        if (test == "self._hand_over" and not in_body) or ("not self._hand_over" in test and in_body):
            return True
    return False


def test_hand_over_flag_comes_from_message_fields():
    init = _method("__init__")
    assigns = [n for n in ast.walk(init) if isinstance(n, ast.Assign)
               and any(ast.unparse(t) == "self._hand_over" for t in n.targets)]
    assert assigns and "supports_ask_fields(VicaIntentMsg())" in ast.unparse(assigns[0].value)


def test_hand_over_runs_after_quiet_and_before_publish():
    fn = _method("_publish_intent")
    quiet = _calls(fn, "quiet_for_mission")[0].lineno
    hand = _calls(fn, "hand_over_question")[0].lineno
    publish = min(c.lineno for c in _calls(fn, "self._intent_pub.publish"))
    assert quiet < hand < publish


def test_hand_over_is_skipped_without_ask_fields():
    fn = _method("_publish_intent")
    (call,) = _calls(fn, "hand_over_question")
    assert any("self._hand_over" in test and in_body for test, in_body in _branch_tests(fn, call))


def test_tts_is_gated_by_voice_may_speak():
    fn = _method("_publish_intent")
    (call,) = _calls(fn, "request_for_intent")
    assert any("voice_may_speak(" in test and in_body for test, in_body in _branch_tests(fn, call))


def test_listen_request_only_without_hand_over():
    for name in ("_publish_intent", "_on_listen_state_for_resume"):
        fn = _method(name)
        calls = _calls(fn, "self._listen_pub.publish")
        assert calls, name
        assert all(_only_without_hand_over(fn, c) for c in calls), name


def test_held_resume_goes_out_as_carrier():
    for name in ("_publish_intent", "_on_listen_state_for_resume"):
        fn = _method(name)
        assert any(_calls(c, "held_resume_carrier") for c in _calls(fn, "self._intent_pub.publish")), name


def test_history_keeps_the_question_text_when_handed_over():
    """말한 기록을 쓰지 않는 모드는 넘기기 전 reply 를 대화 기록에 남긴다 — 다음 "네"가 그 질문의 답이 되게."""
    fn = _method("_publish_intent")
    lines = [ast.unparse(c) for c in _calls(fn, "self._history.extend")]
    assert any("AIMessage(history_line)" in line for line in lines)


def test_no_logger_picked_by_condition():
    """rclpy 로거는 같은 호출 줄에서 심각도가 바뀌면 ValueError 로 노드가 죽는다(2단계 검토 C1)."""
    assert not [n for n in ast.walk(TREE) if isinstance(n, ast.Call) and isinstance(n.func, ast.IfExp)]

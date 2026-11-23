"""빔포밍 회전(호출 방향 회전) 실행 중 켜고 끄기 — 2026-11-19 사용자.

앱 대시보드 '빔포밍 회전' 스위치가 `/vica/wake_doa_turn/enable`(SetBool)을 부르고
`/vica/wake_doa_turn/enabled`(latched Bool)를 보고 그린다. 노드는 rclpy 없이 못 띄우므로
배선은 test_sw_aec_node_wiring.py 처럼 AST 로 본다.
"""
import ast
from pathlib import Path

import pytest

from src.wake_doa_turn import (WAKE_DOA_TURN_SERVICE, WAKE_DOA_TURN_STATE_TOPIC,
                               wake_doa_turn_from_env, wake_doa_turn_reply)

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / "src" / "ros_wakeword_node.py").read_text(encoding="utf-8"))


def _method(name):
    return next(n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef) and n.name == name)


def _calls(node, dotted):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == dotted]


def test_names_match_the_app():
    # 앱(VICA_Supervisor supervisor_provider.dart)이 같은 글자로 부른다 — 저장소를 건너는 계약.
    assert WAKE_DOA_TURN_SERVICE == "/vica/wake_doa_turn/enable"
    assert WAKE_DOA_TURN_STATE_TOPIC == "/vica/wake_doa_turn/enabled"


@pytest.mark.parametrize("value", ["on", "1", "true", " ON ", "True"])
def test_env_on(value):
    assert wake_doa_turn_from_env(value) is True


@pytest.mark.parametrize("value", [None, "", "off", "0", "false", "no"])
def test_env_off(value):
    assert wake_doa_turn_from_env(value) is False


def test_reply_text():
    # 앱이 이 글자를 그대로 알림으로 띄운다(캔버스 101번 보드, 사용자 승인).
    assert wake_doa_turn_reply(True) == "빔포밍 회전을 켰습니다"
    assert wake_doa_turn_reply(False) == "빔포밍 회전을 껐습니다"


def test_init_reads_env_through_helper():
    init = _method("__init__")
    (assign,) = [n for n in ast.walk(init) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == "self._wake_doa_turn"]
    assert ast.unparse(assign.value) == "wake_doa_turn_from_env(os.environ.get('VICA_WAKE_DOA_TURN'))"


def test_service_and_latched_state_are_created_before_the_mic_thread():
    init = _method("__init__")
    (service,) = _calls(init, "self.create_service")
    assert [ast.unparse(a) for a in service.args] == [
        "SetBool", "WAKE_DOA_TURN_SERVICE", "self._on_wake_doa_turn_set"]
    (pub,) = [c for c in _calls(init, "self.create_publisher")
              if ast.unparse(c.args[1]) == "WAKE_DOA_TURN_STATE_TOPIC"]
    assert ast.unparse(pub.args[0]) == "Bool"
    assert "TRANSIENT_LOCAL" in ast.unparse(pub.args[2])
    (first_state,) = _calls(init, "self._publish_wake_doa_turn_state")
    (thread,) = _calls(init, "threading.Thread")
    assert pub.lineno < first_state.lineno < thread.lineno


def test_state_is_republished_every_second():
    # rosbridge 는 구독 순간 발행자가 없으면 VOLATILE·BEST_EFFORT 로 구독한다
    # (rosbridge_library subscribers.py). 앱(vica-base)이 음성(vica-drive)보다 먼저 붙는 게
    # 기본 흐름이라 기동 때 한 번 낸 값은 놓친다 — 1 s 마다 다시 낸다(전방 초음파 선례, 11-19 검토).
    init = _method("__init__")
    timers = [ast.unparse(c) for c in _calls(init, "self.create_timer")]
    assert "self.create_timer(1.0, self._publish_wake_doa_turn_state)" in timers


def test_set_callback_updates_publishes_and_replies():
    cb = _method("_on_wake_doa_turn_set")
    assigned = [ast.unparse(t) for n in ast.walk(cb) if isinstance(n, ast.Assign) for t in n.targets]
    for target in ("self._wake_doa_turn", "response.success", "response.message"):
        assert target in assigned, target
    assert len(_calls(cb, "self._publish_wake_doa_turn_state")) == 1
    assert len(_calls(cb, "wake_doa_turn_reply")) == 1
    assert ast.unparse(cb.body[-1]) == "return response"


def test_state_publisher_sends_the_current_value():
    pub = _method("_publish_wake_doa_turn_state")
    (call,) = _calls(pub, "self._pub_wake_doa_turn.publish")
    assert ast.unparse(call.args[0]) == "Bool(data=bool(self._wake_doa_turn))"


def test_turn_off_still_keeps_doa_from_mission():
    # 끄면 방향은 로그에만 — 미션은 /vica/wake_doa 가 와야만 돈다(기존 동작 잠금).
    body = _method("_publish_wake_doa").body
    first = next(n for n in body if not isinstance(n, ast.Expr))  # docstring 건너뜀
    assert isinstance(first, ast.If) and ast.unparse(first.test) == "not self._wake_doa_turn"
    assert isinstance(first.body[-1], ast.Return)


def test_env_example_names_the_service():
    assert "/vica/wake_doa_turn/enable" in (ROOT / ".env.example").read_text(encoding="utf-8")

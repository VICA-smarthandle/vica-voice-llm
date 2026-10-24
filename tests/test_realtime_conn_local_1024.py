"""Realtime 연결 오류면 글자 경로를 바로 로컬 LLM 으로, 응답 지연이면 클라우드 글자 LLM 으로(2026-10-24 사용자).

사용자: "realtime 에 연결이 오류가 나면 바로 로컬로 돌리도록 해주세요. 응답이 지연되는거면 gpt 5.4 mini 로".
예전에는 둘 다 같은 길이었다 — 소리 경로 실패 → 글자 경로가 클라우드(gpt-5.4-mini)부터 부르고, 인터넷이 끊긴
경우 그 호출도 실패(최대 VICA_LLM_CLOUD_TIMEOUT 6 s)한 뒤에야 로컬로 갔다.
- 연결 오류 = 접속 실패, 또는 주고받는 중 연결이 끊김 → RealtimeConnectionError
- 응답 지연 = 연결은 됐는데 timeout_sec 안에 response.done 이 없음 → TimeoutError(그대로)
"""
import types
from pathlib import Path

import numpy as np
import pytest

from src.llm_backend import BackendState, LlmBackendManager, ProbeResult
from src.realtime_intent import RealtimeConnectionError, RealtimeIntentClient
from tests.test_realtime_intent import FakeConn, FakeEvent

PCM = np.zeros(1600, dtype="<i2").tobytes()


class ConnectionClosed(Exception):
    """websockets.exceptions 흉내 — 클래스 이름으로 분류한다(websockets 를 import 하지 않는다)."""


class ConnectionClosedError(ConnectionClosed):
    """비정상 끊김(1006 close frame 없음·1011 keepalive 시간 초과) — 인터넷이 끊긴 경우."""


class ConnectionClosedOK(ConnectionClosed):
    """서버의 정상 종료(1000) — 세션 최대 시간 등. 인터넷 문제가 아니다."""


def client_with(conn_or_exc, timeout=1.0):
    def connect():
        if isinstance(conn_or_exc, BaseException):
            raise conn_or_exc
        return conn_or_exc
    return RealtimeIntentClient(model="fake", timeout_sec=timeout, connect=connect)


class TestConnectionErrorKind:
    @pytest.mark.parametrize("exc", [OSError("Name or service not known"), ConnectionRefusedError("refused"),
                                     TimeoutError("timed out during opening handshake"), RuntimeError("401")])
    def test_connect_failure_is_connection_error(self, exc):
        # 접속 단계의 실패는 무엇이든 연결 오류 — 여는 중 시간 초과(websockets open_timeout)도 지연이 아니다
        with pytest.raises(RealtimeConnectionError):
            client_with(exc).ask(PCM, None, "x")

    def test_send_on_closed_socket_is_connection_error(self):
        conn = FakeConn([])

        def closed(response):
            raise ConnectionClosedError("no close frame received or sent")
        conn.response = types.SimpleNamespace(create=closed)
        client = client_with(conn)
        with pytest.raises(RealtimeConnectionError):
            client.ask(PCM, None, "x")
        assert conn.closed is True and client._conn is None      # 다음 발화는 새로 접속한다

    def test_socket_dropped_while_waiting_is_connection_error(self):
        class Dropping(FakeConn):
            def __iter__(self):
                raise ConnectionClosedError("sent 1011")
        with pytest.raises(RealtimeConnectionError):
            client_with(Dropping([])).ask(PCM, None, "x")

    def test_session_update_on_dead_socket_is_connection_error(self):
        conn = FakeConn([])

        def dead(session):
            raise BrokenPipeError("broken pipe")
        conn.session = types.SimpleNamespace(update=dead)
        with pytest.raises(RealtimeConnectionError):
            client_with(conn).ask(PCM, None, "x")

    def test_connection_error_still_marks_recent_failure(self):
        client = client_with(OSError("no wifi"))
        with pytest.raises(RealtimeConnectionError):
            client.ask(PCM, None, "x")
        assert client.recently_failed(30)                         # 30 s 소리 경로 생략은 그대로


class TestNotConnectionError:
    def test_no_answer_in_time_is_plain_timeout(self):
        class Hanging(FakeConn):
            def __iter__(self):
                import time
                while not self.closed:
                    time.sleep(0.05)
                return iter(())
        with pytest.raises(TimeoutError) as info:
            client_with(Hanging([]), timeout=0.3).ask(PCM, None, "x")
        assert not isinstance(info.value, RealtimeConnectionError)

    def test_server_closed_session_normally_is_not_connection_error(self):
        conn = FakeConn([])

        def closed_ok(response):
            raise ConnectionClosedOK("received 1000 (OK)")
        conn.response = types.SimpleNamespace(create=closed_ok)
        with pytest.raises(ConnectionClosedOK):
            client_with(conn).ask(PCM, None, "x")

    def test_stream_ended_without_answer_is_not_connection_error(self):
        # openai 이벤트 반복은 정상 종료를 조용히 끝낸다 — 서버가 닫은 것이다
        with pytest.raises(RuntimeError, match="응답 없이 닫혔다") as info:
            client_with(FakeConn([])).ask(PCM, None, "x")
        assert not isinstance(info.value, RealtimeConnectionError)

    def test_server_error_event_is_not_connection_error(self):
        conn = FakeConn([FakeEvent("error", error=types.SimpleNamespace(message="server_error"))])
        with pytest.raises(RuntimeError) as info:
            client_with(conn).ask(PCM, None, "x")
        assert not isinstance(info.value, RealtimeConnectionError)

    def test_session_rejected_is_not_connection_error(self):
        conn = FakeConn([])

        def reject(session):
            raise RuntimeError("session rejected")
        conn.session = types.SimpleNamespace(update=reject)
        with pytest.raises(RuntimeError, match="session rejected") as info:
            client_with(conn).ask(PCM, None, "x")
        assert not isinstance(info.value, RealtimeConnectionError)


# ---- 전환 장치: 클라우드를 부르지 않고 바로 LOCAL
class Backend:
    def __init__(self, name):
        self.name, self.calls = name, 0

    def __call__(self, messages):
        self.calls += 1
        return f"{self.name}:ok"


def manager(local=True, probe=ProbeResult.DEAD):
    cloud, loc = Backend("cloud"), (Backend("local") if local else None)
    logs = []
    clock = types.SimpleNamespace(t=1000.0)
    mgr = LlmBackendManager(cloud, loc, lambda: probe, clock=lambda: clock.t,
                            logger=lambda level, msg: logs.append((level, msg)), local_name="midm2-mini")
    return mgr, cloud, loc, logs, clock


class TestGoLocal:
    def test_switches_without_calling_cloud(self):
        mgr, cloud, loc, logs, _ = manager()
        assert mgr.go_local("Realtime 연결 오류") is True
        assert mgr.state is BackendState.LOCAL
        assert mgr.invoke(["s", "화장실"]) == "local:ok" and cloud.calls == 0
        assert logs[-1] == ("warning", "[LLM] Realtime 연결 오류 → 클라우드 건너뛰고 로컬(midm2-mini)로")

    def test_without_local_stays_cloud(self):
        mgr, cloud, _, logs, _ = manager(local=False)
        assert mgr.go_local("Realtime 연결 오류") is False
        assert mgr.state is BackendState.CLOUD and logs == []

    def test_already_local_is_noop(self):
        mgr, *_ = manager()
        mgr.go_local("Realtime 연결 오류")
        assert mgr.go_local("Realtime 연결 오류") is False
        assert mgr.state is BackendState.LOCAL

    def test_returns_to_cloud_by_the_usual_rule(self):
        # 복귀는 클라우드 실패 때와 같다 — 확인(probe) 성공 + 주행 끝·정지
        mgr, _, _, _, clock = manager(probe=ProbeResult.ALIVE)
        mgr.go_local("Realtime 연결 오류")
        mgr.on_goal_event("goal_sent")
        clock.t += 30
        assert mgr.tick() is BackendState.LOCAL                   # 주행 중이면 그대로 로컬
        mgr.on_goal_event("goal_succeeded")
        assert mgr.state is BackendState.CLOUD


# ---- 노드: 소리 경로 실패의 종류로 글자 경로의 첫 백엔드를 고른다(rclpy 없이 소스 글자로 본다)
def test_node_sends_text_path_local_only_on_connection_error():
    src = (Path(__file__).resolve().parents[1] / "src" / "ros_node.py").read_text(encoding="utf-8")
    body = src[src.index("    def _on_user_audio"):src.index("    def _situation_block")]
    handler = body[body.index("except Exception as exc:"):body.index("finally:")]
    assert "isinstance(exc, RealtimeConnectionError)" in handler
    assert "self._backend.go_local(" in handler
    assert "TimeoutError" not in handler                         # 지연은 글자 경로가 클라우드부터


def test_node_preloads_local_model_at_start_even_when_cloud_is_up():
    # 2026-10-24 사용자 "미리 올려주세요" — Realtime 연결 오류 때 첫 답이 적재(약 11 s)를 기다리지 않게.
    # 예전에는 워밍업 중 클라우드가 실패해 LOCAL 일 때만 올렸다.
    src = (Path(__file__).resolve().parents[1] / "src" / "ros_node.py").read_text(encoding="utf-8")
    body = src[src.index("    def _warmup_llm"):]
    body = body[:body.index("\n    def ", 10)]
    assert "self._backend.warm_local_async()" in body
    assert "if self._backend.state is BackendState.LOCAL:" not in body

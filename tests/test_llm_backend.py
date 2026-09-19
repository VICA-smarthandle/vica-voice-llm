"""클라우드→로컬 LLM 자동 전환 담당 모듈 시험 (네트워크·ROS 없음).

정본: docs/superpowers/specs/2026-09-19-llm-local-fallback-design.md §4.1·§6.1.
가짜 백엔드·접속 확인 함수·시계로 규칙을 고정한다.
"""
import pytest

from src.llm_backend import (
    BackendState,
    FailureKind,
    LlmBackendManager,
    ProbeResult,
    classify_failure,
)


class FakeBackend:
    """호출 기록을 남기고, fail_with 가 있으면 그 예외를 던진다."""

    def __init__(self, name: str, fail_with: Exception | None = None):
        self.name = name
        self.fail_with = fail_with
        self.calls: list = []

    def __call__(self, messages):
        self.calls.append(list(messages))
        if self.fail_with is not None:
            raise self.fail_with
        return f"{self.name}:ok"


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, sec: float) -> None:
        self.t += sec


class FakeProbe:
    def __init__(self, result: ProbeResult = ProbeResult.DEAD):
        self.result = result
        self.calls = 0

    def __call__(self) -> ProbeResult:
        self.calls += 1
        return self.result


class StatusError(Exception):
    """openai.APIStatusError 흉내 — status_code 속성만 있으면 된다."""

    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class APIConnectionError(Exception):
    """openai.APIConnectionError 와 같은 클래스 이름 — 이름으로 분류한다."""


def make(cloud_fail=None, local=True, probe=None, clock=None, logs=None, **kw):
    cloud = FakeBackend("cloud", cloud_fail)
    local_b = FakeBackend("local") if local else None
    probe = probe or FakeProbe()
    clock = clock or FakeClock()
    logs = logs if logs is not None else []
    mgr = LlmBackendManager(
        cloud, local_b, probe, clock=clock,
        logger=lambda level, msg: logs.append((level, msg)),
        local_name="gemma4-e2b-text", **kw,
    )
    return mgr, cloud, local_b, probe, clock, logs


MSGS = ["system", "화장실로 안내해줘"]


class TestInvoke:
    def test_cloud_ok_stays_cloud(self):
        mgr, cloud, local, *_ = make()
        assert mgr.invoke(MSGS) == "cloud:ok"
        assert mgr.state is BackendState.CLOUD
        assert mgr.heartbeat_enabled is True
        assert local.calls == []

    def test_cloud_fail_retries_same_messages_on_local(self):
        mgr, cloud, local, _, _, logs = make(cloud_fail=APIConnectionError("boom"))
        assert mgr.invoke(MSGS) == "local:ok"
        assert local.calls == [MSGS]  # 같은 발화를 그대로 재처리
        assert mgr.state is BackendState.LOCAL
        assert mgr.heartbeat_enabled is False
        assert logs[-1][0] == "warning"
        assert logs[-1][1] == (
            "[LLM] 클라우드 실패(연결 오류: boom) → 로컬(gemma4-e2b-text)로 대피. 같은 발화 재처리")

    def test_both_fail_raises_local_error(self):
        mgr, cloud, local, *_ = make(cloud_fail=APIConnectionError("boom"))
        local.fail_with = RuntimeError("ollama down")
        with pytest.raises(RuntimeError, match="ollama down"):
            mgr.invoke(MSGS)
        assert mgr.state is BackendState.LOCAL  # 다음 tick 에 확인을 이어 간다

    def test_no_local_means_old_behaviour(self):
        mgr, cloud, local, *_ = make(cloud_fail=APIConnectionError("boom"), local=False)
        assert mgr.has_local is False
        with pytest.raises(APIConnectionError):
            mgr.invoke(MSGS)
        assert mgr.state is BackendState.CLOUD


class TestClassify:
    @pytest.mark.parametrize("exc, kind", [
        (APIConnectionError("x"), FailureKind.CONNECTION),
        (TimeoutError("x"), FailureKind.CONNECTION),
        (ConnectionError("x"), FailureKind.CONNECTION),
        (StatusError(401), FailureKind.AUTH),
        (StatusError(403), FailureKind.AUTH),
        (StatusError(429), FailureKind.SERVER),
        (StatusError(503), FailureKind.SERVER),
        (StatusError(400), FailureKind.REQUEST),
        (ValueError("x"), FailureKind.REQUEST),
    ])
    def test_kinds(self, exc, kind):
        assert classify_failure(exc) is kind

    def test_auth_and_request_log_as_error_but_still_switch(self):
        for exc in (StatusError(401), ValueError("bad")):
            mgr, cloud, local, _, _, logs = make(cloud_fail=exc)
            assert mgr.invoke(MSGS) == "local:ok"
            assert mgr.state is BackendState.LOCAL
            assert logs[-1][0] == "error"


def switched(probe_result=ProbeResult.ALIVE, **kw):
    """클라우드 실패 한 번으로 LOCAL 이 된 관리자를 돌려준다."""
    probe = FakeProbe(probe_result)
    mgr, cloud, local, _, clock, logs = make(
        cloud_fail=APIConnectionError("boom"), probe=probe, **kw)
    mgr.invoke(MSGS)
    assert mgr.state is BackendState.LOCAL
    return mgr, cloud, local, probe, clock, logs


class TestProbeAndReturn:
    def test_no_probe_before_interval(self):
        mgr, _, _, probe, clock, _ = switched()
        clock.advance(29)
        mgr.tick()
        assert probe.calls == 0

    def test_probe_after_interval_and_reschedule_on_dead(self):
        mgr, _, _, probe, clock, _ = switched(ProbeResult.DEAD)
        clock.advance(30)
        mgr.tick()
        assert probe.calls == 1
        assert mgr.cloud_ready is False
        clock.advance(29)
        mgr.tick()
        assert probe.calls == 1  # 다음 30초까지 재확인 없음
        clock.advance(1)
        mgr.tick()
        assert probe.calls == 2

    def test_alive_sets_ready_and_returns_when_idle(self):
        mgr, _, _, _, clock, logs = switched()
        clock.advance(30)
        assert mgr.tick() is BackendState.CLOUD  # 주행 중이 아니므로 즉시 복귀
        assert ("info", "[LLM] 클라우드 살아남(확인 1회째). 주행 끝나면 복귀") in logs
        assert logs[-1] == ("info", "[LLM] tick → 클라우드 복귀")
        assert mgr.heartbeat_enabled is True

    def test_alive_but_run_active_waits_for_end_event(self):
        mgr, _, _, _, clock, logs = switched()
        mgr.on_goal_event("goal_sent")
        clock.advance(30)
        assert mgr.tick() is BackendState.LOCAL
        assert mgr.cloud_ready is True
        mgr.on_goal_event("goal_paused")  # 일시정지는 주행 유지
        assert mgr.state is BackendState.LOCAL
        mgr.on_goal_event("goal_succeeded")
        assert mgr.state is BackendState.CLOUD
        assert logs[-1] == ("info", "[LLM] goal_succeeded → 클라우드 복귀")

    @pytest.mark.parametrize("end_event", [
        "goal_succeeded", "goal_failed", "goal_rejected", "goal_canceled",
        "return_home_succeeded", "return_home_failed", "return_home_canceled", "state_idle",
    ])
    def test_every_end_event_opens_return(self, end_event):
        mgr, _, _, _, clock, _ = switched()
        mgr.on_goal_event("return_home_sent" if end_event.startswith("return") else "goal_accepted")
        clock.advance(30)
        mgr.tick()
        assert mgr.state is BackendState.LOCAL
        mgr.on_goal_event(end_event)
        assert mgr.state is BackendState.CLOUD

    @pytest.mark.parametrize("moving, paused", [(True, False), (False, True)])
    def test_moving_or_paused_blocks_return(self, moving, paused):
        mgr, _, _, _, clock, _ = switched()
        mgr.on_robot_state(is_moving=moving, is_paused=paused)
        clock.advance(30)
        assert mgr.tick() is BackendState.LOCAL
        mgr.on_robot_state(is_moving=False, is_paused=False)  # 멈추면 복귀
        assert mgr.state is BackendState.CLOUD

    def test_auth_failed_probe_never_returns(self):
        mgr, _, _, _, clock, logs = switched(ProbeResult.AUTH_FAILED)
        clock.advance(30)
        assert mgr.tick() is BackendState.LOCAL
        assert mgr.cloud_ready is False
        assert logs[-1][0] == "error"

    def test_local_invoke_while_local(self):
        mgr, cloud, local, *_ = switched()
        cloud.calls.clear()
        assert mgr.invoke(MSGS) == "local:ok"
        assert cloud.calls == []  # LOCAL 에선 클라우드를 부르지 않는다


class TestFlapping:
    def test_interval_doubles_when_refailing_within_window(self):
        mgr, cloud, local, probe, clock, _ = switched()
        seen = []
        for expected in (60, 120, 240, 300, 300):
            clock.advance(mgr.probe_interval)
            mgr.tick()                       # ALIVE → 복귀
            assert mgr.state is BackendState.CLOUD
            clock.advance(10)                # 복귀 10초 만에 또 실패
            cloud.fail_with = APIConnectionError("again")
            mgr.invoke(MSGS)
            seen.append(mgr.probe_interval)
            assert mgr.probe_interval == expected
        assert seen == [60, 120, 240, 300, 300]

    def test_interval_resets_when_refailing_after_window(self):
        mgr, cloud, local, probe, clock, _ = switched()
        clock.advance(30)
        mgr.tick()
        assert mgr.state is BackendState.CLOUD
        clock.advance(301)                   # 창(300초) 밖
        cloud.fail_with = APIConnectionError("again")
        mgr.invoke(MSGS)
        assert mgr.probe_interval == 30


class TestStartLocal:
    def test_start_local_enters_local_and_warms(self):
        warmed = []
        mgr, cloud, local, probe, clock, logs = make(warm_local=lambda: warmed.append(1))
        mgr.start_local("워밍업 실패")
        assert mgr.state is BackendState.LOCAL
        assert mgr.heartbeat_enabled is False
        mgr.warm_local_async()
        import time as _t
        for _ in range(50):
            if warmed:
                break
            _t.sleep(0.01)
        assert warmed == [1]

    def test_start_local_without_local_is_noop(self):
        mgr, *_ = make(local=False)
        mgr.start_local("워밍업 실패")
        assert mgr.state is BackendState.CLOUD


import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from src.llm_backend import http_probe, ollama_warm, parse_goal_event


class _Handler(BaseHTTPRequestHandler):
    status = 200
    seen: list = []

    def do_GET(self):
        self.send_response(self.status)
        self.end_headers()
        self.wfile.write(b"{}")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        _Handler.seen.append(json.loads(self.rfile.read(length)))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *a):  # 시험 출력 잡음 제거
        pass


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield srv
    srv.shutdown()
    srv.server_close()


class TestHelpers:
    @pytest.mark.parametrize("status, result", [
        (200, ProbeResult.ALIVE), (401, ProbeResult.AUTH_FAILED),
        (403, ProbeResult.AUTH_FAILED), (429, ProbeResult.DEAD), (503, ProbeResult.DEAD),
    ])
    def test_http_probe_status(self, server, status, result):
        _Handler.status = status
        url = f"http://127.0.0.1:{server.server_port}/v1/models"
        assert http_probe(url, headers={"Authorization": "Bearer x"}) is result

    def test_http_probe_dead_when_unreachable(self):
        assert http_probe("http://127.0.0.1:9/v1/models", timeout_sec=0.5) is ProbeResult.DEAD

    def test_ollama_warm_posts_keep_alive(self, server):
        _Handler.status = 200
        _Handler.seen.clear()
        logs = []
        ollama_warm(f"http://127.0.0.1:{server.server_port}", "gemma4-e2b-text",
                    logger=lambda lv, m: logs.append((lv, m)))
        assert _Handler.seen == [{"model": "gemma4-e2b-text", "keep_alive": -1}]
        assert logs == [("info", "[LLM] 로컬 모델 예열 완료: gemma4-e2b-text")]

    def test_ollama_warm_failure_is_logged_not_raised(self):
        logs = []
        ollama_warm("http://127.0.0.1:9", "m", timeout_sec=0.5,
                    logger=lambda lv, m: logs.append((lv, m)))
        assert logs[0][0] == "warning"

    @pytest.mark.parametrize("data, event", [
        ('{"event": "goal_succeeded", "map_id": "x"}', "goal_succeeded"),
        ('{"map_id": "x"}', None),
        ("not json", None),
        ("", None),
    ])
    def test_parse_goal_event(self, data, event):
        assert parse_goal_event(data) == event

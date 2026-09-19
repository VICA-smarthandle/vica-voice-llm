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

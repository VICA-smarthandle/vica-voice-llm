# LLM 클라우드→로컬 자동 폴백 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 클라우드 LLM(gpt-5.4-mini)이 닿지 않으면 젯슨 안의 Ollama 모델 `gemma4-e2b-text` 로 자동 전환하고, 클라우드가 살아나면 주행이 끝난 뒤에만 되돌아가며, 그 상태를 로그와 앱 진단 화면에 보인다.

**Architecture:** ROS 를 모르는 순수 모듈 `src/llm_backend.py` 의 `LlmBackendManager` 가 클라우드·로컬 두 호출 함수와 상태(CLOUD/LOCAL)를 든다. 파서 `parse_intent` 는 이 관리자를 거쳐 LLM 을 부르고, ROS 노드는 주행 사건·로봇 상태를 관리자에 흘려 넣고 1 Hz 로 `tick()`·생존 신호를 낸다. 진단 노드는 생존 신호 토픽의 주기만 보고 `LLM_CLOUD_OFFLINE` 결함을 앱에 띄운다.

**Tech Stack:** Python 3.10, pydantic, langchain-openai(`ChatOpenAI`), langchain-ollama(`ChatOllama`), Ollama 0.30.6(젯슨, CUDA), rclpy(Humble), pytest(젯슨 `.venv`), ROS 저장소는 `vica_system_monitor`(ament_python).

**Spec:** `docs/superpowers/specs/2026-09-19-llm-local-fallback-design.md`

## 용어(쉬운 말)

| 문서의 말 | 뜻 |
| --- | --- |
| 자동 전환(절체) | 클라우드가 죽으면 로컬로, 살아나면 클라우드로 스위치를 넘기는 것 |
| 전환 담당 모듈 | 그 스위치 역할을 하는 코드 `src/llm_backend.py` 의 `LlmBackendManager` |
| 왔다 갔다 반복(플래핑) | 인터넷이 붙었다 끊겼다를 반복해 스위치가 계속 넘어가는 상태. 복귀 후 5분 안에 또 끊기면 다음 확인을 더 늦게 해서 막는다 |
| 접속 확인(probe) | 로컬 상태에서 30초마다 "클라우드 살아 있나" 문만 두드리는 것. 답을 시키지 않아 요금 0 |
| 생존 신호(heartbeat) | 클라우드로 잘 돌 때만 1초마다 내는 "이상 없음" 신호. 끊기면 앱 진단에 경고 |
| 미리 올리기(예열·warm) | 로컬 모델을 메모리에 먼저 실어 첫 답이 늦지 않게 하는 것 |
| 백엔드 | 답을 만드는 쪽 — 클라우드 GPT 또는 로컬 gemma |
| 가짜 백엔드·가짜 시계 | 시험용 대역. 진짜 GPT·시간 대신 쓴다 |

## Global Constraints

- 모든 실행·시험은 젯슨에서 한다. 파이썬은 `vica-voice-llm/.venv/bin/python`.
- 폴백 모델 이름(`VICA_LLM_FALLBACK_MODEL`)이 비어 있으면 **지금과 완전히 같은 동작**이어야 한다.
- 클라우드 대기 시간: 폴백이 켜졌을 때 `VICA_LLM_CLOUD_TIMEOUT` 기본 6초·재시도 0. 꺼졌을 때 기존 15초·재시도 1.
- 확인 간격: 기본 30초, 복귀 후 300초 안 재실패 시 2배(30→60→120→240→300), 300초 밖이면 30.
- 복귀 조건: `LOCAL` ∧ `cloud_ready` ∧ ¬`run_active` ∧ ¬`is_moving` ∧ ¬`is_paused`.
- 주행 시작 사건: `goal_sent`·`goal_accepted`·`return_home_sent`. 종료 사건: `goal_succeeded`·`goal_failed`·`goal_rejected`·`goal_canceled`·`return_home_succeeded`·`return_home_failed`·`return_home_canceled`·`state_idle`. `goal_paused` 는 유지.
- 생존 신호 토픽 `/vica/llm_cloud_alive`(std_msgs/Bool, True), CLOUD 상태일 때만 1 Hz. 진단 결함 코드 `LLM_CLOUD_OFFLINE`, 컴포넌트 `voice`, 등급 WARN, `optional: true`.
- 사용자 멘트는 추가하지 않는다(멘트 최소주의). 전환·복귀는 로그로만.
- 로그 문구(정확히): `[LLM] 클라우드 실패({분류}: {예외}) → 로컬({모델})로 대피. 같은 발화 재처리` / `[LLM] 클라우드 살아남(확인 {n}회째). 주행 끝나면 복귀` / `[LLM] {계기} → 클라우드 복귀`.
- commit 은 GOVERNANCE §4 에 따라 **사용자가 허락한 경우에만** 실행한다. 허락 전에는 각 Task 의 커밋 단계를 건너뛰고 변경을 쌓아 둔다.
- 코드 주석·문서는 저장소의 기존 문체(평서형 한국어)를 따른다.
- 젯슨 실기(Task 11)는 사용자 승인 뒤에만 한다(AGENTS §5·§9).

---

## 파일 구조

음성 저장소 `vica-voice-llm` (브랜치 `feat/llm-local-fallback`, 이미 생성됨):

| 파일 | 책임 |
| --- | --- |
| `src/llm_backend.py` (신규) | 자동 전환 상태표, 실패 분류, 확인·예열 도우미, 주행 사건 JSON 파싱. ROS·LangChain 무관 |
| `tests/test_llm_backend.py` (신규) | 위 모듈의 규칙 전부 |
| `src/langchain_intent_parser.py` (수정) | `_get_structured_llm` 에 timeout 인자, 관리자 싱글턴 `get_backend_manager()`, `parse_intent(model=None)` 이 관리자를 거침 |
| `tests/test_parser_backend.py` (신규) | 파서가 관리자를 쓰는 경로·명시 모델 우회 경로 |
| `src/schema.py`·`src/ros_convert.py` (수정) | `RobotState.is_paused` |
| `src/ros_node.py` (수정) | `/vica_goal_event` 구독, 백엔드 스레드(tick + 생존 신호), 워밍업 실패 시 로컬 예열 |
| `src/main.py` (수정) | 발화마다 `tick()` |
| `launch/vica_voice.launch.py` (수정) | `ollama serve` 프로세스 |
| `ollama/Modelfile.gemma4-e2b-text` (신규) | 텍스트 전용 모델 조립 설명서 |
| `scripts/setup_local_llm.sh` (신규) | HF 가중치 받기 → 모델 조립 |
| `.env.example`·`docs/jetson-setup.md` (수정) | 설정·설치 안내 |

ROS 저장소 `vica_ros2_ws` (worktree `/home/ji_w/wt-llmprobe`, 브랜치 `feat/llm-cloud-probe`):

| 파일 | 책임 |
| --- | --- |
| `src/vica_system_monitor/config/probes.yaml` | `voice_llm_cloud` 프로브 |
| `src/vica_system_monitor/vica_system_monitor/fault_catalog.py` | `LLM_CLOUD_OFFLINE` 문구 |

---

### Task 1: 자동 전환 상태표 — 호출·대피 (`LlmBackendManager.invoke`)

**Files:**
- Create: `src/llm_backend.py`
- Test: `tests/test_llm_backend.py`

**Interfaces:**
- Produces:
  - `class BackendState(enum.Enum)`: `CLOUD = "cloud"`, `LOCAL = "local"`
  - `class FailureKind(enum.Enum)`: `CONNECTION = "연결 오류"`, `SERVER = "서버 오류"`, `AUTH = "인증 실패"`, `REQUEST = "요청 오류"`
  - `def classify_failure(exc: BaseException) -> FailureKind`
  - `class LlmBackendManager`:
    - `__init__(self, cloud: Callable[[Sequence[Any]], Any], local: Optional[Callable[[Sequence[Any]], Any]], probe: Callable[[], "ProbeResult"], *, warm_local: Optional[Callable[[], None]] = None, clock: Callable[[], float] = time.monotonic, logger: Callable[[str, str], None] = _stderr_logger, local_name: str = "local", probe_interval_sec: float = 30.0, probe_interval_max_sec: float = 300.0, flap_window_sec: float = 300.0)`
    - 속성 `state: BackendState`, `cloud_ready: bool`, `run_active: bool`, `is_moving: bool`, `is_paused: bool`, `probe_interval: float`, `next_probe_at: float`, `last_return_at: Optional[float]`
    - `has_local -> bool`, `heartbeat_enabled -> bool`
    - `invoke(self, messages: Sequence[Any]) -> Any`
  - `class ProbeResult(enum.Enum)`: `ALIVE`, `DEAD`, `AUTH_FAILED` (Task 2 가 씀; 여기서 정의)

- [ ] **Step 1: 실패 시험 작성 — 파일 골격과 Task 1 시험 4개**

`tests/test_llm_backend.py`:

```python
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
```

- [ ] **Step 2: 실패 확인**

Run: `cd /home/ji_w/VICA-smarthandle/vica-voice-llm && .venv/bin/python -m pytest tests/test_llm_backend.py -q`
Expected: `ImportError`/`ModuleNotFoundError: No module named 'src.llm_backend'`

- [ ] **Step 3: 모듈 작성 — 상태기계와 실패 분류**

`src/llm_backend.py`:

```python
"""클라우드→로컬 LLM 자동 전환(폴백) 담당 모듈. ROS·LangChain 을 모르는 순수 로직.

정본 설계: docs/superpowers/specs/2026-09-19-llm-local-fallback-design.md
비유: 한전(클라우드)과 발전기(로컬) 사이의 자동 전환 스위치. 스위치는 전기가
지나가는 자리, 즉 LLM 호출 바로 앞에 둔다.

규칙 요약:
- CLOUD 에서 클라우드 호출이 실패하면 **같은 messages 를 로컬로 재호출**하고 LOCAL 로.
- LOCAL 에서는 probe_interval 마다 클라우드를 확인한다(토큰 소모 없음).
- 살아나도 바로 안 돌아간다. 주행이 끝나고(run_active False) 멈춰 있을 때만 CLOUD 로.
- 복귀 뒤 flap_window 안에 또 실패하면 확인 간격을 2배씩 늘린다(상한 있음).
"""
from __future__ import annotations

import enum
import json
import sys
import threading
import time
from typing import Any, Callable, Optional, Sequence


class BackendState(enum.Enum):
    CLOUD = "cloud"
    LOCAL = "local"


class ProbeResult(enum.Enum):
    ALIVE = "alive"          # 클라우드에 닿고 인증도 됨
    DEAD = "dead"            # 연결 불가·timeout·5xx·429
    AUTH_FAILED = "auth"     # 닿지만 401/403 — 키 문제. 복귀하지 않는다


class FailureKind(enum.Enum):
    CONNECTION = "연결 오류"
    SERVER = "서버 오류"
    AUTH = "인증 실패"
    REQUEST = "요청 오류"


RUN_START_EVENTS = frozenset({"goal_sent", "goal_accepted", "return_home_sent"})
RUN_END_EVENTS = frozenset({
    "goal_succeeded", "goal_failed", "goal_rejected", "goal_canceled",
    "return_home_succeeded", "return_home_failed", "return_home_canceled",
    "state_idle",
})


def classify_failure(exc: BaseException) -> FailureKind:
    """예외를 네 갈래로 나눈다. openai 패키지를 import 하지 않고 모양으로 판별한다.

    - openai.APIStatusError 계열은 status_code 를 가진다.
    - openai.APIConnectionError / APITimeoutError 는 클래스 이름으로 본다
      (Exception 직계라 isinstance 로는 못 잡는다).
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status in (401, 403):
            return FailureKind.AUTH
        if status == 429 or status >= 500:
            return FailureKind.SERVER
        return FailureKind.REQUEST
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return FailureKind.CONNECTION
    if type(exc).__name__ in ("APIConnectionError", "APITimeoutError"):
        return FailureKind.CONNECTION
    return FailureKind.REQUEST


def _stderr_logger(level: str, msg: str) -> None:
    print(f"[{level}] {msg}", file=sys.stderr)


class LlmBackendManager:
    """CLOUD/LOCAL 자동 전환 스위치. 스레드 두 개(노드 콜백·백엔드 루프)가 만지므로 잠근다."""

    def __init__(
        self,
        cloud: Callable[[Sequence[Any]], Any],
        local: Optional[Callable[[Sequence[Any]], Any]],
        probe: Callable[[], ProbeResult],
        *,
        warm_local: Optional[Callable[[], None]] = None,
        clock: Callable[[], float] = time.monotonic,
        logger: Callable[[str, str], None] = _stderr_logger,
        local_name: str = "local",
        probe_interval_sec: float = 30.0,
        probe_interval_max_sec: float = 300.0,
        flap_window_sec: float = 300.0,
    ) -> None:
        self._cloud = cloud
        self._local = local
        self._probe = probe
        self._warm_local = warm_local
        self._clock = clock
        self._log = logger
        self._local_name = local_name
        self._interval_min = probe_interval_sec
        self._interval_max = probe_interval_max_sec
        self._flap_window = flap_window_sec
        self._lock = threading.RLock()

        self.state = BackendState.CLOUD
        self.cloud_ready = False
        self.run_active = False
        self.is_moving = False
        self.is_paused = False
        self.probe_interval = probe_interval_sec
        self.next_probe_at = 0.0
        self.last_return_at: Optional[float] = None
        self.probe_ok_streak = 0

    # ----- 조회 ---------------------------------------------------------
    @property
    def has_local(self) -> bool:
        return self._local is not None

    @property
    def heartbeat_enabled(self) -> bool:
        return self.state is BackendState.CLOUD

    # ----- 호출 ---------------------------------------------------------
    def invoke(self, messages: Sequence[Any]) -> Any:
        """현재 상태의 백엔드로 호출한다. 클라우드 실패는 로컬 재호출로 받는다.

        로컬까지 실패하면 예외를 그대로 올린다 — 파서가 LLM_UNAVAILABLE 로 받는다.
        로컬이 없으면(폴백 꺼짐) 클라우드 예외를 그대로 올린다 = 예전 동작.
        """
        with self._lock:
            use_cloud = self.state is BackendState.CLOUD
        if use_cloud:
            try:
                return self._cloud(messages)
            except Exception as exc:
                if not self.has_local:
                    raise
                self._switch_to_local(exc)
        assert self._local is not None
        return self._local(messages)

    # ----- 내부 ---------------------------------------------------------
    def _switch_to_local(self, exc: BaseException) -> None:
        kind = classify_failure(exc)
        level = "error" if kind in (FailureKind.AUTH, FailureKind.REQUEST) else "warning"
        self._log(level, f"[LLM] 클라우드 실패({kind.value}: {exc}) → 로컬({self._local_name})로 대피. "
                         "같은 발화 재처리")
        self._enter_local()

    def _enter_local(self) -> None:
        with self._lock:
            now = self._clock()
            if self.last_return_at is not None and now - self.last_return_at < self._flap_window:
                self.probe_interval = min(self.probe_interval * 2, self._interval_max)
                self._log("warning", f"[LLM] 복귀 {now - self.last_return_at:.0f}초 만에 재실패 — "
                                     f"클라우드 확인 간격 {self.probe_interval:.0f}초로")
            else:
                self.probe_interval = self._interval_min
            self.state = BackendState.LOCAL
            self.cloud_ready = False
            self.probe_ok_streak = 0
            self.next_probe_at = now + self.probe_interval
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_llm_backend.py -q`
Expected: `14 passed` (TestInvoke 4 + TestClassify 파라미터 9 + 1)

- [ ] **Step 5: 커밋(사용자 허락 시)**

```bash
git add src/llm_backend.py tests/test_llm_backend.py
git commit -m "feat(voice): LLM 전환 담당 모듈 — 클라우드 실패 시 같은 발화를 로컬로 재처리"
```

---

### Task 2: 확인·복귀·왔다 갔다 반복 규칙 (`tick`·`on_goal_event`·`on_robot_state`)

**Files:**
- Modify: `src/llm_backend.py` (Task 1 의 클래스에 메서드 추가)
- Test: `tests/test_llm_backend.py`

**Interfaces:**
- Consumes: Task 1 의 `LlmBackendManager`, `ProbeResult`, `RUN_START_EVENTS`, `RUN_END_EVENTS`
- Produces:
  - `tick(self, now: Optional[float] = None) -> BackendState`
  - `on_goal_event(self, event: str) -> None`
  - `on_robot_state(self, is_moving: bool, is_paused: bool) -> None`
  - `start_local(self, reason: str) -> None` (처음부터 로컬로 시작 + 예열)
  - `warm_local_async(self) -> None`

- [ ] **Step 1: 실패 시험 작성**

`tests/test_llm_backend.py` 끝에 추가:

```python
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
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_llm_backend.py -q`
Expected: `AttributeError: 'LlmBackendManager' object has no attribute 'tick'` 계열 실패 다수

- [ ] **Step 3: 메서드 추가**

`src/llm_backend.py` 의 `LlmBackendManager` 에 (`_enter_local` 아래) 추가:

```python
    # ----- 주기 확인·복귀 ---------------------------------------------
    def tick(self, now: Optional[float] = None) -> BackendState:
        """1초쯤마다 부른다. LOCAL 이면 간격에 맞춰 클라우드를 확인하고 복귀를 시도한다."""
        with self._lock:
            now = self._clock() if now is None else now
            due = self.state is BackendState.LOCAL and now >= self.next_probe_at
        if due:
            result = self._probe()          # 잠금 밖에서 부른다 — 최대 3초 걸린다
            with self._lock:
                self.next_probe_at = now + self.probe_interval
                if result is ProbeResult.ALIVE:
                    self.probe_ok_streak += 1
                    if not self.cloud_ready:
                        self.cloud_ready = True
                        self._log("info", f"[LLM] 클라우드 살아남(확인 {self.probe_ok_streak}회째). "
                                          "주행 끝나면 복귀")
                elif result is ProbeResult.AUTH_FAILED:
                    self.cloud_ready = False
                    self.probe_ok_streak = 0
                    self._log("error", "[LLM] 클라우드는 닿지만 인증 실패 — 키를 확인하세요. 복귀 보류")
                else:
                    self.cloud_ready = False
                    self.probe_ok_streak = 0
        self._maybe_return("tick")
        return self.state

    def on_goal_event(self, event: str) -> None:
        """미션 매니저의 /vica_goal_event 사건 이름. 주행 시작/종료를 추적한다."""
        with self._lock:
            if event in RUN_START_EVENTS:
                self.run_active = True
                return
            if event in RUN_END_EVENTS:
                self.run_active = False
        if event in RUN_END_EVENTS:
            self._maybe_return(event)

    def on_robot_state(self, is_moving: bool, is_paused: bool) -> None:
        """/vica/robot_state 의 보조 신호. 노드 재시작으로 run_active 를 놓친 경우의 안전띠."""
        with self._lock:
            self.is_moving = bool(is_moving)
            self.is_paused = bool(is_paused)
        self._maybe_return("robot_state")

    def start_local(self, reason: str) -> None:
        """시작 워밍업 실패 등으로 처음부터 로컬로 갈 때."""
        if not self.has_local:
            return
        self._log("warning", f"[LLM] {reason} → 처음부터 로컬({self._local_name})")
        self._enter_local()

    def warm_local_async(self) -> None:
        """로컬 모델 적재를 백그라운드로 시작한다(없으면 아무것도 안 함)."""
        if self._warm_local is None:
            return
        threading.Thread(target=self._warm_local, daemon=True, name="llm-warm-local").start()

    def _maybe_return(self, trigger: str) -> None:
        with self._lock:
            if (self.state is BackendState.LOCAL and self.cloud_ready
                    and not self.run_active and not self.is_moving and not self.is_paused):
                self.state = BackendState.CLOUD
                self.cloud_ready = False
                self.probe_ok_streak = 0
                self.last_return_at = self._clock()
                self._log("info", f"[LLM] {trigger} → 클라우드 복귀")
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_llm_backend.py -q`
Expected: 모두 통과 (`34 passed`)

- [ ] **Step 5: 커밋(사용자 허락 시)**

```bash
git add src/llm_backend.py tests/test_llm_backend.py
git commit -m "feat(voice): 전환 담당 모듈 — 30초 확인·주행 끝난 뒤 복귀·왔다 갔다 반복 억제"
```

---

### Task 3: 확인·예열 도우미와 사건 JSON 파싱

**Files:**
- Modify: `src/llm_backend.py`
- Test: `tests/test_llm_backend.py`

**Interfaces:**
- Produces:
  - `def http_probe(url: str, headers: Optional[dict] = None, timeout_sec: float = 3.0) -> ProbeResult`
  - `def ollama_warm(host: str, model: str, timeout_sec: float = 180.0, logger=_stderr_logger) -> None`
  - `def parse_goal_event(data: str) -> Optional[str]` (JSON 의 `event` 키, 아니면 None)

- [ ] **Step 1: 실패 시험 작성**

`tests/test_llm_backend.py` 끝에 추가:

```python
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
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_llm_backend.py -q -k "Helpers"`
Expected: `ImportError: cannot import name 'http_probe'`

- [ ] **Step 3: 도우미 구현**

`src/llm_backend.py` 끝에 추가:

```python
# ----- 도우미(네트워크·JSON) ------------------------------------------------
def http_probe(url: str, headers: Optional[dict] = None, timeout_sec: float = 3.0) -> ProbeResult:
    """GET 한 번으로 클라우드가 닿는지 본다. 토큰을 쓰지 않는다.

    200 → ALIVE, 401/403 → AUTH_FAILED, 그 밖의 HTTP 오류·연결 불가·timeout → DEAD.
    """
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return ProbeResult.ALIVE if resp.status == 200 else ProbeResult.DEAD
    except urllib.error.HTTPError as err:
        return ProbeResult.AUTH_FAILED if err.code in (401, 403) else ProbeResult.DEAD
    except (urllib.error.URLError, OSError, TimeoutError):
        return ProbeResult.DEAD


def ollama_warm(host: str, model: str, timeout_sec: float = 180.0,
                logger: Callable[[str, str], None] = _stderr_logger) -> None:
    """로컬 Ollama 에 모델을 미리 올린다(keep_alive=-1). 실패해도 예외를 올리지 않는다.

    scripts/warmup_llm.py 와 같은 요청이다. 서버가 아직 안 떴을 수 있으니 timeout 을 길게 둔다.
    """
    import urllib.request

    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=json.dumps({"model": model, "keep_alive": -1}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=timeout_sec).read()
        logger("info", f"[LLM] 로컬 모델 예열 완료: {model}")
    except Exception as exc:  # 네트워크·서버 부재 — 노드는 계속 가야 한다
        logger("warning", f"[LLM] 로컬 모델 예열 실패(무시 가능): {exc}")


def parse_goal_event(data: str) -> Optional[str]:
    """/vica_goal_event 의 JSON 문자열에서 event 이름만 꺼낸다. 아니면 None."""
    try:
        payload = json.loads(data)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    event = payload.get("event")
    return event if isinstance(event, str) and event else None
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_llm_backend.py -q`
Expected: 모두 통과 (`46 passed`)

- [ ] **Step 5: 커밋(사용자 허락 시)**

```bash
git add src/llm_backend.py tests/test_llm_backend.py
git commit -m "feat(voice): 클라우드 확인·로컬 예열·주행 사건 파싱 도우미"
```

---

### Task 4: 파서 통합 — 관리자 싱글턴과 `parse_intent(model=None)`

**Files:**
- Modify: `src/langchain_intent_parser.py:44-51` (상수), `:297-329` (`_get_structured_llm`), `:331-337` (시그니처), `:437-457` (호출·예외)
- Test: `tests/test_parser_backend.py` (신규)

**Interfaces:**
- Consumes: Task 1~3 의 `LlmBackendManager`, `http_probe`, `ollama_warm`, `ProbeResult`
- Produces:
  - `FALLBACK_MODEL: str`, `FALLBACK_HOST: str`, `CLOUD_TIMEOUT_SEC: float`, `CLOUD_PROBE_SEC: float` (모듈 상수, env 에서)
  - `_get_structured_llm(model: str, *, timeout: float = 15, max_retries: int = 1)`
  - `get_backend_manager() -> LlmBackendManager` (지연 생성 싱글턴)
  - `reset_backend_manager() -> None` (시험용)
  - `parse_intent(..., model: Optional[str] = None)`: `model` 명시 = 그 모델 직접 호출(벤치 경로), 없으면 관리자

- [ ] **Step 1: 실패 시험 작성**

`tests/test_parser_backend.py`:

```python
"""파서가 전환 담당 모듈를 거치는지, 명시 모델은 우회하는지 (LLM 없이 검증)."""
import pytest

from src import langchain_intent_parser as parser
from src.langchain_intent_parser import _IntentDraft, parse_intent
from src.llm_backend import BackendState, LlmBackendManager, ProbeResult
from src.replies import LLM_UNAVAILABLE
from src.schema import DestinationData

DEST = DestinationData(
    id="starlight_1f_restroom",
    name="별빛관 1층 화장실",
    confirm_prompt="별빛관 1층 화장실로 안내해드릴까요?",
)
DRAFT = _IntentDraft(intent="navigate", destination_candidate="별빛관 1층 화장실")


class Boom(Exception):
    pass


def _manager(cloud_fail: bool, local: bool = True) -> LlmBackendManager:
    def cloud(messages):
        if cloud_fail:
            raise Boom("cloud down")
        return DRAFT

    return LlmBackendManager(cloud, (lambda m: DRAFT) if local else None,
                             lambda: ProbeResult.DEAD, logger=lambda *_: None)


@pytest.fixture(autouse=True)
def _reset():
    parser.reset_backend_manager()
    yield
    parser.reset_backend_manager()


def test_parse_intent_uses_manager_and_falls_back(monkeypatch):
    mgr = _manager(cloud_fail=True)
    monkeypatch.setattr(parser, "get_backend_manager", lambda: mgr)
    intent = parse_intent("화장실로 안내해줘", [DEST])
    assert intent.intent == "navigate"
    assert intent.matched_destination_id == "starlight_1f_restroom"
    assert mgr.state is BackendState.LOCAL


def test_parse_intent_unavailable_when_both_fail(monkeypatch):
    mgr = _manager(cloud_fail=True, local=False)
    monkeypatch.setattr(parser, "get_backend_manager", lambda: mgr)
    intent = parse_intent("화장실로 안내해줘", [DEST])
    assert intent.intent == "unknown"
    assert intent.reply == LLM_UNAVAILABLE


def test_explicit_model_bypasses_manager(monkeypatch):
    called = []

    class Direct:
        def invoke(self, messages):
            called.append("direct")
            return DRAFT

    monkeypatch.setattr(parser, "_get_structured_llm", lambda model, **kw: Direct())
    monkeypatch.setattr(parser, "get_backend_manager",
                        lambda: (_ for _ in ()).throw(AssertionError("관리자를 부르면 안 된다")))
    intent = parse_intent("화장실로 안내해줘", [DEST], model="gemma4-e2b-text")
    assert called == ["direct"]
    assert intent.intent == "navigate"


def test_manager_is_built_once(monkeypatch):
    built = []

    class Direct:
        def invoke(self, messages):
            return DRAFT

    monkeypatch.setattr(parser, "_get_structured_llm",
                        lambda model, **kw: built.append(model) or Direct())
    monkeypatch.setattr(parser, "FALLBACK_MODEL", "")
    a = parser.get_backend_manager()
    b = parser.get_backend_manager()
    assert a is b
    assert built == [parser.DEFAULT_MODEL]
    assert a.has_local is False  # 폴백 모델이 비면 로컬 없음 = 옛 동작


def test_manager_has_local_when_fallback_set(monkeypatch):
    class Direct:
        def invoke(self, messages):
            return DRAFT

    monkeypatch.setattr(parser, "_get_structured_llm", lambda model, **kw: Direct())
    monkeypatch.setattr(parser, "FALLBACK_MODEL", "gemma4-e2b-text")
    monkeypatch.setattr(parser, "_build_local_structured", lambda: Direct())
    mgr = parser.get_backend_manager()
    assert mgr.has_local is True
    assert mgr.state is BackendState.CLOUD
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_parser_backend.py -q`
Expected: `AttributeError: module 'src.langchain_intent_parser' has no attribute 'reset_backend_manager'`

- [ ] **Step 3: 파서 수정**

(a) `src/langchain_intent_parser.py` 상단 import 에 추가:

```python
from .llm_backend import LlmBackendManager, ProbeResult, http_probe, ollama_warm
```

(b) 상수 블록(`PROVIDER = ...` 아래, 51행 뒤)에 추가:

```python
# ----- 로컬 폴백(2026-09-19 설계) --------------------------------------------
# 비어 있으면 폴백이 꺼진다 = 예전 동작(클라우드 실패 = LLM_UNAVAILABLE).
FALLBACK_MODEL = os.environ.get("VICA_LLM_FALLBACK_MODEL", "").strip()
FALLBACK_HOST = os.environ.get("VICA_LLM_FALLBACK_HOST", "http://localhost:11434")
# 폴백이 켜졌을 때의 클라우드 대기 시간. 8월 실측 최대 1.86초의 3배. 재시도 없음.
CLOUD_TIMEOUT_SEC = float(os.environ.get("VICA_LLM_CLOUD_TIMEOUT", "6"))
CLOUD_PROBE_SEC = float(os.environ.get("VICA_LLM_CLOUD_PROBE_SEC", "30"))
```

(c) `_get_structured_llm` 시그니처와 ChatOpenAI 인자를 바꾼다:

```python
def _get_structured_llm(model: str, *, timeout: float = 15, max_retries: int = 1):
    """구조화 출력(_IntentDraft) LLM 을 만든다. 백엔드는 PROVIDER 가 정한다.

    timeout/max_retries 는 openai 경로에만 쓴다. 폴백이 켜지면 관리자가
    (CLOUD_TIMEOUT_SEC, 0) 을 넘긴다 — 로컬이 받아 주므로 오래 기다릴 이유가 없다.
    """
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            temperature=0,
            timeout=timeout,
            max_retries=max_retries,
        )
        return llm.with_structured_output(_IntentDraft, method="json_schema", strict=True)
    # (아래 ollama 분기는 그대로)
```

기존 주석("로봇 대화에서 무한 대기는 곧 침묵이다 …")은 docstring 으로 옮긴다.

(d) `_get_structured_llm` 바로 아래에 관리자 조립을 추가:

```python
def _build_local_structured():
    """로컬 Ollama 구조화 LLM. 09-19 실측: gemma4-e2b-text 8/8, 발화당 3.0초."""
    llm = ChatOllama(
        model=FALLBACK_MODEL,
        base_url=FALLBACK_HOST,
        temperature=0,
        reasoning=False,
        keep_alive=-1,
    )
    return llm.with_structured_output(_IntentDraft)


def _cloud_probe() -> ProbeResult:
    """토큰을 쓰지 않는 접속 확인. openai 는 /models, ollama 클라우드는 /api/version."""
    if PROVIDER == "openai":
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        key = os.environ.get("OPENAI_API_KEY", "")
        return http_probe(f"{base}/models", headers={"Authorization": f"Bearer {key}"})
    key = os.environ.get("OLLAMA_API_KEY", "")
    headers = {"Authorization": f"Bearer {key}"} if key else None
    return http_probe(f"{OLLAMA_HOST.rstrip('/')}/api/version", headers=headers)


def _log(level: str, msg: str) -> None:
    import sys

    print(msg, file=sys.stderr)


_MANAGER: Optional[LlmBackendManager] = None


def get_backend_manager() -> LlmBackendManager:
    """전환 담당 모듈 싱글턴. 처음 부를 때 만든다(시험이 임포트만으로 클라이언트를 만들지 않게)."""
    global _MANAGER
    if _MANAGER is not None:
        return _MANAGER
    fallback_on = bool(FALLBACK_MODEL)
    try:
        cloud_llm = _get_structured_llm(
            DEFAULT_MODEL,
            timeout=CLOUD_TIMEOUT_SEC if fallback_on else 15,
            max_retries=0 if fallback_on else 1,
        )
        cloud = cloud_llm.invoke
    except Exception as build_error:  # 키 없음 등 — 호출 때 실패로 드러나 로컬로 대피한다
        def cloud(messages, _err=build_error):
            raise _err
    local = _build_local_structured().invoke if fallback_on else None
    _MANAGER = LlmBackendManager(
        cloud, local, _cloud_probe,
        warm_local=(lambda: ollama_warm(FALLBACK_HOST, FALLBACK_MODEL, logger=_log)) if fallback_on else None,
        logger=_log,
        local_name=FALLBACK_MODEL or "-",
        probe_interval_sec=CLOUD_PROBE_SEC,
    )
    return _MANAGER


def reset_backend_manager() -> None:
    """시험용. 다음 get_backend_manager() 가 새로 만든다."""
    global _MANAGER
    _MANAGER = None
```

(e) `parse_intent` 시그니처의 `model: str = DEFAULT_MODEL` → `model: Optional[str] = None`. 본문의 호출부(437행 근처)를 다음으로 바꾼다:

```python
    messages: list[BaseMessage] = [SystemMessage(_build_system_prompt(destinations, robot_state))]
    if history:
        messages.extend(history)
    messages.append(HumanMessage(user_text))

    try:
        if model is not None:
            # 벤치·수동 지정 경로(scripts/bench_models.py): 그 모델 하나로 직접 부른다.
            draft: _IntentDraft = _get_structured_llm(model).invoke(messages)
        else:
            draft = get_backend_manager().invoke(messages)
    except Exception as exc:
        # 클라우드→로컬까지 실패했거나 폴백이 꺼진 상태의 실패.
        # 크래시 대신 안전한 fallback 응답을 돌려준다.
        # (긴급어는 LLM 이전 단계에서 처리되므로 이 실패의 영향을 받지 않는다.)
        import sys

        print(f"[LLM] 호출 실패: {exc}", file=sys.stderr)
        return VicaIntent(
            intent="unknown",
            reply=LLM_UNAVAILABLE,
            confidence=0.0,
            need_confirm=False,
        )
```

(`structured = _get_structured_llm(model)` 한 줄은 지운다.)

- [ ] **Step 4: 통과 확인 + 회귀**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 새 시험 5개 포함 전부 통과. 기존 시험은 `model` 인자를 안 쓰므로 영향 없음.

Run: `.venv/bin/python -c "import src.langchain_intent_parser as p; print(p.FALLBACK_MODEL or '(폴백 꺼짐)')"`
Expected: `.env` 에 폴백 모델이 없으니 `(폴백 꺼짐)`

- [ ] **Step 5: 커밋(사용자 허락 시)**

```bash
git add src/langchain_intent_parser.py tests/test_parser_backend.py
git commit -m "feat(parser): parse_intent 가 전환 담당 모듈를 거친다 — 명시 모델은 우회"
```

---

### Task 5: `RobotState.is_paused` 전달

**Files:**
- Modify: `src/schema.py:94-103`, `src/ros_convert.py:54-60`
- Test: `tests/test_llm_backend.py` (schema 기본값만; ros_convert 는 ROS 메시지가 필요해 Task 11 실기에서 확인)

**Interfaces:**
- Produces: `RobotState.is_paused: bool = False`; `msg_to_robot_state` 가 `is_paused=msg.is_paused` 를 채움

- [ ] **Step 1: 실패 시험 작성**

`tests/test_llm_backend.py` 끝에 추가:

```python
def test_robot_state_has_is_paused_default_false():
    from src.schema import RobotState
    assert RobotState().is_paused is False
    assert RobotState(is_paused=True).is_paused is True
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_llm_backend.py -q -k is_paused`
Expected: `AttributeError: 'RobotState' object has no attribute 'is_paused'`

- [ ] **Step 3: 구현**

`src/schema.py` 의 `RobotState`:

```python
    current_floor: Optional[int] = None
    current_building: str = ""
    is_moving: bool = False
    # 일시정지(목적지를 기억한 채 정지). 전환 담당 모듈의 복귀 조건에 쓴다 (2026-09-19).
    is_paused: bool = False
```

`src/ros_convert.py` 의 `msg_to_robot_state`:

```python
    return RobotState(
        current_floor=None if msg.current_floor < 0 else msg.current_floor,
        current_building=msg.current_building,
        is_moving=msg.is_moving,
        is_paused=msg.is_paused,
    )
```

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 전부 통과

- [ ] **Step 5: 커밋(사용자 허락 시)**

```bash
git add src/schema.py src/ros_convert.py tests/test_llm_backend.py
git commit -m "feat(voice): RobotState 에 is_paused 를 싣는다 — 전환 복귀 조건용"
```

---

### Task 6: ROS 노드 — 주행 사건 구독, 백엔드 루프, 생존 신호

**Files:**
- Modify: `src/ros_node.py` (import·`__init__`·`_on_robot_state`·`_warmup_llm`·신규 메서드)

**Interfaces:**
- Consumes: `get_backend_manager()`, `BackendState`, `parse_goal_event`
- Produces: 토픽 `/vica/llm_cloud_alive`(std_msgs/Bool) 1 Hz(CLOUD 일 때만); 구독 `/vica_goal_event`(std_msgs/String)

- [ ] **Step 1: import 추가**

`src/ros_node.py` 상단:

```python
from .langchain_intent_parser import (
    SHORTCUT_REPLIES, get_backend_manager, is_instant_utterance, parse_intent)
from .llm_backend import BackendState, parse_goal_event
```

- [ ] **Step 2: `__init__` 에 발행·구독·루프 추가**

`self.create_subscription(String, "/vica/wake", self._on_wake_signal, 10)` 다음 줄에:

```python
        # ----- 클라우드→로컬 자동 전환 (2026-09-19 설계) ------------------------
        # 관리자는 파서와 같은 싱글턴이다. 주행 사건·로봇 상태를 흘려 넣고,
        # 별도 스레드가 1초마다 tick(클라우드 확인·복귀)과 생존 신호를 맡는다.
        # 스레드로 두는 이유: LLM 호출(3~6초) 동안에도 생존 신호가 끊기면 안 된다.
        self._backend = get_backend_manager()
        self._cloud_alive_pub = self.create_publisher(Bool, "/vica/llm_cloud_alive", 10)
        self.create_subscription(String, "/vica_goal_event", self._on_goal_event, 10)
        threading.Thread(target=self._backend_loop, daemon=True, name="llm-backend").start()
```

- [ ] **Step 3: 콜백·루프 메서드 추가**

`_on_robot_state` 를 바꾸고, 그 아래 두 메서드를 추가:

```python
    def _on_robot_state(self, msg: RobotStateMsg) -> None:
        """로봇 상태 메시지를 받아 최신값으로 보관한다. 전환 담당 모듈에도 넘긴다."""
        self._robot_state = msg_to_robot_state(msg)
        self._backend.on_robot_state(msg.is_moving, msg.is_paused)

    def _on_goal_event(self, msg: String) -> None:
        """미션 매니저의 주행 사건 → 전환 담당 모듈(주행 중엔 클라우드로 안 돌아간다)."""
        event = parse_goal_event(msg.data)
        if event:
            self._backend.on_goal_event(event)

    def _backend_loop(self) -> None:
        """1 Hz: 클라우드 확인·복귀 판정, CLOUD 상태면 생존 신호 발행."""
        alive = Bool(data=True)
        while rclpy.ok():
            try:
                before = self._backend.state
                after = self._backend.tick()
                if before is not after:
                    self.get_logger().info(f"[LLM] 백엔드 {before.value} → {after.value}")
                if self._backend.heartbeat_enabled:
                    self._cloud_alive_pub.publish(alive)
            except Exception as exc:  # 루프는 죽지 않는다
                self.get_logger().warning(f"[LLM] 백엔드 루프 오류(무시): {exc}")
            time.sleep(1.0)
```

- [ ] **Step 4: 워밍업 실패 시 로컬로 시작**

`_warmup_llm` 을 다음으로 바꾼다:

```python
    def _warmup_llm(self) -> None:
        started = time.monotonic()
        try:
            parse_intent("워밍업", [], history=[])
            self.get_logger().info(
                f"LLM 워밍업 완료 ({time.monotonic() - started:.1f}초)")
        except Exception as exc:
            self.get_logger().warning(f"LLM 워밍업 실패(무시 가능): {exc}")
        # 워밍업 중 클라우드가 실패했으면 관리자는 이미 LOCAL 이다. 그 경우 로컬
        # 모델을 지금 미리 올려 첫 발화가 적재 7초를 기다리지 않게 한다.
        if self._backend.state is BackendState.LOCAL:
            self._backend.warm_local_async()
```

(`parse_intent` 는 실패해도 `LLM_UNAVAILABLE` 을 돌려주므로 except 는 거의 안 탄다. 상태로 판단하는 이유다.)

- [ ] **Step 5: 정적 확인**

Run: `.venv/bin/python -m pyflakes src/ros_node.py 2>/dev/null || .venv/bin/python -m py_compile src/ros_node.py && echo compiled`
Expected: 오류 없음. (실행 확인은 Task 11 실기: `ros2 topic hz /vica/llm_cloud_alive` ≈ 1.0)

- [ ] **Step 6: 커밋(사용자 허락 시)**

```bash
git add src/ros_node.py
git commit -m "feat(voice): LLM 노드가 주행 사건을 듣고 생존 신호를 낸다 — 전환 담당 모듈 연결"
```

---

### Task 7: CLI(`main.py`) 에서 tick

**Files:**
- Modify: `src/main.py:93-94`

- [ ] **Step 1: 구현**

`src/main.py` 의 `parse_intent(...)` 호출 바로 위에:

```python
        # 2) 일반 발화는 LLM intent 파서로 해석한다. CLI 엔 타이머가 없어 여기서 tick.
        get_backend_manager().tick()
        intent = parse_intent(text, destinations, history=history, robot_state=robot_state)
```

import 줄(`from .langchain_intent_parser import parse_intent`)을:

```python
from .langchain_intent_parser import get_backend_manager, parse_intent
```

- [ ] **Step 2: 확인**

Run: `.venv/bin/python -m py_compile src/main.py && echo compiled`
Expected: `compiled`

- [ ] **Step 3: 커밋(사용자 허락 시)**

```bash
git add src/main.py
git commit -m "feat(voice): CLI 도 발화마다 전환 담당 모듈 tick"
```

---

### Task 8: launch 에 `ollama serve`

**Files:**
- Modify: `launch/vica_voice.launch.py` (LaunchDescription 목록 끝)

- [ ] **Step 1: 구현**

`_python_node("src.ros_wakeword_node", ...)` 항목 뒤에 추가:

```python
            # 로컬 LLM 폴백용 Ollama 서버 (2026-09-19 설계 §4.4). 다른 노드처럼 같이
            # 뜨고 같이 꺼진다. 포트 11434 를 이미 누가 쓰면 바인드 실패로 끝나고
            # 그 서버를 쓴다 — launch 는 이 종료를 치명으로 보지 않는다.
            # 모델은 미리 올리지 않는다(필요할 때 적재). 올라온 뒤엔 내리지 않는다.
            ExecuteProcess(
                cmd=["ollama", "serve"],
                name="ollama_serve",
                output="screen",
                additional_env={
                    "OLLAMA_KEEP_ALIVE": "-1",
                    "OLLAMA_NUM_PARALLEL": "1",
                    "OLLAMA_MAX_LOADED_MODELS": "1",
                },
            ),
```

- [ ] **Step 2: 확인(정적)**

Run: `.venv/bin/python -m py_compile launch/vica_voice.launch.py && echo compiled`
Expected: `compiled`. 실행 확인은 Task 11(`curl -s localhost:11434/api/version`).

- [ ] **Step 3: 커밋(사용자 허락 시)**

```bash
git add launch/vica_voice.launch.py
git commit -m "feat(launch): 음성 스택이 ollama serve 를 함께 띄운다"
```

---

### Task 9: 모델 조립 설명서·스크립트·설정·문서

**Files:**
- Create: `ollama/Modelfile.gemma4-e2b-text`, `scripts/setup_local_llm.sh`
- Modify: `.env.example`, `docs/jetson-setup.md:50-58`

- [ ] **Step 1: Modelfile**

`ollama/Modelfile.gemma4-e2b-text`:

```text
# Gemma 4 E2B 텍스트 전용 변형 (2026-09-19 실측 근거: 8/8 정답·3.0초·RAM 0.46 GB)
#
# - FROM 은 HF unsloth/gemma-4-E2B-it-GGUF:Q4_K_M 의 가중치 blob 경로다. projector
#   (이미지·음성 인코더 0.9 GB)는 넣지 않는다 — 글자만 쓴다.
# - 모델 턴 앞의 빈 생각 블록이 자발적 추론을 막는다. HF 임포트는 thinking
#   capability 가 없어 think=false 가 무시되고, JSON 을 요구하면 한국어 추론
#   ~100토큰을 숨어서 써 3배 느려졌다. 공식 태그의 think=false 와 같은 효과다.
# - 경로는 scripts/setup_local_llm.sh 가 채운다. 손으로 만들 땐 아래 줄을 바꾼다.
FROM ./gemma-4-E2B-it-Q4_K_M.gguf
TEMPLATE """{{- range .Messages }}<|turn>{{ if eq .Role "assistant" }}model{{ else }}{{ .Role }}{{ end }}
{{ .Content }}<turn|>
{{ end }}<|turn>model
<|channel>thought
<channel|>"""
PARAMETER stop "<turn|>"
PARAMETER stop "<|turn>"
PARAMETER temperature 0
```

- [ ] **Step 2: 조립 스크립트**

`scripts/setup_local_llm.sh`:

```bash
#!/usr/bin/env bash
# 로컬 폴백 LLM 준비: HF Gemma 4 E2B Q4_K_M 을 받아 텍스트 전용 모델 gemma4-e2b-text 로 조립한다.
#
# 사용 (저장소 루트, ollama 서버가 떠 있어야 한다):
#     scripts/setup_local_llm.sh
# 재실행해도 안전하다(같은 이름으로 다시 만든다). 새 젯슨에서는 이 한 줄이면 된다.
set -euo pipefail

SRC_TAG="hf.co/unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"
SRC_MANIFEST="hf.co/unsloth/gemma-4-E2B-it-GGUF/Q4_K_M"
NAME="gemma4-e2b-text"
OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

if ! curl -sf localhost:11434/api/version >/dev/null; then
    echo "ollama 서버가 없습니다. 먼저 'ollama serve' 를 띄우세요 (음성 launch 가 띄워 줍니다)." >&2
    exit 1
fi

echo "[1/3] 가중치 받기: $SRC_TAG (3.1 GB + projector 0.9 GB)"
ollama pull "$SRC_TAG"

echo "[2/3] projector 를 뺀 가중치 blob 찾기"
BLOB=$(python3 - "$OLLAMA_MODELS/manifests/$SRC_MANIFEST" "$OLLAMA_MODELS" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
digest = next(l["digest"] for l in manifest["layers"] if l["mediaType"].endswith(".model"))
print(f"{sys.argv[2]}/blobs/{digest.replace(':', '-')}")
PY
)
test -f "$BLOB" || { echo "blob 이 없습니다: $BLOB" >&2; exit 1; }

echo "[3/3] 모델 조립: $NAME"
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
sed "s|^FROM .*|FROM $BLOB|" "$HERE/ollama/Modelfile.gemma4-e2b-text" > "$TMP"
ollama create "$NAME" -f "$TMP"
ollama show "$NAME" | sed -n '1,12p'
echo "완료. .env 에 VICA_LLM_FALLBACK_MODEL=$NAME 을 넣으면 폴백이 켜진다."
```

Run: `chmod +x scripts/setup_local_llm.sh && bash -n scripts/setup_local_llm.sh && echo syntax-ok`
Expected: `syntax-ok`

- [ ] **Step 3: `.env.example`**

"(선택) OpenAI 백엔드" 블록 뒤에 추가:

```bash
# ===== (선택) 로컬 폴백 — 클라우드가 닿지 않으면 젯슨 안의 Ollama 모델로 자동 전환 =====
# 비우면 폴백이 꺼진다(클라우드 실패 = 고정 멘트). 모델은 scripts/setup_local_llm.sh 로 만든다.
# VICA_LLM_FALLBACK_MODEL=gemma4-e2b-text
# VICA_LLM_FALLBACK_HOST=http://localhost:11434
# VICA_LLM_CLOUD_TIMEOUT=6          # 초. 폴백이 켜졌을 때만 적용, 재시도 없음
# VICA_LLM_CLOUD_PROBE_SEC=30       # 로컬 상태에서 클라우드 확인 간격(초)
```

- [ ] **Step 4: `docs/jetson-setup.md` 4절 교체**

"## 4. (선택) 오프라인/폴백용 로컬 LLM (Ollama + gemma4 e2b)" 절 전체를 다음으로 바꾼다:

````markdown
## 4. (선택) 로컬 폴백 LLM (Ollama + gemma4-e2b-text)

클라우드가 닿지 않을 때 자동으로 넘어가는 로컬 모델이다. 설계·규칙은
`docs/superpowers/specs/2026-09-19-llm-local-fallback-design.md`.

```bash
curl -fsSL https://ollama.com/install.sh | sh   # ARM64 지원, CUDA(jetpack6) 자동 감지
ollama serve &                                  # 평소엔 음성 launch 가 띄운다
scripts/setup_local_llm.sh                      # HF 가중치 받기 → 텍스트 전용 모델 조립
```

빠른 확인: `ollama run gemma4-e2b-text "안녕"` → 한두 문장이 3초 안에 나오면 정상.
공식 `gemma4:e2b` 태그(6.7 GB)를 쓰지 않는 이유: 같은 속도인데 RAM 을 6배 쓴다
(3.05 GB vs 0.46 GB, 2026-09-19 실측). 켜는 법은 5절의 `VICA_LLM_FALLBACK_MODEL`.
````

5절의 "오프라인/폴백으로 로컬 Ollama 를 쓰려면 …" 단락은 다음으로 바꾼다:

````markdown
로컬 폴백을 켜려면(4번 설치 후) 위 설정은 그대로 두고 한 줄을 더한다:
```bash
VICA_LLM_FALLBACK_MODEL=gemma4-e2b-text
```
→ 코드 수정 없이 `.env` 만으로 켜고 끈다.
````

- [ ] **Step 5: 확인**

Run: `grep -n "FALLBACK" .env.example docs/jetson-setup.md | wc -l`
Expected: 5 이상

- [ ] **Step 6: 커밋(사용자 허락 시)**

```bash
git add ollama/Modelfile.gemma4-e2b-text scripts/setup_local_llm.sh .env.example docs/jetson-setup.md
git commit -m "docs(voice): 로컬 폴백 모델 조립 스크립트·Modelfile·설정 안내"
```

---

### Task 10: ROS 진단 항목 (`vica_ros2_ws`, worktree)

**Files:**
- Modify: `src/vica_system_monitor/config/probes.yaml` (`topic_probe_names` 목록 + 정의 블록)
- Modify: `src/vica_system_monitor/vica_system_monitor/fault_catalog.py` (`CATALOG` 의 voice 구역, `VOICE_NODE_SILENT` 항목 뒤)
- Test: 기존 `test/test_config_contract.py`, `test/test_probe_config.py`, `test/test_fault_catalog.py`

**Interfaces:**
- Consumes: Task 6 의 토픽 `/vica/llm_cloud_alive`
- Produces: 결함 코드 `LLM_CLOUD_OFFLINE`

- [ ] **Step 1: worktree 생성**

```bash
git -C /home/ji_w/VICA-smarthandle/vica_ros2_ws fetch -q origin dev
git -C /home/ji_w/VICA-smarthandle/vica_ros2_ws worktree add /home/ji_w/wt-llmprobe -b feat/llm-cloud-probe origin/dev
cd /home/ji_w/wt-llmprobe/src/vica_system_monitor && git branch --show-current
```

Expected: `feat/llm-cloud-probe`

- [ ] **Step 2: 실패 시험 실행(기준선)**

Run: `cd /home/ji_w/wt-llmprobe/src/vica_system_monitor && PYTHONPATH=$PWD python3 -m pytest test/test_config_contract.py test/test_probe_config.py test/test_fault_catalog.py -q`
Expected: 전부 통과(기준선). 아직 새 항목이 없으니 실패할 것은 없다 — 이 단계는 실행 환경 확인이다.

- [ ] **Step 3: probes.yaml**

`topic_probe_names:` 목록 끝(`- app_bridge` 다음)에 `- voice_llm_cloud` 를 추가하고, `app_bridge:` 정의 블록 뒤에:

```yaml
    # ===== 음성 LLM 클라우드 생존 신호 (2026-09-19 로컬 폴백 설계 §4.6) =====
    #
    # 음성 LLM 노드는 **클라우드 백엔드로 정상 동작 중일 때만** 이 토픽을 1 Hz 로
    # 낸다(값 True 는 의미 없음). 클라우드가 닿지 않아 젯슨 안의 모델로 대피하면
    # 발행을 멈춘다. 이 프로브가 값이 아니라 **주기만** 보기 때문에, 멈춤이 곧
    # "클라우드 LLM 오프라인" 경고가 된다. 앱 코드 변경 없이 진단 화면에 뜬다.
    #
    # 등급 WARN·optional — 로봇 동작과 무관하다. 안내는 로컬 모델로 계속되며 답이
    # 2~3초 느려질 뿐이다. 주행이 끝나면 노드가 알아서 클라우드로 돌아가고 신호가
    # 다시 온다.
    voice_llm_cloud:
      component: voice
      topic: /vica/llm_cloud_alive
      msg_type: std_msgs/msg/Bool
      qos: default
      min_hz: 0.5
      max_hz: 2.0
      fault_code: LLM_CLOUD_OFFLINE
      optional: true
```

- [ ] **Step 4: fault_catalog.py**

`'VOICE_NODE_SILENT': FaultSpec(...)` 항목 바로 뒤에:

```python
    # [2026-09-19] 음성 LLM 이 클라우드에 닿지 않아 로컬 모델로 대피한 상태.
    # 로봇은 정상 안내 중이다 — 관리자가 로봇을 세우러 갈 일이 아니다. 주행이
    # 끝나면 노드가 알아서 클라우드로 돌아간다(설계: vica-voice-llm
    # docs/superpowers/specs/2026-09-19-llm-local-fallback-design.md).
    'LLM_CLOUD_OFFLINE': FaultSpec(
        'voice',
        SEVERITY_WARN,
        'LLM이 클라우드에 닿지 않아 로봇 안의 모델로 안내 중입니다. 답이 2~3초 느려질 수 있습니다.',
        '인터넷 연결을 확인해 주세요. 주행이 끝나면 자동으로 클라우드로 돌아갑니다.',
    ),
```

- [ ] **Step 5: 시험 통과 확인**

Run: `cd /home/ji_w/wt-llmprobe/src/vica_system_monitor && PYTHONPATH=$PWD python3 -m pytest test/test_config_contract.py test/test_probe_config.py test/test_fault_catalog.py -q`
Expected: 전부 통과 (목록·정의 블록 짝, fault_code 가 카탈로그에 있음, component 유효)

Run: `PYTHONPATH=$PWD python3 -c "from vica_system_monitor.probe_config import validate_fault_code as v; print(v('LLM_CLOUD_OFFLINE'))"`
Expected: `None`

- [ ] **Step 6: 커밋(사용자 허락 시, worktree 에서)**

```bash
cd /home/ji_w/wt-llmprobe
git add src/vica_system_monitor/config/probes.yaml src/vica_system_monitor/vica_system_monitor/fault_catalog.py
git commit -m "feat(monitor): 음성 LLM 클라우드 생존 신호 프로브 — LLM_CLOUD_OFFLINE 경고"
```

---

### Task 11: 젯슨 실기 검증 (사용자 승인 뒤)

**Files:** 없음(측정 기록은 `devlog/2026-09-XX-llm-로컬-폴백-실기.md` 로 남긴다 — 저장소 루트)

**전제:** 주행 스택 전체가 떠 있고, `.env` 에 `VICA_LLM_FALLBACK_MODEL=gemma4-e2b-text` 가 있고, `scripts/setup_local_llm.sh` 가 끝났고, 음성 스택을 재기동했다(`.env` 는 시작 때만 읽는다). 네트워크를 끊는 방법은 사용자와 합의한다(랜선 뽑기 또는 와이파이 끄기).

- [ ] **Step 1: 평소 상태 확인**

```bash
source /opt/ros/humble/setup.bash
ros2 topic hz /vica/llm_cloud_alive          # ≈ 1.0 Hz
curl -s localhost:11434/api/version           # {"version":"0.30.6"}
ollama ps                                     # 비어 있음(필요할 때 적재)
```

앱 진단 화면: voice 영역 경고 없음. 발화 1회 → 클라우드 답(로그에 대피 없음).

- [ ] **Step 2: 대화 중 네트워크 끊기**

발화 → 답 확인 → 네트워크 끊기 → 발화 "화장실로 안내해줘". 기록:

| 항목 | 합격선 | 측정 |
| --- | --- | --- |
| 끊긴 뒤 첫 답까지 | ≤ 12초 | |
| 그다음 발화 답 | ≤ 4초 | |
| 로그 | `[LLM] 클라우드 실패(연결 오류: …) → 로컬(gemma4-e2b-text)로 대피. 같은 발화 재처리` | |
| `ros2 topic hz /vica/llm_cloud_alive` | 신호 없음 | |
| 앱 진단 | `LLM_CLOUD_OFFLINE` 노란 경고 | |
| `free -m` 가용 / `ollama ps` | 기록 | |

- [ ] **Step 3: 주행 중 복구**

로컬 상태에서 목적지 주행 시작 → 주행 중 네트워크 복구 → 30초 뒤 로그에 `[LLM] 클라우드 살아남(확인 1회째). 주행 끝나면 복귀` → 도착 전 발화 1회는 여전히 로컬(대피 로그 없음, 답 ≤ 4초).

- [ ] **Step 4: 도착 뒤 복귀**

도착(`goal_succeeded`) → 로그 `[LLM] goal_succeeded → 클라우드 복귀` → 다음 발화 클라우드(답 ≈ 1초) → 생존 신호 재개 → 앱 경고 60초 안 해제.

- [ ] **Step 5: 네트워크 없이 노드 시작**

네트워크 끊은 채 음성 스택 재기동 → 로그 `[LLM] … 처음부터 로컬(gemma4-e2b-text)` 또는 워밍업 대피 로그 → `ollama ps` 에 모델 적재됨 → 첫 발화 ≤ 12초.

- [ ] **Step 6: 기록과 정리**

측정표를 devlog 에 남기고, 스택 없이 잰 3.0초·0.46 GB 대비 증가폭을 적는다. 통과하면 사용자에게 dev 머지 시점을 묻는다(음성·ROS 두 저장소 같은 날).

---

## Self-Review

**Spec coverage**

| 스펙 절 | Task |
| --- | --- |
| §4.1 관리자 규칙 전부 | 1, 2 |
| §4.1 접속 확인 함수·예열 | 3, 4(`_cloud_probe`·`ollama_warm`) |
| §4.2 파서 통합·timeout 6/0 | 4 |
| §4.3 노드: 사건 구독·robot_state·1 Hz·생존 신호·워밍업 실패 시 예열, `is_paused` | 5, 6 |
| §4.3 CLI tick | 7 |
| §4.4 launch·Modelfile·스크립트·문서 | 8, 9 |
| §4.5 설정 4줄 | 4(상수), 9(.env.example) |
| §4.6 진단 프로브·문구 | 10 |
| §5 오류 처리 표 | 1(둘 다 실패·폴백 꺼짐), 2(인증), 4(키 없음), 6(루프 생존) |
| §6.1 단위 시험 11개 | 1, 2, 3 |
| §6.2 ROS 시험 | 10 |
| §6.3 실기 | 11 |
| §7 되돌리기 | 4(`FALLBACK_MODEL` 빈 값 = 옛 동작, 시험 `test_no_local_means_old_behaviour`·`test_manager_is_built_once`) |

**Placeholder scan:** TBD/TODO 없음. 모든 코드 단계에 실제 코드가 있다.

**Type consistency:** `LlmBackendManager(cloud, local, probe, *, warm_local, clock, logger, local_name, probe_interval_sec, ...)` 시그니처는 Task 1 정의·Task 2/3 시험·Task 4 조립에서 동일. `logger(level: str, msg: str)` 규약 동일. `ProbeResult.ALIVE/DEAD/AUTH_FAILED` 동일. `parse_goal_event(str) -> Optional[str]` 은 Task 3 정의·Task 6 사용 동일. `_get_structured_llm(model, *, timeout, max_retries)` 는 Task 4 정의와 시험의 `lambda model, **kw` 가 맞는다.

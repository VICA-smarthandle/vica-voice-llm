"""클라우드→로컬 LLM 자동 전환(폴백) 담당 모듈. ROS·LangChain 을 모르는 순수 로직."""
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
    ALIVE = "alive"
    DEAD = "dead"
    AUTH_FAILED = "auth"


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
    """예외를 네 갈래로 나눈다. openai 패키지를 import 하지 않고 모양으로 판별한다."""
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
        self.probe_attempts = 0
        self._last_probe_result: Optional[ProbeResult] = None

    @property
    def has_local(self) -> bool:
        return self._local is not None

    def set_logger(self, logger: Callable[[str, str], None]) -> None:
        """로그 출력 함수를 바꾼다(ROS 노드가 자기 로거로 갈아 끼울 때). level 은 "info"/"warning"/"error"."""
        def safe(level: str, msg: str) -> None:
            try:
                logger(level, msg)
            except Exception as exc:
                import sys
                print(f"[{level.upper()}] {msg} (로거 실패: {exc})", file=sys.stderr)

        with self._lock:
            self._log = safe

    def invoke(self, messages: Sequence[Any]) -> Any:
        """현재 상태의 백엔드로 호출한다. 클라우드 실패는 로컬 재호출로 받는다."""
        with self._lock:
            use_cloud = self.state is BackendState.CLOUD
        if use_cloud:
            try:
                return self._cloud(messages)
            except Exception as exc:
                if not self.has_local:
                    raise
                self._switch_to_local(exc)
        if self._local is None:
            raise RuntimeError("LOCAL 상태인데 로컬 백엔드가 없다")
        return self._local(messages)

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
            self.probe_attempts = 0
            self._last_probe_result = None
            self.next_probe_at = now + self.probe_interval

    def tick(self, now: Optional[float] = None) -> BackendState:
        """1초쯤마다 부른다. LOCAL 이면 간격에 맞춰 클라우드를 확인하고 복귀를 시도한다."""
        with self._lock:
            now = self._clock() if now is None else now
            due = self.state is BackendState.LOCAL and now >= self.next_probe_at
        if due:
            result = self._probe()
            with self._lock:
                self.probe_attempts += 1
                self.next_probe_at = now + self.probe_interval
                if result is ProbeResult.ALIVE:
                    self.probe_ok_streak += 1
                    if not self.cloud_ready:
                        self.cloud_ready = True
                        self._log("info", f"[LLM] 클라우드 살아남(확인 {self.probe_attempts}회째). "
                                          "주행 끝나면 복귀")
                elif result is ProbeResult.AUTH_FAILED:
                    self.cloud_ready = False
                    self.probe_ok_streak = 0
                    if result != self._last_probe_result:
                        self._log("error", "[LLM] 클라우드는 닿지만 인증 실패 — 키를 확인하세요. 복귀 보류")
                else:
                    self.cloud_ready = False
                    self.probe_ok_streak = 0
                self._last_probe_result = result
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


def http_probe(url: str, headers: Optional[dict] = None, timeout_sec: float = 3.0) -> ProbeResult:
    """GET 한 번으로 클라우드가 닿는지 본다. 토큰을 쓰지 않는다."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return ProbeResult.ALIVE if resp.status == 200 else ProbeResult.DEAD
    except urllib.error.HTTPError as err:
        try:
            return ProbeResult.AUTH_FAILED if err.code in (401, 403) else ProbeResult.DEAD
        finally:
            err.close()
    except (urllib.error.URLError, OSError, TimeoutError):
        return ProbeResult.DEAD


def _is_connection_failure(exc: BaseException) -> bool:
    """재시도할 가치가 있는 실패인가. 서버가 응답은 했지만 실패한 경우(HTTPError 등)는"""
    import urllib.error

    if isinstance(exc, urllib.error.HTTPError):
        return False
    return isinstance(exc, (urllib.error.URLError, ConnectionError, OSError, TimeoutError))


def ollama_warm(host: str, model: str, timeout_sec: float = 180.0,
                retry_sec: float = 3.0, max_attempts: int = 20,
                logger: Callable[[str, str], None] = _stderr_logger) -> None:
    """로컬 Ollama 에 모델을 미리 올린다(keep_alive=-1). 실패해도 예외를 올리지 않는다."""
    import urllib.request

    req_body = json.dumps({"model": model, "keep_alive": -1}).encode()

    for attempt in range(1, max_attempts + 1):
        req = urllib.request.Request(
            f"{host.rstrip('/')}/api/generate",
            data=req_body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                resp.read()
        except Exception as exc:
            if _is_connection_failure(exc) and attempt < max_attempts:
                time.sleep(retry_sec)
                continue
            logger("warning", f"[LLM] 로컬 모델 예열 실패(무시 가능): {exc}")
            return
        else:
            if attempt == 1:
                logger("info", f"[LLM] 로컬 모델 예열 완료: {model}")
            else:
                logger("info", f"[LLM] 로컬 모델 예열 완료: {model} (재시도 {attempt - 1}회)")
            return


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

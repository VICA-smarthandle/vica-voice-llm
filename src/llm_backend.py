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
        self.probe_attempts = 0
        self._last_probe_result: Optional[ProbeResult] = None

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
        if self._local is None:
            raise RuntimeError("LOCAL 상태인데 로컬 백엔드가 없다")
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
            self.probe_attempts = 0
            self._last_probe_result = None
            self.next_probe_at = now + self.probe_interval

    # ----- 주기 확인·복귀 ---------------------------------------------
    def tick(self, now: Optional[float] = None) -> BackendState:
        """1초쯤마다 부른다. LOCAL 이면 간격에 맞춰 클라우드를 확인하고 복귀를 시도한다."""
        with self._lock:
            now = self._clock() if now is None else now
            due = self.state is BackendState.LOCAL and now >= self.next_probe_at
        if due:
            result = self._probe()          # 잠금 밖에서 부른다 — 최대 3초 걸린다
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
                    # 30초마다 같은 401 을 반복 보고하면 진단 로그가 도배된다 —
                    # 결과가 바뀔 때만(첫 발생·복구 뒤 재발) 오류를 남긴다.
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
        try:
            return ProbeResult.AUTH_FAILED if err.code in (401, 403) else ProbeResult.DEAD
        finally:
            err.close()
    except (urllib.error.URLError, OSError, TimeoutError):
        return ProbeResult.DEAD


def _is_connection_failure(exc: BaseException) -> bool:
    """재시도할 가치가 있는 실패인가. 서버가 응답은 했지만 실패한 경우(HTTPError 등)는
    재시도해도 똑같으므로 제외한다."""
    import urllib.error

    if isinstance(exc, urllib.error.HTTPError):
        return False
    return isinstance(exc, (urllib.error.URLError, ConnectionError, OSError, TimeoutError))


def ollama_warm(host: str, model: str, timeout_sec: float = 180.0,
                retry_sec: float = 3.0, max_attempts: int = 20,
                logger: Callable[[str, str], None] = _stderr_logger) -> None:
    """로컬 Ollama 에 모델을 미리 올린다(keep_alive=-1). 실패해도 예외를 올리지 않는다.

    scripts/warmup_llm.py 와 같은 요청이다. launch 는 `ollama serve` 와 이 예열을
    동시에 띄우므로, 서버가 포트를 열기 전이면 연결이 1초 안에 거부된다 —
    timeout 을 길게 둬도 그 실패는 못 막는다(연결 거부는 즉시 실패이지 느린 실패가
    아니다). 대신 연결류 실패(URLError·ConnectionError·OSError·TimeoutError,
    HTTPError 는 제외)만 retry_sec 간격으로 최대 max_attempts 번 재시도해 서버가
    뜨는 기동 경쟁을 넘긴다. 그 밖의 실패(모델 없음 등 HTTPError)는 재시도해도
    똑같으므로 바로 멈춘다.
    """
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
        except Exception as exc:  # 네트워크·서버 부재 — 노드는 계속 가야 한다
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

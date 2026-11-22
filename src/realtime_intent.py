"""OpenAI Realtime 로 발화 **소리**에서 바로 의도(_IntentDraft 인자)를 받는다.

설계 근거: docs/realtime-api-vica-적용-가이드.md §4 방식 2. 2026-09-20 실측으로
확인한 방식 — out-of-band 응답(conversation: "none") 에 대화 이력(글자)과 소리
항목을 넣고 set_intent 함수 호출을 강제한다. 세션 대화 상태를 쓰지 않으므로
우리 ConversationHistory 가 유일한 맥락이고, 세션이 길어져도 쌓이는 것이 없다.

안전 경계: 이 모듈은 의도 초안(dict)만 돌려준다. 발행은 파서(deliver_audio_draft:
id 매핑·계약만)와 LLM 노드가 한다. 긴급어는 상류(웨이크워드 노드 whisper)가 잡는다.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from . import answer_field, fixed_answers, intent_recut
from .tool_calls import RESERVE, audio_tools_on, reserve_tool

SAMPLE_RATE_IN = 16000   # 웨이크워드 노드 클립
SAMPLE_RATE_RT = 24000   # Realtime 입력 형식
DEFAULT_MODEL = "gpt-realtime-2.1-mini"
TOOL_NAME = "set_intent"
_INTENTS = ["navigate", "question", "clarify", "unknown", "cancel", "pause", "resume",
            "affirm", "deny", "wait", "finish", "help"]


def float32_to_pcm16(audio) -> bytes:
    """오디오 배열 → int16 LE bytes. 정수형(int16 등)은 값 그대로, 실수형은 [-1, 1] 로 보고 스케일한다.

    웨이크워드 모니터 클립은 int16(±32768) 이고 supertonic 합성음은 float32(±1.0) 이다 —
    둘을 같은 함수로 받는다. 2026-09-20 실기에서 int16 을 ±1 로 잘라 사각파 잡음을
    보냈던 사고의 수리.
    """
    arr = np.asarray(audio)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if np.issubdtype(arr.dtype, np.integer):
        if arr.dtype == np.int16:
            return arr.astype("<i2").tobytes()
        return np.clip(arr, -32768, 32767).astype("<i2").tobytes()
    farr = np.clip(arr.astype(np.float32), -1.0, 1.0)
    return (farr * 32767.0).astype("<i2").tobytes()


def resample_pcm16(pcm16: bytes, src_rate: int = SAMPLE_RATE_IN, dst_rate: int = SAMPLE_RATE_RT) -> bytes:
    """선형 보간 리샘플. 음성 명령(대역 4 kHz 이하)엔 충분하고 의존성이 없다."""
    if src_rate == dst_rate:
        return pcm16
    src = np.frombuffer(pcm16, dtype="<i2").astype(np.float32)
    if src.size == 0:
        return b""
    n_dst = int(round(src.size * dst_rate / src_rate))
    x_src = np.arange(src.size, dtype=np.float64)
    x_dst = np.linspace(0.0, src.size - 1, n_dst)
    return np.interp(x_dst, x_src, src).astype("<i2").tobytes()


# 도구가 둘일 때(VICA_TOOLS_AUDIO=on)의 set_intent 설명 — "반드시 이 함수로만 답한다"는 맞지 않는다.
_INTENT_DESCRIPTION_WITH_RESERVE = "사용자 발화(소리)의 의도를 정한다. 시설 예약이 아닌 모든 말에 호출한다."
# 정해진 대답(VICA_TALK)이 켜졌을 때 reply 칸 설명(원칙 25·10, Task N1) — 2026-11-22 실기: question 에 reply 를 비워 로봇이 침묵했다.
_REPLY_DESCRIPTION_TALK = ("사용자에게 할 짧은 말. question 은 반드시 한 문장을 적는다. "
                           "할 말이 없는 의도(affirm·deny·repeat·not_for_robot 등)는 빈 문자열")


def _facility_names() -> list[str]:
    """예약할 수 있는 시설 이름. 시설 파일이 깨졌으면 빈 목록(예약 도구를 주지 않는다)."""
    from .reservation_store import default_facilities   # 지연 import — 꺼져 있으면 읽지 않는다
    try:
        return [f.name for f in default_facilities()]
    except Exception:
        return []


def session_tools() -> tuple[list[dict], Any]:
    """(세션 tools, tool_choice). 기본은 set_intent 하나를 강제한다(2026-09-20).

    VICA_TOOLS_AUDIO=on 이고 예약할 시설이 있으면 set_intent · reserve_facility 둘을 주고 하나를 반드시 부르게 한다
    (설계 2026-11-18 5.1). 11-18 측정(글자 입력) 32개 중 31개. 시설 목록이 비면 꺼진 것과 같다 — 빈 enum 은 보내지 않는다.
    """
    names = _facility_names() if audio_tools_on() else []
    if not names:
        return [build_intent_tool()], {"type": "function", "name": TOOL_NAME}
    reserve = {"type": "function", **reserve_tool(names, with_heard_text=True)["function"]}   # Realtime 은 평평한 꼴
    return [build_intent_tool(_INTENT_DESCRIPTION_WITH_RESERVE), reserve], "required"


def audio_tools_active() -> bool:
    """소리 경로가 지금 도구 둘(set_intent · reserve_facility)을 쓰는가."""
    return len(session_tools()[0]) > 1


def build_intent_tool(description: str = "사용자 발화(소리)의 의도를 정한다. 반드시 이 함수로만 답한다.") -> dict:
    """_IntentDraft 와 같은 필드 + heard_text. 파서가 인자를 _IntentDraft 로 검증한다.

    대답 칸이 켜졌으면(VICA_ANSWER=on) answer 인자를 더한다(설계 2026-11-18 4.1) — 허용 밖의 값은 파서 쪽
    answer_field.fill 이 unclear 로 바꾼다. 꺼져 있으면 지금 인자 그대로다.
    정해진 대답이 켜졌으면(fixed_answers.on()) reply 를 꼭 채우는 칸으로 하고 설명을 바꾼다(Task N1). 꺼져 있으면 그대로다.
    """
    tool = {
        "type": "function",
        "name": TOOL_NAME,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": _INTENTS},
                "heard_text": {"type": "string", "description": "들린 말을 한국어로 그대로 적는다. 말이 아니면 빈 문자열"},
                "destination_candidate": {"type": ["string", "null"], "description": "목적지 표현(목록의 이름 그대로)"},
                "is_confirmation": {"type": ["boolean", "null"], "description": "직전 확인 질문에 대한 답이면 true"},
                "need_confirm": {"type": ["boolean", "null"], "description": "되물어야 하면 true(새 목적지 제안·취소·다시 출발). 확정·단답이면 false"},
                "confidence": {"type": "number", "description": "0~1"},
                "wait_minutes": {"type": ["integer", "null"], "description": "대기 요청이면 분"},
                "reply": {"type": "string", "description": "사용자에게 할 짧은 말. 없으면 빈 문자열"},
            },
            "required": ["intent", "heard_text"],
        },
    }
    if intent_recut.on():  # 원칙 4·9·10·11 — required 는 그대로
        props = tool["parameters"]["properties"]
        props["intent"] = {"type": "string", "enum": [*_INTENTS, intent_recut.NOT_FOR_ROBOT, intent_recut.DISMISS]}
        props["clarify_kind"] = {"type": "string", "enum": ["", *intent_recut.CLARIFY_KINDS],
                                 "description": "intent 가 clarify 일 때만: several / not_available / no_destination. 그 외 빈 문자열"}
        props["clarify_options"] = {"type": "array", "items": {"type": "string"},
                                    "description": "clarify_kind 가 several 일 때 목록의 name 을 모두"}
        props["clarify_wanted"] = {"type": "string",
                                   "description": "clarify_kind 가 not_available 일 때 사용자가 찾는 것을 말한 낱말 그대로"}
        if fixed_answers.on():   # 원칙 28·29 — 재단 위에서만 켜진다
            props["intent"]["enum"] += [fixed_answers.REPEAT, fixed_answers.PLACES]
    if answer_field.on():
        tool["parameters"]["properties"]["answer"] = {
            "type": "string", "enum": ["", *answer_field.ANSWERS], "description": answer_field.ANSWER_DESCRIPTION}
        # 3.5 — 도구 목록은 세션 단위라 질문이 열렸는지 모른다. 늘 채우게 하고, 질문이 없으면 모델이 내는 빈 문자열을
        # answer_field.fill 이 쓰지 않는다.
        tool["parameters"]["required"].append("answer")
    if fixed_answers.on():   # 원칙 25·10(Task N1) — question 의 할 말을 비워 보내지 못하게 꼭 채우는 칸으로
        tool["parameters"]["properties"]["reply"] = {"type": "string", "description": _REPLY_DESCRIPTION_TALK}
        tool["parameters"]["required"].append("reply")
    return tool


def _trail_items(n: int, entry: dict) -> list[dict]:
    """예약 도구 호출 기록 하나 → function_call + function_call_output 항목."""
    call_id = f"reserve_{n}"
    args = {**entry.get("args", {}), "heard_text": entry.get("heard", "")}
    return [{"type": "function_call", "call_id": call_id, "name": RESERVE,
             "arguments": json.dumps(args, ensure_ascii=False)},
            {"type": "function_call_output", "call_id": call_id,
             "output": json.dumps({"status": "asked", "question": entry.get("asked", "")}, ensure_ascii=False)}]


def history_to_items(history: Optional[Sequence[Any]], trail: Optional[Sequence[dict]] = None) -> list[dict]:
    """LangChain 이력 → Realtime 대화 항목. Human→user/input_text, AI→assistant/output_text.

    trail(예약 도구 호출 기록, ReservationFlow.tool_trail)이 있으면 각 호출을 그 결과로 물은 질문(assistant 줄) 바로
    앞에 function_call·function_call_output 으로 넣는다. 그 질문이 기록에 아직 없으면 맨 끝에 둔다.
    """
    items = _message_items(history)
    if not trail:
        return items
    at, tail = 0, []
    for n, entry in enumerate(trail):
        found = next((i for i in range(at, len(items))
                      if items[i].get("role") == "assistant"
                      and items[i]["content"][0]["text"] == entry.get("asked")), None)
        if found is None:
            tail += _trail_items(n, entry)
            continue
        extra = _trail_items(n, entry)
        items[found:found] = extra
        at = found + len(extra) + 1
    return items + tail


def _message_items(history: Optional[Sequence[Any]]) -> list[dict]:
    items: list[dict] = []
    for msg in history or []:
        role = getattr(msg, "type", "")
        text = str(getattr(msg, "content", "") or "")
        if not text:
            continue
        if role == "human":
            items.append({"type": "message", "role": "user",
                          "content": [{"type": "input_text", "text": text}]})
        elif role == "ai":
            items.append({"type": "message", "role": "assistant",
                          "content": [{"type": "output_text", "text": text}]})
    return items


class RealtimeConnectionError(ConnectionError):
    """Realtime 에 닿지 못했다 — 접속 실패, 또는 주고받는 중 연결이 끊김(2026-10-24 사용자).

    응답 지연(TimeoutError: 연결은 됐는데 timeout_sec 안에 답이 없음)과 나눈다. 호출부(ros_node)는
    연결 오류면 글자 경로를 바로 로컬 LLM 으로, 지연이면 클라우드 글자 LLM(gpt-5.4-mini)부터 보낸다.
    """


def _is_link_failure(exc: BaseException) -> bool:
    """소켓이 끊겼거나 닿지 않는 실패인가. websockets 를 import 하지 않고 클래스 이름으로도 본다
    (ConnectionClosedError 는 OSError 가 아니다). TimeoutError 는 OSError 의 하위지만 지연이라 뺀다.
    ConnectionClosedOK(서버의 정상 종료 1000 — 세션 최대 시간 등)는 인터넷 문제가 아니라 뺀다."""
    if isinstance(exc, TimeoutError):
        return False
    if isinstance(exc, (OSError, EOFError)):
        return True
    names = {c.__name__ for c in type(exc).__mro__}
    # WebSocketConnectionClosedError: openai 가 보내지 못한 메시지를 안은 채 ConnectionClosedError 를 다시 던지는 꼴
    return bool(names & {"ConnectionClosedError", "WebSocketConnectionClosedError", "APIConnectionError"})


def _as_connection_error(exc: BaseException, where: str) -> BaseException:
    if isinstance(exc, RealtimeConnectionError) or not _is_link_failure(exc):
        return exc
    return RealtimeConnectionError(f"{where}({type(exc).__name__}: {exc})")


@dataclass
class RealtimeResult:
    draft: dict
    heard_text: str
    latency_sec: float
    usage: dict = field(default_factory=dict)
    tool: str = TOOL_NAME        # 모델이 부른 도구 — "set_intent" 또는 "reserve_facility"(draft 는 그 도구의 인자)


def _default_connect(model: str):
    from openai import OpenAI  # 지연 import — 시험은 가짜 연결을 쓴다
    return OpenAI().realtime.connect(model=model).__enter__()


class RealtimeIntentClient:
    """연결 하나를 유지하며 발화마다 out-of-band 응답을 받는다. 스레드 안전(락)."""

    def __init__(self, model: str = DEFAULT_MODEL, timeout_sec: float = 8.0,
                 connect: Optional[Callable[[], Any]] = None) -> None:
        self.model = model
        self.timeout_sec = timeout_sec
        self._connect = connect or (lambda: _default_connect(model))
        self._conn = None
        self._lock = threading.Lock()
        # 마지막 실패 시각(monotonic). 복도처럼 와이파이가 끊긴 곳에서 발화마다 8초를
        # 기다리지 않도록, 호출부가 recently_failed() 로 잠시 건너뛴다(2026-09-20 실기).
        self.last_failure_at: Optional[float] = None

    def recently_failed(self, within_sec: float) -> bool:
        """마지막 ask 가 within_sec 안에 실패했는가(성공하면 지워진다)."""
        return (self.last_failure_at is not None
                and (time.monotonic() - self.last_failure_at) < within_sec)

    # ----- 연결 --------------------------------------------------------
    def _ensure_conn(self):
        if self._conn is None:
            try:
                conn = self._connect()
            except Exception as exc:
                # 접속 단계의 실패는 무엇이든 연결 오류다 — DNS·거부·여는 중 시간 초과(open_timeout)·인증 거절
                raise RealtimeConnectionError(f"접속 실패({type(exc).__name__}: {exc})") from exc
            try:
                conn.session.update(session=self._session_config())
            except BaseException as exc:
                try:
                    conn.close()   # 세션 설정에 실패한 연결은 바로 닫는다 — 누수 방지
                except Exception:
                    pass
                err = _as_connection_error(exc, "세션 설정 중 끊김")
                if err is exc:
                    raise
                raise err from exc
            self._conn = conn
        return self._conn

    def _session_config(self) -> dict:
        tools, tool_choice = session_tools()
        return {
            "type": "realtime",
            "output_modalities": ["text"],
            "audio": {"input": {
                "format": {"type": "audio/pcm", "rate": SAMPLE_RATE_RT},
                "noise_reduction": {"type": "far_field"},
                "turn_detection": None,
            }},
            "tools": tools,
            "tool_choice": tool_choice,
        }

    def close(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def warm(self) -> float:
        """접속을 미리 맺어 첫 발화의 콜드스타트를 없앤다. 실패는 그대로 올린다(호출부가 무시한다)."""
        with self._lock:
            started = time.monotonic()
            self._ensure_conn()
            return time.monotonic() - started

    # ----- 호출 --------------------------------------------------------
    def ask(self, pcm16_16k: bytes, history: Optional[Sequence[Any]], instructions: str,
            tool_trail: Optional[Sequence[dict]] = None) -> RealtimeResult:
        """소리 + 이력 + 지시문 → 도구 인자(set_intent, 켜져 있으면 reserve_facility 도). 실패·timeout 이면 예외(연결은 버린다)."""
        try:
            result = self._ask(pcm16_16k, history, instructions, tool_trail=tool_trail)
        except BaseException:
            self.last_failure_at = time.monotonic()
            raise
        self.last_failure_at = None
        return result

    def ask_text(self, text: str, history: Optional[Sequence[Any]], instructions: str,
                 tool_trail: Optional[Sequence[dict]] = None) -> RealtimeResult:
        """재생 측정용(tools/reserve_replay) — 소리 대신 글자를 넣는다. 운영 경로는 VICA_TALK 이 켜졌을 때 소리 경로의 빈 대답
        재시도(Task N1)에서만 부른다. 들린 말(heard_text)이 비어 오면 넣은 글자를 쓴다."""
        result = self._ask(b"", history, instructions, text=text, tool_trail=tool_trail)
        if not result.heard_text:
            result.heard_text = text
        return result

    def _ask(self, pcm16_16k: bytes, history: Optional[Sequence[Any]], instructions: str,
             text: Optional[str] = None, tool_trail: Optional[Sequence[dict]] = None) -> RealtimeResult:
        with self._lock:
            started = time.monotonic()
            conn = self._ensure_conn()
            # 예약 도구가 세션에 없으면 그 호출 기록도 넣지 않는다 — 모르는 함수의 기록이 된다.
            trail = tool_trail if tool_trail and audio_tools_active() else None
            items = history_to_items(history, trail)
            if text is None:
                audio_b64 = base64.b64encode(resample_pcm16(pcm16_16k)).decode("ascii")
                items.append({"type": "message", "role": "user",
                              "content": [{"type": "input_audio", "audio": audio_b64}]})
            else:
                items.append({"type": "message", "role": "user",
                              "content": [{"type": "input_text", "text": text}]})
            try:
                conn.response.create(response={
                    "conversation": "none",
                    "output_modalities": ["text"],
                    "instructions": instructions,
                    "tool_choice": session_tools()[1],
                    "input": items,
                })
                done = self._wait_done(conn)
            except BaseException as exc:
                self.close()   # 다음 호출이 새로 연결한다
                err = _as_connection_error(exc, "주고받는 중 끊김")
                if err is exc:
                    raise
                raise err from exc
            tool, draft, usage = self._parse_done(done)
            heard = str(draft.pop("heard_text", "") or "")
            return RealtimeResult(draft=draft, heard_text=heard,
                                  latency_sec=time.monotonic() - started, usage=usage, tool=tool)

    def _wait_done(self, conn):
        """이벤트를 워커 스레드에서 읽고 timeout 이면 연결을 닫아 워커를 깨운다."""
        box: dict = {}

        def worker():
            try:
                for ev in conn:
                    if ev.type == "error":
                        err = getattr(ev, "error", None)
                        box["exc"] = RuntimeError(str(getattr(err, "message", err)))
                        return
                    if ev.type == "response.done":
                        box["done"] = ev
                        return
                # openai 의 이벤트 반복은 정상 종료(ConnectionClosedOK)를 조용히 끝낸다 — 서버가 세션을 닫은 것이라
                # 연결 오류로 보지 않는다(비정상 끊김은 ConnectionClosedError 로 아래 except 에 온다).
                box["exc"] = RuntimeError("연결이 응답 없이 닫혔다")
            except Exception as exc:  # 소켓 오류 등
                box["exc"] = exc

        th = threading.Thread(target=worker, daemon=True, name="realtime-recv")
        th.start()
        th.join(self.timeout_sec)
        if th.is_alive():
            self.close()
            th.join(1.0)
            raise TimeoutError(f"Realtime 응답 {self.timeout_sec:.0f}초 초과")
        if "exc" in box:
            raise box["exc"]
        return box["done"]

    @staticmethod
    def _parse_done(done) -> tuple[str, dict, dict]:
        """response.done → (도구 이름, 인자, usage). 세션에 준 도구가 아니면 ValueError."""
        allowed = {t["name"] for t in session_tools()[0]}
        resp = done.response
        status = getattr(resp, "status", None) or "completed"
        if status != "completed":
            # 한도 초과(rate_limit_exceeded)·취소·불완전 응답은 output 이 비어 온다 — 사유를 그대로 싣는다.
            details = getattr(resp, "status_details", None)
            err = getattr(details, "error", None)
            why = getattr(err, "code", None) or getattr(details, "reason", None) or "사유 없음"
            msg = getattr(err, "message", None) or ""
            raise RuntimeError(f"Realtime 응답 {status}: {why} {msg}".strip())
        for item in getattr(resp, "output", []) or []:
            name = getattr(item, "name", "")
            if getattr(item, "type", "") == "function_call" and name in allowed:
                try:
                    draft = json.loads(item.arguments or "{}")
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{name} 인자가 JSON 이 아니다: {exc}") from exc
                if not isinstance(draft, dict):
                    raise ValueError(f"{name} 인자가 객체가 아니다")
                return name, draft, _usage_dict(getattr(resp, "usage", None))
        raise ValueError("응답에 도구 호출이 없다")


def _usage_dict(usage) -> dict:
    if usage is None:
        return {}
    det = getattr(usage, "input_token_details", None)
    return {
        "audio_tokens": int(getattr(det, "audio_tokens", 0) or 0),
        "text_tokens": int(getattr(det, "text_tokens", 0) or 0),
        # 직전 호출과 앞부분이 같아 반값으로 계산된 입력 토큰(지시문 캐시 실측용, 2026-09-21)
        "cached_tokens": int(getattr(det, "cached_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
    }


_CLIENT: Optional[RealtimeIntentClient] = None


def get_realtime_client() -> RealtimeIntentClient:
    """env 기반 싱글턴. VICA_REALTIME_MODEL, VICA_REALTIME_TIMEOUT."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = RealtimeIntentClient(
            model=os.environ.get("VICA_REALTIME_MODEL", DEFAULT_MODEL),
            timeout_sec=float(os.environ.get("VICA_REALTIME_TIMEOUT", "8")),
        )
    return _CLIENT


def reset_realtime_client() -> None:
    global _CLIENT
    if _CLIENT is not None:
        _CLIENT.close()
    _CLIENT = None


AUDIO_MSG_LABEL = "pcm16_mono_16000"


def pcm16_from_audio_msg(data, label: str) -> bytes:
    """/vica/user_audio 메시지의 data 와 layout.dim[0].label → pcm16 bytes. 라벨이 다르면 ValueError."""
    if label != AUDIO_MSG_LABEL:
        raise ValueError(f"오디오 라벨이 {AUDIO_MSG_LABEL} 이 아니다: {label!r}")
    return bytes(data)


AUDIO_TURN_MAX_AGE_SEC = 15.0


def audio_turn_applies(turn: dict, now: float, max_age_sec: float = AUDIO_TURN_MAX_AGE_SEC) -> bool:
    """직전 소리 발화 결과를 지금 도착한 텍스트에 붙여도 되는가.

    handled 이고, 만든 지 max_age_sec 안이어야 한다. 한 번 쓰면 노드가 버린다(pop) —
    소리 없이 들어온 텍스트(예: 긴급 검증 구제 경로)에 옛 결과가 새어 발화가
    소실되는 것을 막는다(2026-09-20 리뷰).
    """
    if not turn or not turn.get("handled"):
        return False
    t = turn.get("t")
    return t is not None and 0.0 <= (now - t) <= max_age_sec


def ab_shadow_enabled() -> bool:
    """audio 모드에서 같은 발화를 글자 경로(GPT)로도 판단해 [A/B] 비교 기록을 남길까 (VICA_AB_SHADOW).

    기본은 꺼짐(2026-10-10 사용자 결정). 비교 결과는 로봇 동작에 쓰지 않는데 말할 때마다 OpenAI
    호출이 하나 더 나갔고, 그 호출이 실패하면 백엔드가 로컬로 대피해 비교용으로 로컬 모델까지
    돌았다. Realtime 이 실패했을 때 글자 경로가 대신 판단하는 예비 길은 이 스위치와 무관하다.
    매번 환경변수를 읽는다 — 시험이 monkeypatch 로 켜고 끌 수 있게.
    """
    return os.environ.get("VICA_AB_SHADOW", "").strip().lower() in ("1", "true", "on", "yes")

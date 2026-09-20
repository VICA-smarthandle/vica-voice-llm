"""OpenAI Realtime 로 발화 **소리**에서 바로 의도(_IntentDraft 인자)를 받는다.

설계 근거: docs/realtime-api-vica-적용-가이드.md §4 방식 2. 2026-09-20 실측으로
확인한 방식 — out-of-band 응답(conversation: "none") 에 대화 이력(글자)과 소리
항목을 넣고 set_intent 함수 호출을 강제한다. 세션 대화 상태를 쓰지 않으므로
우리 ConversationHistory 가 유일한 맥락이고, 세션이 길어져도 쌓이는 것이 없다.

안전 경계: 이 모듈은 의도 초안(dict)만 돌려준다. 확정·발행은 파서의 _finalize 와
LLM 노드가 한다. 긴급어는 상류(웨이크워드 노드 whisper)가 잡는다.
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

SAMPLE_RATE_IN = 16000   # 웨이크워드 노드 클립
SAMPLE_RATE_RT = 24000   # Realtime 입력 형식
DEFAULT_MODEL = "gpt-realtime-2.1-mini"
TOOL_NAME = "set_intent"
_INTENTS = ["navigate", "question", "clarify", "unknown", "cancel", "pause", "resume",
            "affirm", "deny", "wait", "finish"]


def float32_to_pcm16(audio) -> bytes:
    """float32 [-1, 1] → int16 LE. 범위를 넘는 값은 자른다."""
    arr = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (arr * 32767.0).astype("<i2").tobytes()


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


def build_intent_tool() -> dict:
    """_IntentDraft 와 같은 필드 + heard_text. 파서가 인자를 _IntentDraft 로 검증한다."""
    return {
        "type": "function",
        "name": TOOL_NAME,
        "description": "사용자 발화(소리)의 의도를 정한다. 반드시 이 함수로만 답한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": _INTENTS},
                "heard_text": {"type": "string", "description": "들린 말을 한국어로 그대로 적는다. 말이 아니면 빈 문자열"},
                "destination_candidate": {"type": ["string", "null"], "description": "목적지 표현(목록의 이름 그대로)"},
                "is_confirmation": {"type": ["boolean", "null"], "description": "직전 확인 질문에 대한 답이면 true"},
                "confidence": {"type": "number", "description": "0~1"},
                "wait_minutes": {"type": ["integer", "null"], "description": "대기 요청이면 분"},
                "reply": {"type": "string", "description": "사용자에게 할 짧은 말. 없으면 빈 문자열"},
            },
            "required": ["intent", "heard_text"],
        },
    }


def history_to_items(history: Optional[Sequence[Any]]) -> list[dict]:
    """LangChain 이력 → Realtime 대화 항목. Human→user/input_text, AI→assistant/output_text."""
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


@dataclass
class RealtimeResult:
    draft: dict
    heard_text: str
    latency_sec: float
    usage: dict = field(default_factory=dict)


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

    # ----- 연결 --------------------------------------------------------
    def _ensure_conn(self):
        if self._conn is None:
            conn = self._connect()
            try:
                conn.session.update(session=self._session_config())
            except BaseException:
                try:
                    conn.close()   # 세션 설정에 실패한 연결은 바로 닫는다 — 누수 방지
                except Exception:
                    pass
                raise
            self._conn = conn
        return self._conn

    def _session_config(self) -> dict:
        return {
            "type": "realtime",
            "output_modalities": ["text"],
            "audio": {"input": {
                "format": {"type": "audio/pcm", "rate": SAMPLE_RATE_RT},
                "noise_reduction": {"type": "far_field"},
                "turn_detection": None,
            }},
            "tools": [build_intent_tool()],
            "tool_choice": {"type": "function", "name": TOOL_NAME},
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
    def ask(self, pcm16_16k: bytes, history: Optional[Sequence[Any]], instructions: str) -> RealtimeResult:
        """소리 + 이력 + 지시문 → set_intent 인자. 실패·timeout 이면 예외(연결은 버린다)."""
        with self._lock:
            started = time.monotonic()
            conn = self._ensure_conn()
            audio_b64 = base64.b64encode(resample_pcm16(pcm16_16k)).decode("ascii")
            items = history_to_items(history)
            items.append({"type": "message", "role": "user",
                          "content": [{"type": "input_audio", "audio": audio_b64}]})
            try:
                conn.response.create(response={
                    "conversation": "none",
                    "output_modalities": ["text"],
                    "instructions": instructions,
                    "tool_choice": {"type": "function", "name": TOOL_NAME},
                    "input": items,
                })
                done = self._wait_done(conn)
            except BaseException:
                self.close()   # 다음 호출이 새로 연결한다
                raise
            draft, usage = self._parse_done(done)
            heard = str(draft.pop("heard_text", "") or "")
            return RealtimeResult(draft=draft, heard_text=heard,
                                  latency_sec=time.monotonic() - started, usage=usage)

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
    def _parse_done(done) -> tuple[dict, dict]:
        for item in getattr(done.response, "output", []) or []:
            if getattr(item, "type", "") == "function_call" and getattr(item, "name", "") == TOOL_NAME:
                try:
                    draft = json.loads(item.arguments or "{}")
                except json.JSONDecodeError as exc:
                    raise ValueError(f"set_intent 인자가 JSON 이 아니다: {exc}") from exc
                if not isinstance(draft, dict):
                    raise ValueError("set_intent 인자가 객체가 아니다")
                return draft, _usage_dict(getattr(done.response, "usage", None))
        raise ValueError("응답에 set_intent 호출이 없다")


def _usage_dict(usage) -> dict:
    if usage is None:
        return {}
    det = getattr(usage, "input_token_details", None)
    return {
        "audio_tokens": int(getattr(det, "audio_tokens", 0) or 0),
        "text_tokens": int(getattr(det, "text_tokens", 0) or 0),
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

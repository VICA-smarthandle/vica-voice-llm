# Realtime 소리→의도 직행(audio 모드) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `.env` 한 줄(`VICA_INTENT_INPUT=audio`)로 발화 소리를 OpenAI Realtime 에 직접 보내 의도(VicaIntent)를 받는 경로를 켜고, 같은 발화의 기존 경로(whisper→gpt-5.4-mini) 결과를 로그에 나란히 남겨 실주행에서 A/B 비교한다.

**Architecture:** 웨이크워드 노드가 청취 창에서 확보한 발화 클립(float32 16 kHz)을 whisper 전에 `/vica/user_audio`(std_msgs/UInt8MultiArray, pcm16 mono 16 kHz)로 발행한다. LLM 노드는 audio 모드에서 그 클립을 `parse_intent_audio()`에 넣는다. 새 모듈 `src/realtime_intent.py`가 Realtime WebSocket(openai SDK `client.realtime.connect`)에 out-of-band 응답(`conversation: "none"`, `input` = 대화 이력 글자 + 소리 항목)을 요청하고 강제 함수 호출 `set_intent` 의 인자(JSON)를 돌려준다. 그 인자는 기존 `_IntentDraft` → `_finalize()` 를 그대로 탄다. 긴급어·Mission·Safety 경로는 무변경. Realtime 실패·오프라인이면 그 발화는 기존 텍스트 경로로 처리한다.

**Tech Stack:** Python 3.10, openai SDK 3.1.0(`client.realtime.connect`, websockets 15), numpy, rclpy(Humble), std_msgs, pydantic, pytest(젯슨 `.venv`).

**Spec:** 설계 근거 `docs/realtime-api-vica-적용-가이드.md` §4 방식 2 (이 계획이 정본 설계를 겸한다). 2026-09-20 실측: 접속+세션 3.3초(1회), 1초 소리 함수 호출 응답 0.68초, 소리 10토큰/초, 삐 소리 → `unknown`.

## Global Constraints

- 모든 실행·시험은 젯슨 `.venv/bin/python`. 자동 시험은 네트워크·ROS·OpenAI 를 쓰지 않는다(가짜 연결). 실제 API 호출은 `tools/` 의 수동 스모크뿐.
- `VICA_INTENT_INPUT` 기본값 `text` = 지금과 완전히 같은 동작(소리 발행 없음, 구독은 있어도 무시).
- 긴급어 경로(웨이크워드 노드 `_verify_emergency` → `/vica/emergency`)와 `/vica/user_text` 발행은 그대로 둔다. audio 모드에서도 whisper 는 계속 돈다.
- Realtime 결과는 반드시 `_finalize()` 를 거친 `VicaIntent` 제안이다. `/cmd_vel*`·Nav2·CAN 을 만지지 않는다.
- Realtime 호출 상한 `VICA_REALTIME_TIMEOUT`(기본 8초). 실패(예외·timeout)·`LlmBackendManager.state is LOCAL` 이면 그 발화는 텍스트 경로가 처리한다.
- 오디오 토픽 `/vica/user_audio`: `std_msgs/msg/UInt8MultiArray`, `layout.dim[0].label == "pcm16_mono_16000"`, data = pcm16 little-endian mono 16 kHz. 생산자 웨이크워드 노드, 소비자 LLM 노드(음성 저장소 내부 계약).
- Realtime 세션: `type: "realtime"`, `output_modalities: ["text"]`, `audio.input.format {type: "audio/pcm", rate: 24000}`, `audio.input.noise_reduction {type: "far_field"}`, `audio.input.turn_detection: None`, `tools: [set_intent]`, `tool_choice: {type: "function", name: "set_intent"}`. 모델 기본 `gpt-realtime-2.1-mini`.
- 로그 문구(정확히): `[A/B] audio={intent}/{dest} {t:.2f}s | text={intent}/{dest} {t:.2f}s | heard='{heard}' | whisper='{text}'`. 텍스트 경로 결과가 아직 없으면 `text=?`. 
- 사용자 멘트 추가 없음. 새 노드 없음. 주석·문서는 평서형 한국어.
- 커밋은 사용자 허락(09-20) — 작업마다 커밋, push 안 함. 브랜치 `feat/realtime-intent`.

---

## 파일 구조

| 파일 | 책임 |
| --- | --- |
| `src/realtime_intent.py` (신규) | 리샘플, 도구 스키마, 이력→항목 변환, `RealtimeIntentClient.ask()`, 싱글턴 |
| `tests/test_realtime_intent.py` (신규) | 위 모듈 전부(가짜 연결) |
| `src/langchain_intent_parser.py` (수정) | 지름길 블록을 `_shortcut_intent()` 로 분리, `parse_intent_audio()` 추가 |
| `tests/test_parser_audio.py` (신규) | `parse_intent_audio` 경로(가짜 클라이언트) |
| `src/wakeword_monitor.py` (수정) | `on_user_audio` 콜백(청취 창 전사 직전) |
| `src/ros_wakeword_node.py` (수정) | audio 모드면 클립을 `/vica/user_audio` 로 발행 |
| `src/ros_node.py` (수정) | `/vica/user_audio` 구독, audio 모드 분기, 그림자 비교 로그, 공통 후처리 분리 |
| `tools/realtime_intent_smoke.py` (신규) | supertonic 으로 문장 합성 → `parse_intent_audio` 실호출(수동) |
| `.env.example` (수정) | `VICA_INTENT_INPUT`, `VICA_REALTIME_*` |

---

### Task 1: Realtime 클라이언트 모듈

**Files:**
- Create: `src/realtime_intent.py`
- Test: `tests/test_realtime_intent.py`

**Interfaces:**
- Produces:
  - `SAMPLE_RATE_IN = 16000`, `SAMPLE_RATE_RT = 24000`
  - `def resample_pcm16(pcm16: bytes, src_rate: int = 16000, dst_rate: int = 24000) -> bytes`
  - `def float32_to_pcm16(audio) -> bytes` (numpy float32 [-1,1] → int16 LE bytes)
  - `def build_intent_tool() -> dict` (Realtime function tool `set_intent`)
  - `def history_to_items(history) -> list[dict]` (LangChain Human/AI 메시지 → Realtime 항목)
  - `@dataclass RealtimeResult: draft: dict, heard_text: str, latency_sec: float, usage: dict`
  - `class RealtimeIntentClient(model: str, timeout_sec: float = 8.0, connect: Optional[Callable] = None)` with `ask(pcm16_16k: bytes, history, instructions: str) -> RealtimeResult`, `close()`
  - `def get_realtime_client() -> RealtimeIntentClient` (env 기반 지연 싱글턴), `reset_realtime_client()`

- [ ] **Step 1: 실패 시험 작성**

`tests/test_realtime_intent.py`:

```python
"""Realtime 소리→의도 클라이언트 시험 (네트워크 없음 — 가짜 연결).

정본: docs/superpowers/plans/2026-09-20-realtime-intent.md Task 1.
"""
import json
import types

import numpy as np
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.realtime_intent import (
    RealtimeIntentClient,
    RealtimeResult,
    build_intent_tool,
    float32_to_pcm16,
    history_to_items,
    resample_pcm16,
)


class TestAudioHelpers:
    def test_float32_to_pcm16_clips_and_scales(self):
        audio = np.array([0.0, 0.5, -0.5, 2.0, -2.0], dtype=np.float32)
        pcm = float32_to_pcm16(audio)
        ints = np.frombuffer(pcm, dtype="<i2")
        assert list(ints) == [0, 16383, -16383, 32767, -32767]

    def test_resample_ratio_and_endpoints(self):
        src = np.linspace(-1000, 1000, 1600).astype("<i2").tobytes()  # 0.1초 @16k
        out = resample_pcm16(src, 16000, 24000)
        ints = np.frombuffer(out, dtype="<i2")
        assert len(ints) == 2400
        assert ints[0] == -1000 and abs(int(ints[-1]) - 1000) <= 1

    def test_resample_same_rate_is_identity(self):
        src = np.arange(10, dtype="<i2").tobytes()
        assert resample_pcm16(src, 16000, 16000) == src


class TestTool:
    def test_tool_shape(self):
        tool = build_intent_tool()
        assert tool["type"] == "function" and tool["name"] == "set_intent"
        props = tool["parameters"]["properties"]
        for key in ("intent", "heard_text", "destination_candidate", "is_confirmation",
                    "confidence", "wait_minutes", "reply"):
            assert key in props
        assert "navigate" in props["intent"]["enum"] and "affirm" in props["intent"]["enum"]
        assert set(tool["parameters"]["required"]) >= {"intent", "heard_text"}


class TestHistory:
    def test_history_to_items(self):
        items = history_to_items([HumanMessage("화장실로 가줘"), AIMessage("화장실로 안내해드릴까요?")])
        assert items == [
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "화장실로 가줘"}]},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "화장실로 안내해드릴까요?"}]},
        ]

    def test_history_none_is_empty(self):
        assert history_to_items(None) == []


class FakeEvent:
    def __init__(self, type_, **kw):
        self.type = type_
        self.__dict__.update(kw)


class FakeConn:
    """openai realtime 연결 흉내: session.update / response.create 기록, 이벤트 재생."""

    def __init__(self, events):
        self.events = list(events)
        self.session_updates = []
        self.responses = []
        self.closed = False
        conn = self
        self.session = types.SimpleNamespace(update=lambda session: conn.session_updates.append(session))
        self.response = types.SimpleNamespace(create=lambda response: conn.responses.append(response))

    def __iter__(self):
        return iter(self.events)

    def close(self):
        self.closed = True


def done_event(args: dict, audio_tokens=20, text_tokens=400):
    item = types.SimpleNamespace(type="function_call", name="set_intent", arguments=json.dumps(args))
    usage = types.SimpleNamespace(
        input_token_details=types.SimpleNamespace(audio_tokens=audio_tokens, text_tokens=text_tokens),
        output_tokens=30)
    return FakeEvent("response.done", response=types.SimpleNamespace(output=[item], usage=usage))


def make_client(events, timeout=2.0):
    conns = []

    def connect():
        conn = FakeConn(events)
        conns.append(conn)
        return conn

    client = RealtimeIntentClient(model="fake", timeout_sec=timeout, connect=connect)
    return client, conns


PCM = np.zeros(1600, dtype="<i2").tobytes()  # 0.1초 무음 @16k


class TestAsk:
    def test_ask_returns_draft_and_heard_text(self):
        client, conns = make_client([
            done_event({"intent": "wait", "heard_text": "오 분", "wait_minutes": 5, "reply": ""})])
        result = client.ask(PCM, [HumanMessage("얼마나 기다릴까요?")], "지시문")
        assert isinstance(result, RealtimeResult)
        assert result.draft["intent"] == "wait" and result.draft["wait_minutes"] == 5
        assert result.heard_text == "오 분"
        assert result.usage["audio_tokens"] == 20
        req = conns[0].responses[0]
        assert req["conversation"] == "none"
        assert req["tool_choice"] == {"type": "function", "name": "set_intent"}
        assert req["instructions"] == "지시문"
        assert req["input"][0]["content"][0]["text"] == "얼마나 기다릴까요?"
        assert req["input"][-1]["content"][0]["type"] == "input_audio"
        assert len(conns[0].session_updates) == 1  # 세션 설정은 연결당 1회

    def test_session_config(self):
        client, conns = make_client([done_event({"intent": "unknown", "heard_text": ""})])
        client.ask(PCM, None, "x")
        session = conns[0].session_updates[0]
        assert session["type"] == "realtime"
        assert session["output_modalities"] == ["text"]
        assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
        assert session["audio"]["input"]["noise_reduction"] == {"type": "far_field"}
        assert session["audio"]["input"]["turn_detection"] is None
        assert session["tools"][0]["name"] == "set_intent"

    def test_reuses_connection_across_calls(self):
        events = [done_event({"intent": "unknown", "heard_text": "a"}),
                  done_event({"intent": "unknown", "heard_text": "b"})]
        client, conns = make_client(events)
        client.ask(PCM, None, "x")
        client.ask(PCM, None, "x")
        assert len(conns) == 1

    def test_error_event_raises_and_reconnects_next_time(self):
        client, conns = make_client([FakeEvent("error", error=types.SimpleNamespace(message="bad"))])
        with pytest.raises(RuntimeError, match="bad"):
            client.ask(PCM, None, "x")
        assert conns[0].closed is True
        conns[0].events = [done_event({"intent": "unknown", "heard_text": ""})]
        client.ask(PCM, None, "x")
        assert len(conns) == 2

    def test_timeout_raises(self):
        class HangingConn(FakeConn):
            def __iter__(self):
                import time
                while not self.closed:
                    time.sleep(0.05)
                return iter(())

        conns = []

        def connect():
            conn = HangingConn([])
            conns.append(conn)
            return conn

        client = RealtimeIntentClient(model="fake", timeout_sec=0.3, connect=connect)
        with pytest.raises(TimeoutError):
            client.ask(PCM, None, "x")
        assert conns[0].closed is True

    def test_bad_arguments_json_raises(self):
        item = types.SimpleNamespace(type="function_call", name="set_intent", arguments="{not json")
        ev = FakeEvent("response.done", response=types.SimpleNamespace(output=[item], usage=None))
        client, _ = make_client([ev])
        with pytest.raises(ValueError):
            client.ask(PCM, None, "x")
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_realtime_intent.py -q`
Expected: `ModuleNotFoundError: No module named 'src.realtime_intent'`

- [ ] **Step 3: 모듈 작성**

`src/realtime_intent.py`:

```python
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
            conn.session.update(session=self._session_config())
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
```

주의: `_default_connect` 는 SDK 컨텍스트 매니저를 `__enter__()` 로 열어 연결 객체를 얻는다. 닫을 때는 `conn.close()` 를 쓴다(2026-09-20 스모크에서 `with client.realtime.connect(...) as conn` 로 확인). SDK 이벤트 객체는 `ev.type`·`ev.response.output[i].type/name/arguments`·`ev.response.usage.input_token_details.audio_tokens` 를 가진다(스모크 출력으로 확인).

- [ ] **Step 4: 통과 확인**

Run: `.venv/bin/python -m pytest tests/test_realtime_intent.py -q`
Expected: `13 passed`

- [ ] **Step 5: 커밋**

```bash
git add src/realtime_intent.py tests/test_realtime_intent.py
git commit -m "feat(voice): Realtime 소리→의도 클라이언트 — out-of-band 응답·set_intent 강제"
```

---

### Task 2: 파서 — 지름길 분리와 `parse_intent_audio`

**Files:**
- Modify: `src/langchain_intent_parser.py` (지름길 블록 ≈ 440~536행, `parse_intent` 끝부분)
- Test: `tests/test_parser_audio.py` (신규)

**Interfaces:**
- Consumes: Task 1 `get_realtime_client()`, `RealtimeResult`
- Produces:
  - `def _shortcut_intent(user_text: str, history, destinations) -> Optional[VicaIntent]` — 기존 `parse_intent` 의 지름길(제어 확인 대기 → 목적지 확인 대기 긍/부정 → 호출어 → 취소 → 잠깐 → 단독 긍/부정)을 그대로 옮긴 함수. `parse_intent` 는 이 함수를 먼저 부르고 결과가 있으면 그대로 돌려준다(동작 불변).
  - `AUDIO_EXTRA_INSTRUCTIONS: str` — 소리 입력용 추가 지시문
  - `def parse_intent_audio(pcm16_16k: bytes, destinations, history=None, robot_state=None) -> tuple[VicaIntent, str, float]` — (의도, 들린 말, Realtime 지연초). Realtime 실패는 예외로 올린다(노드가 텍스트 경로로 넘긴다).

- [ ] **Step 1: 실패 시험 작성**

`tests/test_parser_audio.py`:

```python
"""소리→의도 직행 경로 시험 (Realtime 은 가짜 클라이언트)."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src import langchain_intent_parser as parser
from src.langchain_intent_parser import _shortcut_intent, parse_intent, parse_intent_audio
from src.realtime_intent import RealtimeResult
from src.schema import DestinationData

DEST = DestinationData(id="starlight_1f_restroom", name="별빛관 1층 화장실",
                       confirm_prompt="별빛관 1층 화장실로 안내해드릴까요?")
PCM = b"\x00\x00" * 1600


class FakeRT:
    def __init__(self, draft, heard, fail=None):
        self.draft, self.heard, self.fail = draft, heard, fail
        self.calls = []

    def ask(self, pcm, history, instructions):
        self.calls.append((pcm, list(history or []), instructions))
        if self.fail:
            raise self.fail
        return RealtimeResult(draft=dict(self.draft), heard_text=self.heard, latency_sec=0.7,
                              usage={"audio_tokens": 20})


@pytest.fixture
def rt(monkeypatch):
    holder = {}

    def use(draft, heard, fail=None):
        holder["c"] = FakeRT(draft, heard, fail)
        monkeypatch.setattr(parser, "get_realtime_client", lambda: holder["c"])
        return holder["c"]

    return use


class TestShortcutExtraction:
    def test_text_path_unchanged_for_wake_word(self):
        assert parse_intent("비카야", [DEST]).intent == "unknown"

    def test_shortcut_returns_none_for_normal_sentence(self):
        assert _shortcut_intent("화장실로 안내해줘", None, [DEST]) is None

    def test_shortcut_pending_affirm(self):
        hist = [HumanMessage("화장실"), AIMessage(DEST.confirm_prompt)]
        out = _shortcut_intent("그래", hist, [DEST])
        assert out.intent == "navigate" and out.matched_destination_id == DEST.id


class TestAudioPath:
    def test_navigate_draft_goes_through_finalize(self, rt):
        c = rt({"intent": "navigate", "destination_candidate": "별빛관 1층 화장실", "reply": ""},
               "화장실로 가줘")
        intent, heard, dt = parse_intent_audio(PCM, [DEST])
        assert intent.intent == "navigate" and intent.matched_destination_id == DEST.id
        assert heard == "화장실로 가줘" and dt == pytest.approx(0.7)
        assert "heard_text" in c.calls[0][2]  # 추가 지시문이 들어갔다
        assert parser.AUDIO_EXTRA_INSTRUCTIONS in c.calls[0][2]

    def test_wait_minutes_from_heard_text(self, rt):
        rt({"intent": "wait", "reply": "", "wait_minutes": 5}, "오 분")
        intent, heard, _ = parse_intent_audio(PCM, [DEST])
        assert intent.intent == "wait" and intent.wait_minutes == 5

    def test_affirm_with_pending_becomes_navigate(self, rt):
        rt({"intent": "affirm", "reply": ""}, "응응")
        hist = [HumanMessage("화장실"), AIMessage(DEST.confirm_prompt)]
        intent, _, _ = parse_intent_audio(PCM, [DEST], history=hist)
        assert intent.intent == "navigate" and intent.matched_destination_id == DEST.id
        assert intent.need_confirm is False

    def test_deny_with_pending_stays_deny(self, rt):
        rt({"intent": "deny", "reply": ""}, "아니 됐어")
        hist = [HumanMessage("화장실"), AIMessage(DEST.confirm_prompt)]
        intent, _, _ = parse_intent_audio(PCM, [DEST], history=hist)
        assert intent.intent == "deny"

    def test_heard_text_shortcut_wins_over_draft(self, rt):
        # 들린 말이 호출어면 초안(unknown)과 무관하게 호출 응답
        rt({"intent": "unknown", "reply": ""}, "비카야")
        intent, _, _ = parse_intent_audio(PCM, [DEST])
        assert intent.reply == parser.WAKE_GREETING

    def test_realtime_failure_propagates(self, rt):
        rt({}, "", fail=TimeoutError("8초"))
        with pytest.raises(TimeoutError):
            parse_intent_audio(PCM, [DEST])

    def test_bad_draft_raises_value_error(self, rt):
        rt({"intent": "not-an-intent", "reply": ""}, "x")
        with pytest.raises(ValueError):
            parse_intent_audio(PCM, [DEST])
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest tests/test_parser_audio.py -q`
Expected: `ImportError: cannot import name '_shortcut_intent'`

- [ ] **Step 3: 파서 수정**

(a) import 추가: `from .realtime_intent import get_realtime_client` (모듈 상단). `ValidationError` 는 `pydantic` 에서 import.

(b) `parse_intent` 의 지름길 블록(`pending_command = _pending_command(history)` 부터 `if word in _NEGATIVES: return ...` 까지)을 **그대로** 새 함수로 옮긴다:

```python
def _shortcut_intent(user_text: str, history: Optional[list[BaseMessage]],
                     destinations: Sequence[DestinationData]) -> Optional[VicaIntent]:
    """LLM 없이 코드가 확정하는 경우. 없으면 None. (본문은 종전 parse_intent 지름길 그대로)"""
    pending_command = _pending_command(history)
    if pending_command is not None:
        ...  # 기존 코드 그대로
    pending = _pending_confirm_destination(history, destinations)
    if pending is not None:
        ...  # 기존 코드 그대로
    word = _normalize_short_reply(user_text)
    ...  # _WAKE_WORDS / _CANCEL_WORDS / _PAUSE_WORDS / _SOLO_AFFIRMATIVES / _NEGATIVES 그대로
    return None
```

`parse_intent` 는 시작 부분이 다음이 된다(나머지는 그대로):

```python
    shortcut = _shortcut_intent(user_text, history, destinations)
    if shortcut is not None:
        return shortcut
    pending_command = _pending_command(history)
    pending = _pending_confirm_destination(history, destinations)
```

(c) 소리용 추가 지시문과 `parse_intent_audio` 를 `parse_intent` 뒤에 추가:

```python
AUDIO_EXTRA_INSTRUCTIONS = (
    "\n\n[소리 입력 규칙] 지금 입력은 글자가 아니라 사용자의 목소리다. 반드시 set_intent "
    "함수를 한 번 호출해 답한다. heard_text 에는 들린 말을 한국어로 그대로 적는다(말이 "
    "아니면 빈 문자열, intent 는 unknown). 짧은 긍정(네·응·그래·좋아)은 intent affirm, "
    "짧은 부정(아니·아니요·싫어)은 deny 로 적고, 직전에 로봇이 확인 질문을 했으면 "
    "is_confirmation 을 true 로 둔다. '오 분'·'한 시간'처럼 시간만 말하면 intent wait 와 "
    "wait_minutes 를 채운다. 로봇 자신의 안내 멘트가 들리면 unknown 으로 둔다."
)


def parse_intent_audio(
    pcm16_16k: bytes,
    destinations: Sequence[DestinationData],
    history: Optional[list[BaseMessage]] = None,
    robot_state: Optional[RobotState] = None,
) -> tuple[VicaIntent, str, float]:
    """발화 소리 → Realtime(set_intent) → 기존 _finalize. (의도, 들린 말, 지연초) 를 돌려준다.

    실패(예외·timeout)는 그대로 올린다 — LLM 노드가 그 발화를 텍스트 경로로 넘긴다.
    들린 말(heard_text)에 지름길 어휘가 있으면 텍스트 경로와 같은 지름길이 이긴다.
    """
    instructions = _build_system_prompt(destinations, robot_state) + AUDIO_EXTRA_INSTRUCTIONS
    result = get_realtime_client().ask(pcm16_16k, history, instructions)
    heard = result.heard_text.strip()

    if heard:
        shortcut = _shortcut_intent(heard, history, destinations)
        if shortcut is not None:
            return shortcut, heard, result.latency_sec

    try:
        draft = _IntentDraft(**result.draft)
    except ValidationError as exc:
        raise ValueError(f"set_intent 인자가 _IntentDraft 와 맞지 않는다: {exc}") from exc

    pending_command = _pending_command(history)
    pending = _pending_confirm_destination(history, destinations)
    if pending is not None and draft.intent == "affirm":
        # 텍스트 지름길과 같은 확정 — 들린 말이 어휘 목록에 없어도 모델이 긍정으로 들었으면 믿는다.
        return VicaIntent(
            intent="navigate", destination_candidate=pending.name,
            matched_destination_id=pending.id, confidence=1.0,
            reply=f"{pending.name} 안내를 시작합니다.", need_confirm=False, safety_flag="normal",
        ), heard, result.latency_sec
    if pending is not None and draft.intent == "deny":
        return VicaIntent(intent="deny", confidence=1.0, reply="", need_confirm=False), heard, result.latency_sec

    intent = _finalize(draft, destinations, pending=pending,
                       pending_command=pending_command, user_text=heard)
    return intent, heard, result.latency_sec
```

- [ ] **Step 4: 통과 확인 + 회귀**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 새 시험 9개 포함 전부 통과. 기존 파서 시험(`test_affirm_deny`·`test_command_intents`·`test_intent_confirm_rule` 등)이 지름길 분리 뒤에도 그대로 통과해야 한다 — 동작 불변의 증거.

- [ ] **Step 5: 커밋**

```bash
git add src/langchain_intent_parser.py tests/test_parser_audio.py
git commit -m "feat(parser): parse_intent_audio — 소리→Realtime→_finalize, 지름길은 _shortcut_intent 로 분리"
```

---

### Task 3: 노드 연결 — 소리 발행·audio 모드·A/B 로그

**Files:**
- Modify: `src/wakeword_monitor.py` (생성자 인자 + 청취 창 전사 직전 1곳 ≈ 753~755행)
- Modify: `src/ros_wakeword_node.py` (발행자·콜백)
- Modify: `src/ros_node.py` (구독·모드·공통 후처리·그림자 비교)
- Modify: `.env.example`
- Test: `tests/test_realtime_intent.py` 에 `pcm16_from_audio_msg` 시험 1개 추가

**Interfaces:**
- Consumes: Task 1 `float32_to_pcm16`, Task 2 `parse_intent_audio`
- Produces:
  - `WakewordMonitor(..., on_user_audio: Optional[Callable[[np.ndarray], None]] = None)` — 청취 창 전사 직전(잡음·짧은 클립 관문 통과 뒤)에 클립을 넘긴다
  - 토픽 `/vica/user_audio` (UInt8MultiArray, `layout.dim[0].label="pcm16_mono_16000"`)
  - `src/realtime_intent.py` 에 `def pcm16_from_audio_msg(data, label: str) -> bytes` (label 검사 후 bytes)
  - LLM 노드: `VICA_INTENT_INPUT` 읽기, `_on_user_audio`, `_publish_intent(intent, text)`(공통 후처리), `_shadow_text(text)`

- [ ] **Step 1: 모니터 콜백**

`src/wakeword_monitor.py`: 생성자에 `on_user_audio: Optional[Callable[[np.ndarray], None]] = None` 추가하고 `self._on_user_audio = on_user_audio` 로 보관. 청취 창에서 `stt_started = time.monotonic()` 바로 앞(≈ 754행, `transcribe = ...` 결정 뒤)에:

```python
        if self._on_user_audio is not None:
            # 소리→의도 직행(audio 모드): whisper 보다 먼저 클립을 넘긴다. 실패해도 전사는 계속.
            try:
                self._on_user_audio(audio)
            except Exception:
                pass
```

`_verify_emergency`(긴급 검증) 경로에는 넣지 않는다.

- [ ] **Step 2: 웨이크워드 노드**

`src/ros_wakeword_node.py`: import 에 `from std_msgs.msg import UInt8MultiArray, MultiArrayDimension` 와 `from .realtime_intent import float32_to_pcm16` 추가(기존 `Bool, Float32, String` import 옆). `__init__` 에서 `WakewordMonitor(...)` 를 만들기 전에:

```python
        # 소리→의도 직행(audio 모드, 2026-09-20 실험): 청취 클립을 LLM 노드로 보낸다.
        # text 모드(기본)에서는 아무것도 발행하지 않는다 — 지금과 동일.
        self._intent_input = os.environ.get("VICA_INTENT_INPUT", "text").strip().lower()
        self._pub_audio = self.create_publisher(UInt8MultiArray, "/vica/user_audio", 10)
```

`WakewordMonitor(...)` 호출에 `on_user_audio=self._on_user_audio if self._intent_input == "audio" else None,` 를 추가. 메서드:

```python
    def _on_user_audio(self, audio) -> None:
        pcm = float32_to_pcm16(audio)
        msg = UInt8MultiArray()
        msg.layout.dim.append(MultiArrayDimension(label="pcm16_mono_16000", size=len(pcm), stride=len(pcm)))
        msg.data = list(pcm)
        self._pub_audio.publish(msg)
        self.get_logger().info(f"🎧 발화 소리 -> /vica/user_audio ({len(pcm)/2/16000:.2f}s)")
```

- [ ] **Step 3: 헬퍼 + 시험**

`src/realtime_intent.py` 끝에:

```python
AUDIO_MSG_LABEL = "pcm16_mono_16000"


def pcm16_from_audio_msg(data, label: str) -> bytes:
    """/vica/user_audio 메시지의 data 와 layout.dim[0].label → pcm16 bytes. 라벨이 다르면 ValueError."""
    if label != AUDIO_MSG_LABEL:
        raise ValueError(f"오디오 라벨이 {AUDIO_MSG_LABEL} 이 아니다: {label!r}")
    return bytes(data)
```

`tests/test_realtime_intent.py` 끝에:

```python
from src.realtime_intent import AUDIO_MSG_LABEL, pcm16_from_audio_msg


def test_pcm16_from_audio_msg_roundtrip_and_label():
    assert pcm16_from_audio_msg([1, 2, 3], AUDIO_MSG_LABEL) == b"\x01\x02\x03"
    with pytest.raises(ValueError):
        pcm16_from_audio_msg([1], "float32_48000")
```

- [ ] **Step 4: LLM 노드**

`src/ros_node.py`:

(a) import: `from std_msgs.msg import Bool, String, UInt8MultiArray`; `from .langchain_intent_parser import (SHORTCUT_REPLIES, get_backend_manager, is_instant_utterance, parse_intent, parse_intent_audio)`; `from .llm_backend import BackendState, parse_goal_event`(이미 있음); `from .realtime_intent import pcm16_from_audio_msg`.

(b) `__init__` (백엔드 스레드 시작 앞):

```python
        # ----- 소리→의도 직행 (audio 모드, 2026-09-20 실험) ------------------
        # text(기본): /vica/user_text 로 지금처럼. audio: /vica/user_audio 를 Realtime 에
        # 보내 의도를 받고, 같은 발화의 텍스트 경로 결과는 로그([A/B])로만 남긴다.
        self._intent_input = os.environ.get("VICA_INTENT_INPUT", "text").strip().lower()
        self._audio_turn: dict = {}   # 직전 소리 발화의 결과(그림자 비교·실패 시 텍스트 인계용)
        self.create_subscription(UInt8MultiArray, "/vica/user_audio", self._on_user_audio, 10)
        self.get_logger().info(f"의도 입력 모드: {self._intent_input}")
```

(`import os` 가 없으면 추가.)

(c) `_on_user_text` 를 두 부분으로 나눈다. 긴급 블록과 `thinking`·`parse_intent` 호출까지는 그대로 두되, 파서 호출 **앞**에 audio 모드 분기를 넣는다:

```python
        else:
            if self._intent_input == "audio" and self._audio_turn.get("handled"):
                # 이 발화는 소리 경로가 이미 처리했다. 텍스트 경로 결과는 비교 로그로만.
                self._shadow_text(text)
                return
            # (audio 모드인데 소리 경로가 실패했거나 소리가 오지 않았으면 여기로 내려와 지금처럼 처리한다)
            thinking = not is_instant_utterance(text)
            ...  # 기존 parse_intent 호출 그대로
        self._publish_intent(intent, text)
```

파서 호출 뒤의 공통 후처리(재청취 기각 → should_forward → tts → listen → 로그 → history)는 통째로 `_publish_intent(self, intent, text)` 메서드로 옮긴다(내용 불변).

(d) 새 메서드:

```python
    def _on_user_audio(self, msg: UInt8MultiArray) -> None:
        """audio 모드: 소리를 Realtime 에 보내 의도를 받는다. 실패하면 뒤따라 오는 텍스트가 처리한다."""
        if self._intent_input != "audio":
            return
        self._audio_turn = {"handled": False, "t": time.time()}
        if self._backend.state is BackendState.LOCAL:
            self.get_logger().info("[A/B] 로컬 상태 — 소리 경로 생략, 텍스트 경로가 처리")
            return
        try:
            label = msg.layout.dim[0].label if msg.layout.dim else ""
            pcm = pcm16_from_audio_msg(msg.data, label)
        except ValueError as exc:
            self.get_logger().warning(f"[A/B] 소리 메시지 무시: {exc}")
            return
        self._reload_destinations_if_changed()
        if self._history.begin_turn(time.time()):
            self.get_logger().info("대화가 끊겨 이전 맥락을 비웠다")
        self._thinking_pub.publish(Bool(data=True))
        try:
            intent, heard, dt = parse_intent_audio(
                pcm, self._destinations, history=self._history.messages, robot_state=self._robot_state)
        except Exception as exc:
            self.get_logger().warning(f"[A/B] 소리 경로 실패({type(exc).__name__}: {exc}) — 텍스트 경로가 처리")
            return
        finally:
            self._thinking_pub.publish(Bool(data=False))
        self._audio_turn.update(handled=True, intent=intent, heard=heard, dt=dt)
        self._publish_intent(intent, heard)

    def _shadow_text(self, text: str) -> None:
        """audio 모드에서 같은 발화의 텍스트 경로 결과를 로그로만 남긴다(발행 안 함)."""
        turn = dict(self._audio_turn)
        history = self._history.messages
        robot_state = self._robot_state

        def work():
            started = time.monotonic()
            try:
                shadow = parse_intent(text, self._destinations, history=history, robot_state=robot_state)
                text_part = f"{shadow.intent}/{shadow.matched_destination_id or '-'} {time.monotonic() - started:.2f}s"
            except Exception as exc:
                text_part = f"실패({type(exc).__name__}) {time.monotonic() - started:.2f}s"
            a = turn.get("intent")
            audio_part = (f"{a.intent}/{a.matched_destination_id or '-'} {turn.get('dt', 0.0):.2f}s"
                          if a is not None else "?")
            self.get_logger().info(
                f"[A/B] audio={audio_part} | text={text_part} | heard='{turn.get('heard', '')}' | whisper='{text}'")

        threading.Thread(target=work, daemon=True, name="ab-shadow").start()
```

주의: 그림자 계산은 발행하지 않고 history 도 건드리지 않는다(스냅샷을 넘긴다). `_publish_intent` 안의 history 갱신은 소리 경로가 이미 했다.

- [ ] **Step 5: `.env.example`**

폴백 블록 뒤에 추가:

```bash
# ===== (실험) 소리→의도 직행 — Realtime 이 발화 소리에서 바로 의도를 정한다 (2026-09-20) =====
# VICA_INTENT_INPUT=audio              # text(기본)=지금 방식. audio=Realtime 직행 + 텍스트 경로는 [A/B] 로그
# VICA_REALTIME_MODEL=gpt-realtime-2.1-mini
# VICA_REALTIME_TIMEOUT=8              # 초. 넘기면 그 발화는 텍스트 경로가 처리
```

- [ ] **Step 6: 검사**

Run: `.venv/bin/python -m pytest tests/ -q` (전부 통과) ; `.venv/bin/python -m py_compile src/ros_node.py src/ros_wakeword_node.py src/wakeword_monitor.py && echo compiled` ; ROS 환경 import: `bash -c 'source /opt/ros/humble/setup.bash && source /home/ji_w/VICA-smarthandle/vica_ros2_ws/install/setup.bash && .venv/bin/python -c "import src.ros_node, src.ros_wakeword_node; print(\"import ok\")"'`.

- [ ] **Step 7: 커밋**

```bash
git add src/wakeword_monitor.py src/ros_wakeword_node.py src/ros_node.py src/realtime_intent.py tests/test_realtime_intent.py .env.example
git commit -m "feat(voice): audio 모드 — 발화 소리를 /vica/user_audio 로 보내 Realtime 의도, 텍스트 경로는 [A/B] 로그"
```

---

### Task 4: 실호출 스모크(수동) — supertonic 합성 문장으로 종단 확인

**Files:**
- Create: `tools/realtime_intent_smoke.py`

- [ ] **Step 1: 스크립트**

```python
#!/usr/bin/env python3
"""소리→의도 직행 종단 스모크(실제 OpenAI 호출, 수동 실행 전용).

로봇 TTS(supertonic)로 시험 문장을 합성해 parse_intent_audio 에 넣는다 — 마이크가
없어도 도는 종단 점검이다. 사용: .venv/bin/python tools/realtime_intent_smoke.py [문장...]
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.destination_loader import load_destinations  # noqa: E402
from src.langchain_intent_parser import parse_intent_audio  # noqa: E402
from src.realtime_intent import float32_to_pcm16  # noqa: E402
from src.tts import Tts  # noqa: E402  (클래스 이름은 src/tts.py 에서 확인)

DEFAULT = ["화장실로 안내해줘", "오 분", "한 시간", "그래", "아니요", "여기 몇 층이야", "테스트3으로 가자"]


def synth(tts, text: str) -> bytes:
    audio, rate = tts.synthesize(text)   # src/tts.py 의 실제 API 이름으로 맞춘다
    audio = np.asarray(audio, dtype=np.float32)
    if rate != 16000:
        x = np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / rate))
        audio = np.interp(x, np.arange(len(audio)), audio).astype(np.float32)
    return float32_to_pcm16(audio)


def main() -> None:
    dests = load_destinations()
    tts = Tts()
    for text in sys.argv[1:] or DEFAULT:
        pcm = synth(tts, text)
        t0 = time.monotonic()
        try:
            intent, heard, dt = parse_intent_audio(pcm, dests)
            print(f"[{time.monotonic()-t0:5.2f}s rt={dt:4.2f}s] '{text}' → heard='{heard}' "
                  f"intent={intent.intent} dest={intent.matched_destination_id or '-'} reply='{intent.reply[:30]}'")
        except Exception as exc:
            print(f"[실패] '{text}': {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
```

`src/tts.py` 의 합성 함수 이름·반환(오디오 배열, 샘플레이트)은 파일을 읽고 맞춘다.

- [ ] **Step 2: 실행(실제 API, 승인됨)**

Run: `.venv/bin/python tools/realtime_intent_smoke.py`
Expected: 7문장 모두 `intent` 가 상식과 맞고("오 분"→wait 5, "그래"→affirm, "여기 몇 층이야"→question), 각 1~2초. 결과를 보고서에 붙인다.

- [ ] **Step 3: 커밋**

```bash
git add tools/realtime_intent_smoke.py
git commit -m "chore(tools): Realtime 소리→의도 종단 스모크(supertonic 합성 문장)"
```

---

### Task 5: 실주행 A/B (사용자와 함께)

- `.env` 에 `VICA_INTENT_INPUT=audio` → ⑫ llm+tts 재기동 → 평소처럼 주행·대화.
- 노드 로그(`~/.ros/log/python_<pid>_*.log`)의 `[A/B]` 줄로 발화별 비교표를 만든다: 소리 경로 의도·지연, 텍스트 경로 의도·지연, 들린 말 vs whisper 전사.
- 세는 것: 오전사(whisper vs heard), 의도 정답, 재청취 기각 건수, 지연.
- 필요하면 `VICA_INTENT_INPUT=text` 로 한 번 더.

## Self-Review

- 스펙(가이드 §4 방식 2) 대응: 소리 발행(T3), out-of-band 함수 호출(T1), `_finalize` 재사용(T2), 텍스트 폴백·로컬 상태 우회(T3), A/B 로그(T3), 긴급어 불변(T3 Step 1 주의), 스모크(T4).
- 타입 일관성: `parse_intent_audio(pcm16_16k, destinations, history, robot_state) -> (VicaIntent, str, float)` 는 T2 정의·T3 노드·T4 스모크에서 동일. `float32_to_pcm16`·`pcm16_from_audio_msg`·`AUDIO_MSG_LABEL` 이름 동일. `RealtimeIntentClient.ask(pcm16_16k, history, instructions)` 는 T1·T2 가짜 클라이언트 동일.
- Placeholder: 없음. T4 의 TTS API 이름만 구현자가 파일에서 확인.

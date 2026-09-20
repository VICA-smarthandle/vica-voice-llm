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
        scripts = [
            [FakeEvent("error", error=types.SimpleNamespace(message="bad"))],
            [done_event({"intent": "unknown", "heard_text": ""})],
        ]
        conns = []

        def connect():
            conn = FakeConn(scripts[len(conns)])
            conns.append(conn)
            return conn

        client = RealtimeIntentClient(model="fake", timeout_sec=2.0, connect=connect)
        with pytest.raises(RuntimeError, match="bad"):
            client.ask(PCM, None, "x")
        assert conns[0].closed is True
        client.ask(PCM, None, "x")        # 다음 호출은 새 연결로 성공한다
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

    def test_session_update_failure_closes_connection(self):
        class BrokenSessionConn(FakeConn):
            def __init__(self, events):
                super().__init__(events)
                def boom(session):
                    raise RuntimeError("session rejected")
                self.session = types.SimpleNamespace(update=boom)

        conns = []

        def connect():
            conn = BrokenSessionConn([])
            conns.append(conn)
            return conn

        client = RealtimeIntentClient(model="fake", timeout_sec=1.0, connect=connect)
        with pytest.raises(RuntimeError, match="session rejected"):
            client.ask(PCM, None, "x")
        assert conns[0].closed is True
        assert client._conn is None


from src.realtime_intent import AUDIO_MSG_LABEL, pcm16_from_audio_msg


def test_pcm16_from_audio_msg_roundtrip_and_label():
    assert pcm16_from_audio_msg([1, 2, 3], AUDIO_MSG_LABEL) == b"\x01\x02\x03"
    with pytest.raises(ValueError):
        pcm16_from_audio_msg([1], "float32_48000")

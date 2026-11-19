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

    def test_int16_input_passes_through_unchanged(self):
        audio = np.array([0, 1000, -1000, 20000, -20000], dtype=np.int16)
        pcm = float32_to_pcm16(audio)
        ints = np.frombuffer(pcm, dtype="<i2")
        assert list(ints) == [0, 1000, -1000, 20000, -20000]

    def test_int32_input_is_clipped_to_int16_range(self):
        audio = np.array([40000, -40000], dtype=np.int32)
        pcm = float32_to_pcm16(audio)
        ints = np.frombuffer(pcm, dtype="<i2")
        assert list(ints) == [32767, -32768]

    def test_2d_mono_input_is_flattened(self):
        audio = np.array([[0.0], [0.5], [-0.5]], dtype=np.float32)
        pcm = float32_to_pcm16(audio)
        ints = np.frombuffer(pcm, dtype="<i2")
        assert len(ints) == 3
        assert list(ints) == [0, 16383, -16383]

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

    def test_failed_status_surfaces_error_code(self):
        # TPM 한도 초과: response.done 이 status=failed·output 빈 채로 온다 (2026-09-21 스모크 실측)
        err = types.SimpleNamespace(code="rate_limit_exceeded", type="tokens",
                                    message="Rate limit reached … Please try again in 864ms.")
        resp = types.SimpleNamespace(status="failed", status_details=types.SimpleNamespace(error=err),
                                     output=[], usage=None)
        with pytest.raises(RuntimeError, match="rate_limit_exceeded"):
            RealtimeIntentClient._parse_done(FakeEvent("response.done", response=resp))


class TestWarm:
    def test_warm_connects_once_and_ask_reuses_it(self):
        events = [done_event({"intent": "unknown", "heard_text": "a"})]
        client, conns = make_client(events)
        dt = client.warm()
        assert dt >= 0.0
        assert len(conns) == 1
        client.ask(PCM, None, "x")
        assert len(conns) == 1  # 예열 연결을 그대로 재사용


from src.realtime_intent import AUDIO_MSG_LABEL, audio_turn_applies, pcm16_from_audio_msg


def test_pcm16_from_audio_msg_roundtrip_and_label():
    assert pcm16_from_audio_msg([1, 2, 3], AUDIO_MSG_LABEL) == b"\x01\x02\x03"
    with pytest.raises(ValueError):
        pcm16_from_audio_msg([1], "float32_48000")


def test_audio_turn_applies_empty_turn_is_false():
    assert audio_turn_applies({}, 10.0) is False


def test_audio_turn_applies_not_handled_is_false():
    assert audio_turn_applies({"handled": False, "t": 9.0}, 10.0) is False


def test_audio_turn_applies_handled_and_fresh_is_true():
    assert audio_turn_applies({"handled": True, "t": 9.0}, 10.0) is True


def test_audio_turn_applies_handled_but_stale_is_false():
    assert audio_turn_applies({"handled": True, "t": 9.0}, 30.0) is False


def test_audio_turn_applies_handled_without_t_is_false():
    assert audio_turn_applies({"handled": True}, 10.0) is False


class TestRecentlyFailed:
    def test_failure_sets_and_success_clears(self):
        from src.realtime_intent import RealtimeIntentClient

        def bad_connect():
            raise ConnectionError("no wifi")

        c = RealtimeIntentClient(connect=bad_connect, timeout_sec=1)
        assert not c.recently_failed(30)
        try:
            c.ask(b"\x00\x00" * 160, None, "x")
        except ConnectionError:
            pass
        assert c.recently_failed(30)
        assert not c.recently_failed(0)
        c.last_failure_at = None
        assert not c.recently_failed(30)


def test_usage_dict_reports_cached_tokens():
    from types import SimpleNamespace
    from src.realtime_intent import _usage_dict
    usage = SimpleNamespace(output_tokens=7, input_token_details=SimpleNamespace(
        audio_tokens=20, text_tokens=3000, cached_tokens=2400))
    assert _usage_dict(usage) == {"audio_tokens": 20, "text_tokens": 3000, "cached_tokens": 2400, "output_tokens": 7}
    assert _usage_dict(None) == {}


# ----- 도구 출력(설계 2026-11-18 5.1) — VICA_TOOLS_AUDIO ------------------------
def fake_done(name: str, args: dict):
    item = types.SimpleNamespace(type="function_call", name=name, arguments=json.dumps(args))
    return FakeEvent("response.done", response=types.SimpleNamespace(output=[item], usage=None))


class TestTools:
    def test_off_keeps_the_forced_single_tool(self, monkeypatch):
        monkeypatch.delenv("VICA_TOOLS_AUDIO", raising=False)
        cfg = RealtimeIntentClient(connect=lambda: None)._session_config()
        assert [t["name"] for t in cfg["tools"]] == ["set_intent"]
        assert cfg["tool_choice"] == {"type": "function", "name": "set_intent"}
        assert cfg["tools"][0] == build_intent_tool()

    def test_on_offers_both_and_requires_one(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VICA_TOOLS_AUDIO", "on")
        monkeypatch.setenv("VICA_RESERVATION_DIR", str(tmp_path))
        cfg = RealtimeIntentClient(connect=lambda: None)._session_config()
        assert [t["name"] for t in cfg["tools"]] == ["set_intent", "reserve_facility"]
        assert cfg["tool_choice"] == "required" and "heard_text" in cfg["tools"][1]["parameters"]["properties"]
        assert cfg["tools"][1]["type"] == "function" and "function" not in cfg["tools"][1]     # Realtime 은 평평한 꼴
        assert cfg["tools"][1]["parameters"]["properties"]["facility"]["enum"] == ["세미나실", "상담실"]
        # 도구가 둘이면 set_intent 설명의 "반드시 이 함수로만 답한다"는 맞지 않는다
        assert "반드시 이 함수로만" not in cfg["tools"][0]["description"] and "예약" in cfg["tools"][0]["description"]
        assert cfg["tools"][0]["parameters"] == build_intent_tool()["parameters"]

    def test_parse_done_names_the_tool(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VICA_TOOLS_AUDIO", "on")
        monkeypatch.setenv("VICA_RESERVATION_DIR", str(tmp_path))
        done = fake_done("reserve_facility", {"facility": "세미나실", "date": "내일", "heard_text": "내일 세미나실 예약해 줘"})
        name, args, _usage = RealtimeIntentClient._parse_done(done)
        assert name == "reserve_facility" and args["facility"] == "세미나실"
        with pytest.raises(ValueError):
            RealtimeIntentClient._parse_done(fake_done("go_to", {"location": "화장실"}))

    def test_off_does_not_accept_the_reserve_tool(self, monkeypatch):
        monkeypatch.delenv("VICA_TOOLS_AUDIO", raising=False)
        with pytest.raises(ValueError):
            RealtimeIntentClient._parse_done(fake_done("reserve_facility", {"facility": "세미나실"}))
        name, args, _usage = RealtimeIntentClient._parse_done(fake_done("set_intent", {"intent": "unknown"}))
        assert name == "set_intent" and args == {"intent": "unknown"}

    def test_ask_carries_the_tool_name(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VICA_TOOLS_AUDIO", "on")
        monkeypatch.setenv("VICA_RESERVATION_DIR", str(tmp_path))
        client, conns = make_client([fake_done("reserve_facility", {"name": "김민수", "heard_text": "김민수"})])
        result = client.ask(PCM, None, "지시문")
        assert result.tool == "reserve_facility" and result.draft == {"name": "김민수"} and result.heard_text == "김민수"
        assert conns[0].responses[0]["tool_choice"] == "required"

    def test_result_tool_defaults_to_set_intent(self):
        client, conns = make_client([done_event({"intent": "unknown", "heard_text": ""})])
        assert client.ask(PCM, None, "x").tool == "set_intent"
        assert RealtimeResult(draft={}, heard_text="", latency_sec=0.0).tool == "set_intent"

    def test_broken_facility_file_keeps_the_single_tool(self, monkeypatch, tmp_path):
        """시설 목록이 비면 빈 enum 을 보내지 않는다 — 꺼진 것과 같은 설정으로 돈다."""
        (tmp_path / "facilities.yaml").write_text("facilities: [oops", encoding="utf-8")
        monkeypatch.setenv("VICA_TOOLS_AUDIO", "on")
        monkeypatch.setenv("VICA_RESERVATION_DIR", str(tmp_path))
        cfg = RealtimeIntentClient(connect=lambda: None)._session_config()
        assert [t["name"] for t in cfg["tools"]] == ["set_intent"]
        assert cfg["tool_choice"] == {"type": "function", "name": "set_intent"}

    def test_ask_text_sends_words_instead_of_sound(self):
        """재생용: 소리 대신 글자 항목을 넣는다. 들린 말이 비어 오면 넣은 글자를 쓴다."""
        client, conns = make_client([done_event({"intent": "navigate", "destination_candidate": "화장실"})])
        result = client.ask_text("화장실로 가자.", [AIMessage("네?")], "지시문")
        req = conns[0].responses[0]
        assert req["input"][-1] == {"type": "message", "role": "user",
                                    "content": [{"type": "input_text", "text": "화장실로 가자."}]}
        assert req["input"][0]["role"] == "assistant" and req["instructions"] == "지시문"
        assert result.draft["intent"] == "navigate" and result.heard_text == "화장실로 가자."


def test_audio_tools_active_follows_the_session_tools(monkeypatch, tmp_path):
    from src.realtime_intent import audio_tools_active
    monkeypatch.setenv("VICA_RESERVATION_DIR", str(tmp_path))
    monkeypatch.delenv("VICA_TOOLS_AUDIO", raising=False)
    assert not audio_tools_active()
    monkeypatch.setenv("VICA_TOOLS_AUDIO", "on")
    assert audio_tools_active()
    (tmp_path / "facilities.yaml").write_text("[]", encoding="utf-8")
    assert not audio_tools_active()


def test_tool_has_no_answer_when_off():
    from src.realtime_intent import build_intent_tool
    assert "answer" not in build_intent_tool()["parameters"]["properties"]


def test_tool_has_the_answer_field_when_on(monkeypatch):
    from src import answer_field as af
    from src.realtime_intent import build_intent_tool
    monkeypatch.setenv("VICA_ANSWER", "on")
    tool = build_intent_tool()
    assert tool["parameters"]["properties"]["answer"] == {
        "type": "string", "enum": ["", *af.ANSWERS], "description": af.ANSWER_DESCRIPTION}
    assert tool["parameters"]["required"] == ["intent", "heard_text"]


# ----- 예약 도구 호출 기록(11-19 원인 실험) ------------------------------------------
TRAIL = [{"args": {"facility": "세미나실", "date": "내일", "start_time": "14:00"},
          "heard": "내일 오후 두 시에 세미나실 예약해 줘.", "asked": "성함을 말씀해 주세요."}]


def test_history_items_carry_the_reserve_call_before_its_question():
    """Realtime 은 대화 기록에 예약 도구를 부른 흔적이 없으면 승낙에 예약 도구를 다시 부른다(V0 5/8 → V4 8/8)."""
    history = [HumanMessage("내일 오후 두 시에 세미나실 예약해 줘."), AIMessage("성함을 말씀해 주세요.")]
    items = history_to_items(history, TRAIL)
    assert [i["type"] for i in items] == ["message", "function_call", "function_call_output", "message"]
    call, output = items[1], items[2]
    assert call["name"] == "reserve_facility" and call["call_id"] == output["call_id"]
    assert json.loads(call["arguments"]) == {"facility": "세미나실", "date": "내일", "start_time": "14:00",
                                             "heard_text": "내일 오후 두 시에 세미나실 예약해 줘."}
    assert json.loads(output["output"]) == {"status": "asked", "question": "성함을 말씀해 주세요."}
    assert items[3]["role"] == "assistant"
    assert history_to_items(history) == history_to_items(history, None) == [items[0], items[3]]


def test_trail_without_its_question_in_history_goes_last():
    """소리 모드에서 질문 문장이 아직 기록에 안 들어왔으면(미션이 말하는 중) 맨 끝에 둔다."""
    items = history_to_items([HumanMessage("내일 오후 두 시에 세미나실 예약해 줘.")], TRAIL)
    assert [i["type"] for i in items] == ["message", "function_call", "function_call_output"]


def test_ask_sends_the_trail_only_when_the_session_has_the_reserve_tool(monkeypatch, tmp_path):
    monkeypatch.setenv("VICA_RESERVATION_DIR", str(tmp_path))
    monkeypatch.setenv("VICA_TOOLS_AUDIO", "on")
    client, conns = make_client([done_event({"intent": "affirm", "heard_text": "그래"})] * 2)
    client.ask(PCM, [AIMessage("성함을 말씀해 주세요.")], "x", tool_trail=TRAIL)
    assert any(i["type"] == "function_call" for i in conns[0].responses[0]["input"])
    monkeypatch.delenv("VICA_TOOLS_AUDIO")
    client.ask(PCM, [AIMessage("성함을 말씀해 주세요.")], "x", tool_trail=TRAIL)
    assert not any(i["type"] == "function_call" for i in conns[0].responses[1]["input"])

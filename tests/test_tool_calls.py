"""도구 정의(설계 2026-11-18 5.1)."""
from src import tool_calls
from src.reservation_flow import ReserveCall


def test_reserve_tool_shape():
    tool = tool_calls.reserve_tool(["세미나실", "상담실"])["function"]
    assert tool["name"] == "reserve_facility" and "이동" in tool["description"] and "진행 중인 예약" in tool["description"]
    props = tool["parameters"]["properties"]
    assert set(props) == {"facility", "date", "start_time", "end_time", "name"}
    assert props["facility"]["enum"] == ["세미나실", "상담실"] and tool["parameters"]["required"] == []
    heard = tool_calls.reserve_tool(["세미나실"], with_heard_text=True)["function"]["parameters"]
    assert "heard_text" in heard["properties"] and heard["required"] == ["heard_text"]


def test_reserve_call_keeps_only_known_text_values():
    call = tool_calls.reserve_call({"facility": "세미나실", "date": "내일", "start_time": "14:00", "name": None, "junk": 1})
    assert call == ReserveCall(facility="세미나실", date="내일", start_time="14:00")


def test_switches_default_off(monkeypatch):
    monkeypatch.delenv("VICA_TOOLS_TEXT", raising=False)
    monkeypatch.delenv("VICA_TOOLS_AUDIO", raising=False)
    assert not tool_calls.text_tools_on() and not tool_calls.audio_tools_on()
    monkeypatch.setenv("VICA_TOOLS_TEXT", "on")
    assert tool_calls.text_tools_on() and not tool_calls.audio_tools_on()


def test_any_tools_on_reads_the_env_file_for_processes_without_dotenv(monkeypatch, tmp_path):
    """TTS 노드는 .env 를 읽지 않는다 — 환경 변수에 없으면 저장소 .env 에서 두 스위치만 읽는다(검토 M-1)."""
    import os

    env_file = tmp_path / ".env"
    env_file.write_text("VICA_TOOLS_AUDIO=on\nVICA_MADE_UP_KEY=secret\n", encoding="utf-8")
    monkeypatch.setattr(tool_calls, "ENV_FILE", env_file)
    monkeypatch.delenv("VICA_TOOLS_TEXT", raising=False)
    monkeypatch.delenv("VICA_TOOLS_AUDIO", raising=False)
    assert tool_calls.any_tools_on()
    assert "VICA_MADE_UP_KEY" not in os.environ and "VICA_TOOLS_AUDIO" not in os.environ   # 환경에 아무것도 싣지 않는다
    monkeypatch.setenv("VICA_TOOLS_AUDIO", "off")                # 환경 변수가 있으면 그 값이 이긴다(load_dotenv 와 같은 순서)
    assert not tool_calls.any_tools_on()
    monkeypatch.delenv("VICA_TOOLS_AUDIO")
    monkeypatch.setattr(tool_calls, "ENV_FILE", tmp_path / "없는파일")
    assert not tool_calls.any_tools_on()

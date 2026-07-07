"""EmergencyMonitor 의 순수 로직(process_window) 검증. 마이크/STT 불필요."""
from __future__ import annotations

import numpy as np

from src.emergency_monitor import EmergencyMonitor
from src.schema import EmergencyEvent

LOUD = np.full(16000, 0.1, dtype=np.float32)  # 음량 게이트를 넘는 1초 오디오
QUIET = np.full(16000, 0.001, dtype=np.float32)  # 게이트에 걸리는 조용한 오디오


def make_monitor(text: str, events: list[EmergencyEvent]) -> EmergencyMonitor:
    """항상 text 로 인식하는 가짜 STT 를 쓰는 모니터를 만든다."""
    return EmergencyMonitor(on_event=events.append, transcribe=lambda _audio: text)


def test_emergency_keyword_fires_event():
    events: list[EmergencyEvent] = []
    monitor = make_monitor("멈춰 주세요", events)
    event = monitor.process_window(LOUD, now=100.0)
    assert event is not None
    assert event.keyword == "멈춰"
    assert event.source_text == "멈춰 주세요"
    assert events == [event]


def test_normal_speech_no_event():
    events: list[EmergencyEvent] = []
    monitor = make_monitor("식당에 데려가 주세요", events)
    assert monitor.process_window(LOUD, now=100.0) is None
    assert events == []


def test_quiet_audio_skips_stt():
    events: list[EmergencyEvent] = []
    calls = []

    def fake_stt(_audio):
        calls.append(1)
        return "멈춰"

    monitor = EmergencyMonitor(on_event=events.append, transcribe=fake_stt)
    assert monitor.process_window(QUIET, now=100.0) is None
    assert calls == []  # 조용하면 STT 자체를 부르지 않는다
    assert events == []


def test_empty_audio_no_event():
    events: list[EmergencyEvent] = []
    monitor = make_monitor("멈춰", events)
    assert monitor.process_window(np.zeros(0, dtype=np.float32), now=100.0) is None


def test_cooldown_suppresses_duplicate():
    events: list[EmergencyEvent] = []
    monitor = make_monitor("멈춰", events)
    assert monitor.process_window(LOUD, now=100.0) is not None
    assert monitor.process_window(LOUD, now=101.0) is None  # 쿨다운(2초) 이내
    assert monitor.process_window(LOUD, now=103.0) is not None  # 쿨다운 지남
    assert len(events) == 2

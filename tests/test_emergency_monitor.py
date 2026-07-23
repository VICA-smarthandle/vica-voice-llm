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


# ---- TTS 재생 중 감시 억제 (자가 트리거 차단) --------------------------------


def test_muted_skips_detection():
    """재생 중에는 긴급어가 들려도 이벤트를 내지 않는다."""
    events: list[EmergencyEvent] = []
    monitor = make_monitor("멈춰", events)
    monitor.set_muted(True, now=100.0)
    assert monitor.process_window(LOUD, now=100.5) is None
    assert events == []


def test_muted_skips_stt_entirely():
    """재생 중에는 STT 자체를 부르지 않는다 (GPU 낭비 방지)."""
    events: list[EmergencyEvent] = []
    calls = []

    def fake_stt(_audio):
        calls.append(1)
        return "멈춰"

    monitor = EmergencyMonitor(on_event=events.append, transcribe=fake_stt)
    monitor.set_muted(True, now=100.0)
    monitor.process_window(LOUD, now=100.5)
    assert calls == []


def test_unmute_holds_until_buffer_refills():
    """해제 직후에는 버퍼에 로봇 목소리가 남아 있으므로 창 길이만큼 판정을 보류한다."""
    events: list[EmergencyEvent] = []
    monitor = make_monitor("멈춰", events)  # window_sec 기본 2.0
    monitor.set_muted(True, now=100.0)
    monitor.set_muted(False, now=101.0)

    assert monitor.process_window(LOUD, now=101.5) is None  # 아직 보류 중
    assert monitor.process_window(LOUD, now=102.9) is None  # 여전히 보류 중
    assert monitor.process_window(LOUD, now=103.1) is not None  # 창이 다시 찬 뒤
    assert len(events) == 1


def test_unmute_clears_buffer():
    """해제 시 담아둔 오디오(로봇 목소리)를 버린다."""
    monitor = make_monitor("멈춰", [])
    with monitor._buffer_lock:
        monitor._buffer = LOUD.copy()
    monitor.set_muted(True, now=100.0)
    monitor.set_muted(False, now=101.0)
    assert float(np.abs(monitor._buffer).max()) == 0.0


def test_mute_expires_as_failsafe():
    """TTS 노드가 mute 를 켠 채 죽어도 감시가 영구히 멈추지 않는다."""
    events: list[EmergencyEvent] = []
    monitor = make_monitor("멈춰", events)  # max_mute_sec 기본 8.0
    monitor.set_muted(True, now=100.0)

    assert monitor.is_muted(now=107.0) is True
    assert monitor.is_muted(now=108.5) is False  # 스스로 해제

    # 해제 뒤에도 보류 구간은 지킨다.
    assert monitor.process_window(LOUD, now=109.0) is None
    assert monitor.process_window(LOUD, now=111.0) is not None


def test_repeated_mute_does_not_extend_failsafe():
    """재생 중 반복 발행돼도 fail-safe 기준 시각은 처음 켠 때로 유지된다."""
    monitor = make_monitor("멈춰", [])
    monitor.set_muted(True, now=100.0)
    monitor.set_muted(True, now=104.0)
    assert monitor.is_muted(now=108.5) is False


def test_unmute_when_not_muted_is_noop():
    """mute 가 아닌 상태에서 해제 신호가 와도 판정을 보류하지 않는다."""
    events: list[EmergencyEvent] = []
    monitor = make_monitor("멈춰", events)
    monitor.set_muted(False, now=100.0)
    assert monitor.process_window(LOUD, now=100.1) is not None

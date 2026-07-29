"""SpanTracker 순수 로직 검증 — 이벤트 흐름 → 단계별 시간."""
from __future__ import annotations

from src.metrics import SpanTracker


def test_full_interaction_spans():
    tr = SpanTracker()
    tr.feed("wake", 10.0)
    tr.feed("user_text", 12.5, "화장실 어디야")
    tr.feed("intent", 16.0)
    tr.feed("tts_start", 16.3)
    tr.feed("tts_end", 18.3)
    assert len(tr.interactions) == 1
    it = tr.interactions[0]
    assert it["listen_stt_sec"] == 2.5
    assert it["llm_sec"] == 3.5
    assert it["response_sec"] == 6.3       # 체감: 부르고 → 말 시작
    assert it["tts_play_sec"] == 2.0
    assert it["text"] == "화장실 어디야"


def test_wake_silent_closed_by_next_wake():
    tr = SpanTracker()
    tr.feed("wake", 10.0)                   # 아무 말 없었음
    tr.feed("wake", 20.0)
    tr.feed("user_text", 21.0, "안녕")
    tr.finalize(30.0)
    assert len(tr.interactions) == 2
    assert tr.interactions[0].get("wake_silent") is True
    assert tr.interactions[1]["listen_stt_sec"] == 1.0


def test_emergency_reaction_time():
    tr = SpanTracker()
    tr.feed("emergency", 5.0, "멈춰")
    tr.feed("sim_event", 5.12, "estopped")
    assert tr.emergencies == [{"t": 5.0, "keyword": "멈춰", "react_sec": 0.12}]


def test_sim_event_without_emergency_ignored():
    tr = SpanTracker()
    tr.feed("sim_event", 5.0, "estopped")   # reset 등 다른 이유의 상태 변화
    assert tr.emergencies == []

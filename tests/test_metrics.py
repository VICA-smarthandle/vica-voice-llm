"""SpanTracker 순수 로직 검증 — 이벤트 흐름 → 단계별 시간."""
from __future__ import annotations

from src.metrics import SpanTracker


def test_full_interaction_spans():
    tr = SpanTracker()
    tr.feed("wake", 10.0)
    tr.feed("user_text", 12.5, "화장실 어디야")
    tr.feed("intent", 16.0)
    tr.feed("tts_request", 16.1, "response:안내하겠습니다.")
    tr.feed("tts_start", 16.3)
    tr.feed("tts_end", 18.3)
    tr.finalize(20.0)
    assert len(tr.interactions) == 1
    it = tr.interactions[0]
    assert it["listen_stt_sec"] == 2.5
    assert it["llm_sec"] == 3.5
    assert it["tts_wait_sec"] == 0.2
    assert it["response_sec"] == 6.3
    assert it["tts_play_sec"] == 2.0
    assert it["text"] == "화장실 어디야"


def test_sentence_split_playback_counts_as_one():
    """TTS 가 문장 단위로 끊어 재생해도 한 번의 재생으로 묶인다."""
    tr = SpanTracker()
    tr.feed("wake", 10.0)
    tr.feed("user_text", 11.0, "화장실 어디야")
    tr.feed("intent", 12.0)
    tr.feed("tts_start", 12.2)
    tr.feed("tts_end", 13.0)
    tr.feed("tts_start", 13.4)
    tr.feed("tts_end", 14.5)
    tr.finalize(20.0)
    assert len(tr.interactions) == 1
    it = tr.interactions[0]
    assert it["response_sec"] == 2.2
    assert it["tts_play_sec"] == 2.3


def test_unrelated_later_playback_does_not_extend():
    """한참 뒤의 도착 안내는 앞 상호작용에 붙지 않는다."""
    tr = SpanTracker()
    tr.feed("wake", 10.0)
    tr.feed("user_text", 11.0, "화장실 데려다줘")
    tr.feed("intent", 12.0)
    tr.feed("tts_start", 12.2)
    tr.feed("tts_end", 13.0)
    tr.feed("tts_start", 40.0)
    tr.feed("tts_end", 42.0)
    tr.finalize(50.0)
    assert len(tr.interactions) == 1
    it = tr.interactions[0]
    assert it["closed_by"] == "tts_gap"
    assert it["tts_play_sec"] == 0.8


def test_wake_silent_closed_by_next_wake():
    tr = SpanTracker()
    tr.feed("wake", 10.0)
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
    tr.feed("sim_event", 5.0, "estopped")
    assert tr.emergencies == []

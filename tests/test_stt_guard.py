"""유령 STT 방어 검증 — 무음 환각이 사용자 발화로 둔갑하면 안 된다.

stt_guard 순수 함수와, 청취 창에서 환각 전사가 wake_silent 로 처리되는
monitor 통합을 본다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.stt_guard import accept_segments, is_hallucination
from src.wakeword_monitor import WakewordMonitor

LOUD = np.full(1280, 3000, dtype=np.int16)
QUIET = np.zeros(1280, dtype=np.int16)


@dataclass
class Seg:
    text: str
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0


# ---------------------------------------------------------------- 신뢰도 필터
def test_drops_segment_only_when_both_signals_say_silence():
    """no_speech 높음 + logprob 낮음이 '동시에' 참일 때만 버린다 —
    작게 말한 진짜 발화(낮은 확신 하나만)는 살아야 한다."""
    both_bad = Seg("시청해주셔서 감사합니다", no_speech_prob=0.9, avg_logprob=-1.5)
    quiet_real = Seg("화장실", no_speech_prob=0.3, avg_logprob=-1.5)   # 확신만 낮음
    noisy_real = Seg("가고 싶어요", no_speech_prob=0.9, avg_logprob=-0.4)  # 무음확률만 높음

    assert accept_segments([both_bad]) == ""
    assert accept_segments([quiet_real]) == "화장실"
    assert accept_segments([both_bad, quiet_real, noisy_real]) == "화장실가고 싶어요"


def test_accepts_plain_segments_and_stub_without_attrs():
    class Bare:
        text = "응"

    assert accept_segments([Seg("네 맞아요")]) == "네 맞아요"
    assert accept_segments([Bare()]) == "응"
    assert accept_segments([]) == ""


# ---------------------------------------------------------------- 수배 전단
def test_known_hallucinations_full_match():
    assert is_hallucination("시청해 주셔서 감사합니다.")
    assert is_hallucination("시청해주셔서 감사합니다")
    assert is_hallucination("구독과 좋아요 부탁드립니다!")
    assert is_hallucination("MBC 뉴스 김성현입니다.")


def test_real_speech_is_not_flagged():
    assert not is_hallucination("화장실 데려다줘")
    assert not is_hallucination("감사합니다")            # 합법 발화 — 지우면 안 됨
    assert not is_hallucination("안내해 주셔서 감사합니다")  # 부분 유사도 아님
    assert not is_hallucination("")


# ---------------------------------------------------------------- monitor 통합
def test_hallucinated_listen_window_closes_quietly():
    """소음이 RMS 문지기를 뚫고 whisper 가 환각을 지어내도, user_text 로
    LLM 에 가면 안 된다 — 조용히 닫힌다(wake_silent)."""
    texts = []
    m = WakewordMonitor(
        on_emergency=lambda e: None,
        on_user_text=texts.append,
        predict=lambda f: {"a": 0.0, "b": 0.0},
        transcribe=lambda a: "시청해 주셔서 감사합니다.",
    )
    m._open_listen(followup=False, now=0.0)

    results = []
    for i in range(5):
        # 칩이 발화로 오판한 소음 (사람 말 유사음 등)
        results.append(m.process_frame(LOUD, now=i * 0.08, vad=True))
    for i in range(11):
        results.append(m.process_frame(QUIET, now=0.4 + i * 0.08))

    assert results[-1] == "wake_silent"
    assert texts == []


def test_listen_prefers_guarded_transcribe_but_emergency_uses_raw():
    """전사 경로 분리: 대화 청취는 필터판(_transcribe_listen), 긴급 검증은
    무필터판(_transcribe)을 써야 한다 — 작은 외침 보존이 우선이다
    (외침 10회 실측에서 빈 전사 기각 1건)."""
    texts, events = [], []
    m = WakewordMonitor(
        on_emergency=events.append,
        on_user_text=texts.append,
        predict=lambda f: {"a": 0.0, "b": 0.0},
        transcribe=lambda a: "멈춰",          # 무필터판 (긴급 검증용)
    )
    m._transcribe_listen = lambda a: "화장실 가자"  # 필터판 (대화용)

    m._open_listen(followup=False, now=0.0)
    for i in range(5):
        m.process_frame(LOUD, now=i * 0.08, vad=True)
    for i in range(11):
        m.process_frame(QUIET, now=0.4 + i * 0.08)
    assert texts == ["화장실 가자"]           # 대화 경로가 필터판을 썼다

    m._enter_postroll()
    for i in range(4):
        m.process_frame(LOUD, now=2.0 + i * 0.08)
    assert len(events) == 1                   # 긴급 경로는 무필터판("멈춰")
    assert events[0].keyword == "멈춰"


# ---------------------------------------------------------------- 수음 계측
def test_capture_stats_numbers():
    from src.wakeword_monitor import capture_stats

    silence = np.zeros(1600, dtype=np.int16)
    s = capture_stats(silence)
    assert s == {"rms": 0.0, "peak": 0.0, "clip_ratio": 0.0}

    clipped = np.full(100, 32767, dtype=np.int16)
    s = capture_stats(clipped)
    assert s["peak"] > 0.99 and s["clip_ratio"] == 1.0

    normal = np.full(100, 3000, dtype=np.int16)
    s = capture_stats(normal)
    assert 0.05 < s["rms"] < 0.15 and s["clip_ratio"] == 0.0

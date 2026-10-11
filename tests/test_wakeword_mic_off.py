"""장애물 정지 안내를 말하는 동안 마이크를 닫는다(2026-10-10 사용자 결정).

"앞에 장애물이 있어 정지합니다."의 '정지'는 긴급어(모델 B)라, 귀가 로봇 자기 목소리로 비상정지를
걸 수 있다. TTS 가 그 문장 재생을 시작하면(/vica/tts_now) 마이크를 닫고, 재생 끝 소식(/vica/tts_state
False)에 다시 연다. 다른 말은 예전처럼 재생 중에도 귀가 열려 있다(AEC 모드).
가짜 predict/transcribe 주입 — test_wakeword_speaking 과 같은 패턴.
"""
from __future__ import annotations

import numpy as np

from src.mission_phrases import OBSTACLE_AVOID, OBSTACLE_SLOW, OBSTACLE_STOP
from src.wakeword_monitor import POST_ROLL_FRAMES, WakewordMonitor, mic_off_while_saying

LOUD = np.full(1280, 3000, dtype=np.int16)


class Fake:
    def __init__(self, scores=(), text=""):
        self.scores = list(scores)
        self.text = text

    def predict(self, _frame):
        a, b = self.scores.pop(0) if self.scores else (0.0, 0.0)
        return {"a": a, "b": b}

    def transcribe(self, _audio):
        return self.text


def make(fake: Fake, events: list):
    return WakewordMonitor(
        on_emergency=events.append,
        on_user_text=lambda _t: None,
        on_wake=lambda: None,
        predict=fake.predict,
        transcribe=fake.transcribe,
    )


def run_frames(m, n, t0):
    return [m.process_frame(LOUD, now=t0 + i * 0.08) for i in range(n)]


def start_speaking(m, text, now):
    """TTS 노드 순서 그대로: 문장 소식(tts_now) → 재생 시작(tts_state True)."""
    m.note_robot_speech(text, now=now)
    m.set_speaking(True, now=now)


def test_only_the_stop_sentence_closes_the_mic():
    assert mic_off_while_saying(OBSTACLE_STOP)
    assert mic_off_while_saying(" " + OBSTACLE_STOP + " ")
    for text in (OBSTACLE_AVOID, OBSTACLE_SLOW, "화장실로 안내해드릴까요?", "", None):
        assert not mic_off_while_saying(text)


def test_robot_saying_stop_does_not_trigger_emergency():
    # 로봇 자기 목소리 '정지'가 긴급 관문을 넘을 만큼 들려도 비상정지가 걸리지 않는다
    fake = Fake(scores=[(0, 0.9)] * 20, text="정지")
    events = []
    m = make(fake, events)
    start_speaking(m, OBSTACLE_STOP, now=0.0)
    results = run_frames(m, 2 + POST_ROLL_FRAMES, t0=0.1)
    assert "emergency" not in results and events == []


def test_mic_opens_again_when_that_speech_ends():
    # 닫힌 동안은 점수 계산부터 건너뛴다 — 가짜 점수는 다시 연 뒤부터 쓰인다
    fake = Fake(scores=[(0, 0.0)] * 2 + [(0, 0.9)] * 2, text="멈춰")
    events = []
    m = make(fake, events)
    start_speaking(m, OBSTACLE_STOP, now=0.0)
    run_frames(m, 4, t0=0.1)                       # 로봇 말 — 못 듣는다
    assert fake.scores and events == []
    m.set_speaking(False, now=2.4)                 # 재생 끝(꼬리 0.4초 포함)
    run_frames(m, 2, t0=2.5)                       # 다시 열림 — 조용한 두 장
    results = run_frames(m, 2 + POST_ROLL_FRAMES, t0=2.7)
    assert results[-1] == "emergency" and events[0].keyword == "멈춰"


def test_other_robot_speech_keeps_the_mic_open():
    # 비켜 감·감속 안내 중의 '멈춰'는 예전처럼 들린다
    fake = Fake(scores=[(0, 0.9), (0, 0.9)], text="멈춰")
    events = []
    m = make(fake, events)
    start_speaking(m, OBSTACLE_SLOW, now=0.0)
    results = run_frames(m, 2 + POST_ROLL_FRAMES, t0=0.1)
    assert results[-1] == "emergency" and len(events) == 1


def test_mute_mode_end_also_reopens():
    # VICA_TTS_MUTE=on(옛 뮤트 방식)이면 끝 소식이 set_muted(False) 로 온다
    m = make(Fake(), [])
    m.note_robot_speech(OBSTACLE_STOP, now=0.0)
    m.set_muted(True, now=0.0)
    m.set_muted(False, now=2.0)
    assert not m._is_muted(2.1) and not m._mic_off_speech


def test_lost_end_signal_reopens_by_failsafe():
    # 재생 끝 소식을 놓쳐도 영영 닫혀 있지 않는다(set_muted 의 fail-safe 10초)
    m = make(Fake(), [])
    m.note_robot_speech(OBSTACLE_STOP, now=0.0)
    assert m._is_muted(5.0)
    assert not m._is_muted(10.5) and not m._mic_off_speech

"""WakewordMonitor 의 순수 로직(process_frame) 검증. 마이크/모델/STT 불필요.

가짜 predict/transcribe 를 주입한다 (test_emergency_monitor 와 같은 패턴).
프레임은 80ms(1280샘플) int16.
"""
from __future__ import annotations

import numpy as np

from src.schema import EmergencyEvent
from src.wakeword_monitor import POST_ROLL_FRAMES, WakewordMonitor

LOUD = np.full(1280, 3000, dtype=np.int16)    # 발화 판정 RMS 를 넘는 프레임
QUIET = np.zeros(1280, dtype=np.int16)


class Fake:
    """프레임 순서대로 (a, b) 점수를 돌려주는 가짜 모델 + 고정 전사 STT."""

    def __init__(self, scores, text=""):
        self.scores = list(scores)
        self.text = text
        self.stt_calls = 0

    def predict(self, _frame):
        a, b = self.scores.pop(0) if self.scores else (0.0, 0.0)
        return {"a": a, "b": b}

    def transcribe(self, _audio):
        self.stt_calls += 1
        return self.text


def make(fake: Fake, events: list, texts: list, wakes: list):
    return WakewordMonitor(
        on_emergency=events.append,
        on_user_text=texts.append,
        on_wake=lambda: wakes.append(1),
        predict=fake.predict,
        transcribe=fake.transcribe,
    )


def run_frames(m, n, frame=QUIET, t0=0.0, vad=None):
    out = []
    for i in range(n):
        out.append(m.process_frame(frame, now=t0 + i * 0.08, vad=vad))
    return out


def test_emergency_confirmed():
    # B 점수 2연속 → 포스트롤 → 전사 "멈춰!" → 이벤트 발행
    fake = Fake(scores=[(0, 0.9), (0, 0.9)] + [(0, 0)] * 10, text="멈춰!")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD)
    assert results[-1] == "emergency"
    assert len(events) == 1
    assert events[0].keyword == "멈춰"
    assert events[0].source_text == "멈춰!"
    assert isinstance(events[0], EmergencyEvent)


def test_emergency_rejected_by_stt():
    # 관문은 뚫렸지만 전사가 "멈춤" — STT 층이 막는다 (실측에서 확인된 함정)
    fake = Fake(scores=[(0, 0.9), (0, 0.9)] + [(0, 0)] * 10, text="멈춤")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD)
    assert results[-1] == "reject"
    assert events == []


def test_single_spike_no_stt_call():
    # 1프레임 스파이크는 발동하지 않고 whisper 도 부르지 않는다
    fake = Fake(scores=[(0, 0.9)] + [(0, 0)] * 10, text="멈춰")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    run_frames(m, 8, LOUD)
    assert events == [] and fake.stt_calls == 0


def test_wake_then_user_text():
    # A 2연속 → wake(응답음) → 발화(칩 판정) → 침묵 → 전사 → on_user_text
    speech_frames = 5
    silence_frames = 12          # 12×80ms ≈ 0.96초 > 0.8초
    fake = Fake(scores=[(0.9, 0), (0.9, 0)] + [(0, 0)] * 40, text="화장실 어디야")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    assert run_frames(m, 2, LOUD)[-1] == "wake"
    assert wakes == [1]
    run_frames(m, speech_frames, LOUD, t0=1.0, vad=True)  # 사용자 발화
    results = run_frames(m, silence_frames, QUIET, t0=2.0)  # 침묵 → 종료
    assert "user_text" in results
    assert texts == ["화장실 어디야"]
    assert events == []


def test_wake_silent_timeout():
    # 호출 후 아무 말 없음 → user_text 없이 복귀 (호출 오탐 기록 지점)
    fake = Fake(scores=[(0.9, 0), (0.9, 0)] + [(0, 0)] * 200, text="")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    run_frames(m, 2, LOUD)
    results = run_frames(m, 80, QUIET, t0=1.0)   # 6.4초 — 상한 초과
    assert "wake_silent" in results
    assert texts == []


def test_emergency_preempts_listen():
    # 청취 창 도중 긴급어 — 긴급이 절대 우선 (명세 11절)
    fake = Fake(
        scores=[(0.9, 0), (0.9, 0), (0, 0.9), (0, 0.9)] + [(0, 0)] * 10,
        text="정지",
    )
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    run_frames(m, 2, LOUD)                        # wake → listen
    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD, t0=1.0)
    assert results[-1] == "emergency"
    assert events[0].keyword == "정지"
    assert texts == []                            # 청취는 폐기됐다


def test_muted_suppresses_and_failsafe_unmutes():
    fake = Fake(scores=[(0, 0.9)] * 40, text="멈춰")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    m.set_muted(True, now=0.0, failsafe_sec=10.0)
    run_frames(m, 10, LOUD, t0=0.0)               # 억제 중 — 발동 없음
    assert events == []
    # fail-safe: 10초 뒤엔 해제 신호를 놓쳐도 자동 해제
    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD, t0=11.0)
    assert results[-1] == "emergency"


def test_unmute_clears_ring():
    # 해제 시 버퍼를 비운다 — 억제 중 소리가 검증 전사에 섞이지 않게
    fake = Fake(scores=[(0, 0)] * 5 + [(0, 0.9)] * 10, text="멈춰")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    m.set_muted(True, now=0.0)
    run_frames(m, 5, LOUD, t0=0.0)
    m.set_muted(False, now=0.5)
    assert len(m._ring) == 0


def test_listen_timing_breakdown():
    """계측: 대기·발화·말끝판정·STT 를 분리 기록한다.

    "STT 가 진짜 병목인가"를 조사하려면 청취 6초 묶음을 쪼개야 한다
    (2026-08-28 주행 실측: 청취+STT 중앙값 6.1초, 구간 분리 불가였다).
    """
    import pytest

    fake = Fake(scores=[(0.9, 0), (0.9, 0)] + [(0, 0)] * 40, text="안내소로 가자")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    run_frames(m, 2, LOUD)                      # wake — 창 열림 now=0.08
    run_frames(m, 5, LOUD, t0=1.0, vad=True)    # 발화 1.00~1.32
    results = run_frames(m, 20, QUIET, t0=1.4)  # 최소 개방 2.5초(9/1)를 넘겨 마감  # 침묵 0.8초 누적 → 종료
    assert "user_text" in results
    t = m.last_listen_timing
    assert t is not None
    assert t["wait"] == pytest.approx(0.92, abs=0.01)    # 창 열림 → 말 시작
    assert t["speech"] == pytest.approx(0.32, abs=0.01)  # 말 시작 → 말끝
    # 말끝 → 판정: 침묵 0.8초 이상. 최소 개방(2.5초, 9/1)이 마감을 미루면
    # 그만큼 커진다 — 실제 기다린 시간의 정직한 계측이다.
    assert t["tail"] >= 0.80
    assert t["stt"] >= 0.0                               # 가짜 STT 는 즉시


def test_listen_timing_resets_on_new_window():
    """새 청취 창이 열리면 직전 계측은 비워진다 — 낡은 수치 재사용 방지."""
    fake = Fake(scores=[(0.9, 0), (0.9, 0)] + [(0, 0)] * 60, text="안내소로 가자")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    run_frames(m, 2, LOUD)
    run_frames(m, 5, LOUD, t0=1.0, vad=True)
    run_frames(m, 20, QUIET, t0=1.4)  # 최소 개방 2.5초(9/1)를 넘겨 마감
    assert m.last_listen_timing is not None
    m.arm_followup(now=10.0)
    m.set_muted(False, now=10.0)   # 예약된 재청취 창이 열린다
    assert m.last_listen_timing is None


def test_emergency_false_fire_in_listen_keeps_utterance():
    """청취 중 긴급 게이트 오발동이 긴급어가 아니면 발화를 버리지 않는다.

    실측 결함(2026-08-28): "화장실로 가줘"의 '가줘' 울림이 긴급 모델을
    깨워 청취를 가로챘고, '현실로 가줘' 전사가 긴급어 매칭에 실패하자
    통째로 기각됐다 — 사용자는 같은 말을 다시 해야 했다.
    """
    fake = Fake(
        scores=[(0.9, 0), (0.9, 0)]          # wake
        + [(0, 0)] * 5                        # 사용자 발화
        + [(0, 0.9), (0, 0.9)]                # 긴급 게이트 오발동
        + [(0, 0)] * 10,
        text="화장실로 가줘",                  # 긴급어 아님 — 진짜 명령
    )
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    run_frames(m, 2, LOUD)                            # wake → listen
    run_frames(m, 5, LOUD, t0=1.0, vad=True)          # 발화 수집
    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD, t0=1.5, vad=True)
    assert "listen_resumed" in results                # 오발동 → 청취 복원
    # 복원된 청취가 말끝(침묵 0.8초)까지 정상 진행돼 전체 발화를 전사한다
    results = run_frames(m, 14, QUIET, t0=3.0)
    assert "user_text" in results
    assert texts == ["화장실로 가줘"]
    assert events == []                                # 긴급 이벤트는 없다


def test_emergency_false_fire_in_listen_still_blocks_hallucination():
    """청취 가로챔이라도 환각 단골 문구·빈 전사는 넘기지 않는다."""
    fake = Fake(
        scores=[(0.9, 0), (0.9, 0)] + [(0, 0)] * 5
        + [(0, 0.9), (0, 0.9)] + [(0, 0)] * 10,
        text="",                               # 빈 전사
    )
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    run_frames(m, 2, LOUD)
    run_frames(m, 5, LOUD, t0=1.0, vad=True)
    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD, t0=1.5, vad=True)
    assert "listen_resumed" in results                # 복원은 되지만
    results = run_frames(m, 14, QUIET, t0=3.0)        # 말끝 → 전사 → 환각 기각
    assert "wake_silent" in results
    assert texts == []
    assert events == []

def test_wake_window_timeout_fires_listen_empty():
    """'비카야' 창이 빈손으로 닫히면 on_listen_empty 로 알린다.

    실측(2026-08-28 19:03): 작은 소리로 말해 칩이 발화를 못 봤는데 로봇이
    침묵해서, 사용자는 "아예 감지가 안 된다"고 느꼈다. 못 들었으면
    못 들었다고 말해야 한다.
    """
    fake = Fake(scores=[(0.9, 0), (0.9, 0)] + [(0, 0)] * 200, text="")
    events, texts, wakes, empties = [], [], [], []
    m = WakewordMonitor(
        on_emergency=events.append,
        on_user_text=texts.append,
        on_wake=lambda: wakes.append(1),
        on_listen_empty=lambda: empties.append(1),
        predict=fake.predict,
        transcribe=fake.transcribe,
    )
    run_frames(m, 2, LOUD)
    results = run_frames(m, 80, QUIET, t0=1.0)   # 상한 초과 — 발화 없음
    assert "wake_silent" in results
    assert empties == [1]
    assert texts == []


def test_followup_window_timeout_stays_silent():
    """질문 답변용(followup) 창의 침묵은 알리지 않는다 — 무응답 처리는
    상태를 가진 Mission 몫이라 여기서 말하면 두 번 말하게 된다."""
    fake = Fake(scores=[(0, 0)] * 500, text="")
    events, texts, wakes, empties = [], [], [], []
    m = WakewordMonitor(
        on_emergency=events.append,
        on_user_text=texts.append,
        on_wake=lambda: wakes.append(1),
        on_listen_empty=lambda: empties.append(1),
        predict=fake.predict,
        transcribe=fake.transcribe,
    )
    m.arm_followup(now=0.0)
    m.set_muted(False, now=0.0)                  # 재청취 창 열림
    results = run_frames(m, 400, QUIET, t0=0.1)  # 30초 상한 초과
    assert "wake_silent" in results
    assert empties == []


def test_confirm_window_uses_hinted_transcriber():
    """질문 답변(followup) 창은 정답 후보를 귀띔한 전사기를 쓴다.

    "그래"가 '굿에이'로 전사되던 실측(2026-08-28) — 짧은 답은 후보를
    귀띔(initial_prompt)해야 맞는다. 자유 명령 창에는 적용하지 않는다.
    """
    fake = Fake(scores=[(0, 0)] * 100, text="긴급전사")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    m._transcribe_listen = lambda a: "일반전사"
    m._transcribe_confirm = lambda a: "그래"
    m.arm_followup(now=0.0)
    m.set_muted(False, now=0.0)                   # 재청취(followup) 창 열림
    run_frames(m, 5, LOUD, t0=0.1, vad=True)
    results = run_frames(m, 12, QUIET, t0=0.6)
    assert "user_text" in results
    assert texts == ["그래"]


def test_wake_window_ignores_confirm_transcriber():
    """'비카야' 자유 명령 창은 귀띔 없이 일반 전사기를 쓴다 — 귀띔이
    자유 발화를 후보 쪽으로 왜곡하면 안 된다."""
    fake = Fake(scores=[(0.9, 0), (0.9, 0)] + [(0, 0)] * 60, text="긴급전사")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    m._transcribe_listen = lambda a: "화장실로 가자"
    m._transcribe_confirm = lambda a: "그래"
    run_frames(m, 2, LOUD)                        # wake 창
    run_frames(m, 5, LOUD, t0=1.0, vad=True)
    results = run_frames(m, 20, QUIET, t0=1.4)  # 최소 개방 2.5초(9/1)를 넘겨 마감
    assert "user_text" in results
    assert texts == ["화장실로 가자"]


def test_near_silence_never_reaches_stt():
    """무음에 가까운 수음은 STT 로 보내지 않는다.

    실측(2026-08-28): 순간 잡음이 VAD 를 스쳐 rms 0.0034·발화 0.00초짜리
    무음이 whisper 에 들어갔고, 장소 귀띔이 무음 환각을 '방2'(진짜 목적지)로
    둔갑시켜 유령 주행 명령이 됐다. 진짜 조용한 답("응" rms 0.0185·0.24초)은
    문턱을 넘는다.
    """
    TINY = np.full(1280, 100, dtype=np.int16)     # rms ≈ 0.003 — 무음 수준
    fake = Fake(scores=[(0.9, 0), (0.9, 0)] + [(0, 0)] * 60, text="방2")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    run_frames(m, 2, LOUD)                        # wake → listen
    run_frames(m, 1, TINY, t0=1.0, vad=True)      # 잡음이 VAD 를 한 번 스침
    results = run_frames(m, 70, TINY, t0=1.1)  # 6초 상한(반짝 무효화 9/1) 경과
    assert "wake_silent" in results
    assert texts == []
    assert fake.stt_calls == 0                    # whisper 를 부르지도 않는다


class TestAgcDesiredFromEnv:
    """AGC 목표 레벨 환경변수 해석 — 칩 쓰기는 장치 시험(자동화 제외)."""

    def test_default_string_parses(self):
        from src.dsp_state import agc_desired_from_env
        assert agc_desired_from_env("0.010") == 0.010

    def test_off_values_skip_write(self):
        from src.dsp_state import agc_desired_from_env
        for raw in ("", "0", "off", "none"):
            assert agc_desired_from_env(raw) is None

    def test_garbage_skips_write(self):
        from src.dsp_state import agc_desired_from_env
        assert agc_desired_from_env("두배로") is None

    def test_out_of_range_skips_write(self):
        """말도 안 되는 값(음수·1 초과)은 칩에 쓰지 않는다."""
        from src.dsp_state import agc_desired_from_env
        assert agc_desired_from_env("-0.01") is None
        assert agc_desired_from_env("5.0") is None


def test_confirm_hint_covers_arrival_dialog_vocab():
    """확인 창 귀띔에 도착 후 대화의 답(대기·시간·종료)이 들어있어야 한다 —
    빠지면 오전사로 wait/finish 가 안 잡힌다 (2026-08-30)."""
    from src.wakeword_monitor import CONFIRM_HINT
    for word in ["기다려", "이십", "삼십", "반시간", "됐어", "그만"]:
        assert word in CONFIRM_HINT, f"귀띔에 '{word}' 누락"
    # 너무 길면 딴말이 후보로 둔갑 — 상한을 둔다 (whisper 프롬프트 예산)
    assert len(CONFIRM_HINT) < 200


def test_listen_state_relay():
    """청취 상태 중계(2026-08-30): open → speech → closed(전사 성공).
    미션이 이걸로 무응답 시계를 귀가 바쁜 동안 멈춘다."""
    states = []
    fake = Fake(scores=[(0.9, 0), (0.9, 0)] + [(0, 0)] * 40, text="십 분만 기다려")
    events, texts, wakes = [], [], []
    m = WakewordMonitor(
        on_emergency=events.append, on_user_text=texts.append,
        on_wake=lambda: wakes.append(1), predict=fake.predict,
        transcribe=fake.transcribe, on_listen_state=states.append)
    run_frames(m, 2, LOUD)                      # wake → open
    run_frames(m, 5, LOUD, t0=1.0, vad=True)    # 발화 → speech
    run_frames(m, 12, QUIET, t0=2.0)            # 말끝 → 전사 → closed
    assert states == ["open", "speech", "closed"]


def test_listen_state_empty_on_silence():
    states = []
    fake = Fake(scores=[(0.9, 0), (0.9, 0)] + [(0, 0)] * 200, text="")
    events, texts, wakes = [], [], []
    m = WakewordMonitor(
        on_emergency=events.append, on_user_text=texts.append,
        on_wake=lambda: wakes.append(1), predict=fake.predict,
        transcribe=fake.transcribe, on_listen_state=states.append)
    run_frames(m, 2, LOUD)
    run_frames(m, 80, QUIET, t0=1.0)            # 6.4초 침묵 — 창 만료
    assert states == ["open", "empty"]


def test_out_of_window_answer_rescued_when_followup_armed():
    """질문 답 대기 중(followup 예약)의 창 밖 발화는 답으로 채택한다
    (2026-08-31: "필요없다구"를 정확히 전사하고도 창 밖이라 기각했다)."""
    fake = Fake(scores=[(0, 0.9), (0, 0.9)] + [(0, 0)] * 10, text="필요없다구")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    m.arm_followup(now=0.0)                     # 질문이 나감 — 답 대기
    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD)
    assert "user_text" in results
    assert texts == ["필요없다구"]
    assert events == []


def test_out_of_window_speech_still_rejected_without_followup():
    """예약이 없으면(행인 대화 등) 창 밖 발화는 전처럼 기각 — 규약 유지."""
    fake = Fake(scores=[(0, 0.9), (0, 0.9)] + [(0, 0)] * 10, text="필요없다구")
    events, texts, wakes = [], [], []
    m = make(fake, events, texts, wakes)
    results = run_frames(m, 2 + POST_ROLL_FRAMES, LOUD)
    assert "reject" in results
    assert texts == []

"""'비카야' 판정 관문 (2026-10-07 사용자 결정, 인수인계 문서 작업 계획 탭 호출 반응표).

대답할지는 상태를 아는 미션이 정한다. 감시기는 지금처럼 호출 즉시 자유 창을 열어
"비카야 화장실 가자" 한 호흡을 살리고, 그 창의 말은 미션 판정(/vica/wake_reply
listen·ignore)이 올 때까지 쥐고 있다. listen 이면 내보내고, ignore 면 창을 닫고 버리고,
판정이 끝내 안 오면(미션 멈춤) 시간을 넘겨 버린다. 질문 답 창은 관문과 무관하다.
"""
from __future__ import annotations

import numpy as np

from src.wakeword_monitor import WAKE_REPLY_TIMEOUT_SEC, WakewordMonitor

LOUD = np.full(1280, 3000, dtype=np.int16)
QUIET = np.zeros(1280, dtype=np.int16)
STEP = 0.08


class Fake:
    """프레임 0~1 에서 호출 점수, 전사는 고정."""

    def __init__(self, text: str = "화장실 가자", wake_frames=(0, 1)):
        self.text = text
        self.wake_frames = set(wake_frames)
        self.i = -1

    def predict(self, _frame):
        self.i += 1
        return {"a": 0.9 if self.i in self.wake_frames else 0.0, "b": 0.0}

    def transcribe(self, _audio):
        return self.text


class Rig:
    def __init__(self, gate: bool = True, audio: bool = False, wake_frames=(0, 1)):
        self.events: list = []
        self.states: list[str] = []
        self.fake = Fake(wake_frames=wake_frames)
        self.m = WakewordMonitor(
            on_emergency=lambda e: None,
            on_user_text=lambda t: self.events.append(("text", t)),
            on_wake=lambda: self.events.append(("wake", None)),
            on_listen_state=self.states.append,
            on_user_audio=(lambda a: self.events.append(("audio", len(a)))) if audio else None,
            predict=self.fake.predict,
            transcribe=self.fake.transcribe,
            wake_gate=gate,
        )
        self.t = 0.0

    def frame(self, loud: bool = False):
        r = self.m.process_frame(LOUD if loud else QUIET, now=self.t, vad=loud)
        self.t += STEP
        return r

    def wake(self):
        for _ in range(2):
            self.frame()
        assert ("wake", None) in self.events

    def speak_and_close(self):
        """발화 5프레임 + 침묵 — 최소 개방(2.5초) 뒤 말끝 판정으로 창이 닫힌다."""
        for _ in range(5):
            self.frame(loud=True)
        for _ in range(40):
            self.frame()

    def texts(self):
        return [v for k, v in self.events if k == "text"]


def test_listen_before_the_words_passes_them_on():
    rig = Rig()
    rig.wake()
    rig.m.set_wake_verdict("listen")
    rig.speak_and_close()
    assert rig.texts() == ["화장실 가자"]
    assert "closed" in rig.states


def test_words_wait_for_a_late_listen():
    rig = Rig(audio=True)
    rig.wake()
    rig.speak_and_close()
    assert rig.texts() == []                      # 판정 전 — 쥐고 있다
    assert "closed" not in rig.states
    rig.m.set_wake_verdict("listen")
    rig.frame()
    kinds = [k for k, _ in rig.events if k != "wake"]
    assert kinds == ["audio", "text"]             # 소리 먼저, 글자 다음(원래 순서)
    assert rig.texts() == ["화장실 가자"]


def test_ignore_closes_the_open_window_at_once():
    rig = Rig()
    rig.wake()
    rig.m.set_wake_verdict("ignore")
    rig.frame()
    assert "empty:wake-ignored" in rig.states
    assert rig.m._state == "idle"
    rig.speak_and_close()
    assert rig.texts() == []


def test_ignore_after_the_words_drops_them():
    rig = Rig()
    rig.wake()
    rig.speak_and_close()
    rig.m.set_wake_verdict("ignore")
    rig.frame()
    assert rig.texts() == []
    assert rig.states[-1] == "empty:wake-ignored"


def test_no_reply_means_no_answer():
    """미션이 멈췄으면 판정이 안 온다 — 시간이 지나면 버리고 창을 닫는다."""
    rig = Rig()
    rig.wake()
    # 말을 길게 이어 창이 열린 채로 시한을 넘긴다.
    while rig.t < WAKE_REPLY_TIMEOUT_SEC + 0.5:
        rig.frame(loud=True)
    assert "empty:wake-no-reply" in rig.states
    assert rig.m._state == "idle"
    for _ in range(40):
        rig.frame()
    assert rig.texts() == []


def test_followup_window_is_not_gated():
    """질문 답 창은 판정과 무관하다 — 앞 호출이 ignore 였어도 답은 간다."""
    rig = Rig()
    rig.wake()
    rig.m.set_wake_verdict("ignore")
    rig.frame()
    rig.m.arm_followup(now=rig.t)
    rig.m.set_speaking(True, now=rig.t)
    rig.m.set_speaking(False, now=rig.t)
    assert rig.m._state == "listen" and rig.m._listen_is_followup
    rig.speak_and_close()
    assert rig.texts() == ["화장실 가자"]


def test_gate_off_keeps_the_old_flow():
    rig = Rig(gate=False)
    rig.wake()
    rig.speak_and_close()
    assert rig.texts() == ["화장실 가자"]


def test_a_new_wake_waits_for_its_own_verdict():
    """앞 호출의 listen 이 다음 호출에 새지 않는다."""
    rig = Rig(wake_frames=(0, 1, 60, 61))
    rig.wake()
    rig.m.set_wake_verdict("listen")
    rig.speak_and_close()
    assert rig.texts() == ["화장실 가자"]
    while rig.fake.i < 61:                        # 두 번째 호출(프레임 60~61)까지
        rig.frame()
    assert rig.m._state == "listen" and not rig.m._listen_is_followup
    rig.speak_and_close()
    assert rig.texts() == ["화장실 가자"]          # 새 판정 전 — 두 번째 말은 쥐고 있다
    rig.m.set_wake_verdict("listen")
    rig.frame()
    assert rig.texts() == ["화장실 가자", "화장실 가자"]


def test_a_late_verdict_for_the_previous_call_is_ignored():
    """앞 호출의 판정이 늦게 와 다음 호출에 붙으면 안 된다 — 호출 번호로 짝을 맞춘다(10-07 검토)."""
    rig = Rig(wake_frames=(0, 1, 90, 91))         # 두 번째 호출은 첫 시한(6초) 뒤 7.2초
    rig.wake()
    first = rig.m.wake_seq
    while rig.fake.i < 91:                        # 첫 호출은 판정 없이 시한을 넘긴다
        rig.frame()
    assert "empty:wake-no-reply" in rig.states
    second = rig.m.wake_seq
    assert second == first + 1
    rig.m.set_wake_verdict("listen", first)      # 첫 호출의 늦은 판정
    rig.speak_and_close()
    assert rig.texts() == []                      # 두 번째 호출의 말은 여전히 판정 대기
    rig.m.set_wake_verdict("listen", second)
    rig.frame()
    assert rig.texts() == ["화장실 가자"]


def test_verdict_without_a_number_is_for_the_current_call():
    """번호 없는 판정(옛 미션·가상 로봇)은 지금 호출의 것으로 본다."""
    rig = Rig()
    rig.wake()
    rig.m.set_wake_verdict("listen", None)
    rig.speak_and_close()
    assert rig.texts() == ["화장실 가자"]

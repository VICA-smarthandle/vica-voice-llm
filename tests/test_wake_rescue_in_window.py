"""창 안 호출 구제 — 창이 열린 동안의 "비카야"를 소리로 판정한다."""
from __future__ import annotations

import numpy as np

from src.wakeword_monitor import WAKE_WORD_TEXT, WakewordMonitor

LOUD = np.full(1280, 3000, dtype=np.int16)
QUIET = np.zeros(1280, dtype=np.int16)


class Fake:
    """점수는 프레임마다 지정, 전사는 고정."""

    def __init__(self, text: str, wake_frames: set[int]):
        self.text = text
        self.wake_frames = wake_frames
        self.i = -1

    def predict(self, _frame):
        self.i += 1
        return {"a": 0.9 if self.i in self.wake_frames else 0.0, "b": 0.0}

    def transcribe(self, _audio):
        return self.text


def _run(text: str, wake_frames: set[int]):
    """자유 창을 열고(첫 호출), 그 창 안에서 다시 호출 소리를 낸다."""
    texts, wakes, states = [], [], []
    fake = Fake(text, {0, 1} | wake_frames)
    m = WakewordMonitor(
        on_emergency=lambda e: None,
        on_user_text=texts.append,
        on_wake=lambda: wakes.append(1),
        on_listen_state=states.append,
        predict=fake.predict,
        transcribe=fake.transcribe,
    )
    for i in range(2):
        m.process_frame(QUIET, now=i * 0.08, vad=False)
    t = 2 * 0.08
    for i in range(5):
        m.process_frame(LOUD, now=t + i * 0.08, vad=True)
    t += 5 * 0.08
    out = None
    for i in range(40):
        r = m.process_frame(QUIET, now=t + i * 0.08, vad=False)
        if r is not None:
            out = r
            break
    return out, texts, wakes, states


def _run_followup(text: str, wake_frames: set[int]):
    """재청취(질문) 창을 열고, 그 창 안에서 호출 소리를 낸다."""
    texts, states = [], []
    fake = Fake(text, wake_frames)
    m = WakewordMonitor(
        on_emergency=lambda e: None,
        on_user_text=texts.append,
        on_wake=lambda: None,
        on_listen_state=states.append,
        predict=fake.predict,
        transcribe=fake.transcribe,
    )
    m.arm_followup(now=0.0)
    m.set_muted(False, now=0.0)
    t = 0.0
    for i in range(2):
        m.process_frame(QUIET, now=t, vad=False)
        t += 0.08
    for i in range(5):
        m.process_frame(LOUD, now=t, vad=True)
        t += 0.08
    out = None
    for i in range(40):
        r = m.process_frame(QUIET, now=t + i * 0.08, vad=False)
        if r is not None:
            out = r
            break
    return out, texts, states


class TestFollowupAnswerBeatsRescue:
    """질문 창의 정답 어휘는 창 안 호출 구제보다 우선한다 (2026-09-11 실기)."""

    def test_short_answer_survives_wake_blip(self):
        out, texts, states = _run_followup("아니요.", wake_frames={0, 1})
        assert out == "user_text"
        assert texts == ["아니요."]
        assert not any(s.startswith("wake-rescue") for s in states)
        assert any(s.startswith("answer-beats-rescue") for s in states)

    def test_non_answer_short_call_is_still_rescued(self):
        """정답 어휘가 아닌 짧은 말(진짜 비카야 오전사)은 기존대로 구제한다."""
        out, texts, states = _run_followup("이깨야.", wake_frames={0, 1})
        assert out == "user_text"
        assert texts == [WAKE_WORD_TEXT]
        assert any(s.startswith("wake-rescue") for s in states)

    def test_free_window_short_answer_word_is_still_rescued(self):
        """자유 창은 정답 어휘 예외가 없다 — heard_wake 면 그대로 구제한다."""
        out, texts, _, states = _run("그래.", wake_frames={2, 3})
        assert out == "user_text"
        assert texts == [WAKE_WORD_TEXT]
        assert any(s.startswith("wake-rescue") for s in states)


class TestRescue:
    def test_misheard_call_is_rescued_by_sound(self):
        """'비켜야'로 적혀도 호출 소리를 들었으면 호출이다."""
        out, texts, wakes, states = _run("비켜야.", wake_frames={2, 3})
        assert out == "user_text"
        assert texts == [WAKE_WORD_TEXT]
        assert any(s.startswith("wake-rescue") for s in states)

    def test_rescue_does_not_open_a_new_window(self):
        """창 안 호출은 창을 새로 열지 않는다 — on_wake 는 첫 호출 1회뿐."""
        _, _, wakes, _ = _run("미카야", wake_frames={2, 3})
        assert wakes == [1]

    def test_command_is_not_rescued(self):
        """전사가 길면 명령이다 — 소리를 들었어도 그대로 흘려보낸다."""
        out, texts, _, states = _run("화장실로 가자", wake_frames={2, 3})
        assert out == "user_text"
        assert texts == ["화장실로 가자"]
        assert not any(s.startswith("wake-rescue") for s in states)

    def test_no_wake_sound_means_no_rescue(self):
        """호출 소리가 없으면 짧은 말이어도 구제하지 않는다 — 소리가 정본이다."""
        out, texts, _, states = _run("비켜야.", wake_frames=set())
        assert out == "user_text"
        assert texts == ["비켜야."]
        assert not any(s.startswith("wake-rescue") for s in states)

    def test_hallucinated_call_still_rescued(self):
        """호출을 '감사합니다'로 적어도 살린다 — 구제가 환각 검사보다 앞이다."""
        out, texts, _, _ = _run("감사합니다", wake_frames={2, 3})
        assert out == "user_text"
        assert texts == [WAKE_WORD_TEXT]


class TestLengthBoundary:
    def test_one_char_is_too_short(self):
        """'네' 한 글자는 구제 대상이 아니다 — 짧은 답을 호출로 만들면 안 된다."""
        out, texts, _, _ = _run("네", wake_frames={2, 3})
        assert texts == ["네"]
        assert out == "user_text"

    def test_spaced_variant_is_rescued(self):
        """'비 카야' 처럼 띄어 적어도 2토큰 3자라 구제한다."""
        _, texts, _, _ = _run("비 카야", wake_frames={2, 3})
        assert texts == [WAKE_WORD_TEXT]


def test_listen_gate_does_not_consume_outer_cooldown():
    """창 안 관찰용 게이트는 **별도 객체**다."""
    m = WakewordMonitor(on_emergency=lambda e: None,
                        on_user_text=lambda t: None,
                        predict=lambda f: {"a": 0.0, "b": 0.0},
                        transcribe=lambda a: "")
    assert m.gate_a is not m.gate_a_listen
    assert m.gate_a_listen.cooldown_sec == 0.0
    assert m.gate_a_listen.threshold == m.gate_a.threshold


def test_rescue_state_prefix_is_the_node_contract():
    """구제 보고 문자열의 접두는 노드와의 계약이다."""
    _, _, _, states = _run("비켜야.", wake_frames={2, 3})
    rescued = [s for s in states if s.startswith("wake-rescue")]
    assert rescued, "구제 사실이 listen_state 로 보고돼야 한다"
    assert rescued[0].startswith("wake-rescue ")

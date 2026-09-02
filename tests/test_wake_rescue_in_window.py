"""창 안 호출 구제 — 창이 열린 동안의 "비카야"를 소리로 판정한다.

배경(2026-09-02 실기): 창이 열린 동안에는 호출 음향 모델이 꺼져 있어
"비카야"가 오직 STT 전사로만 판별됐다. 그런데 whisper 는 사전에 없는 이
이름을 못 적는다 — 창 안 재호출 13회 전수에서 '비카야'로 적힌 것이 **0건**
(미카야 4·피카야 3·비켜야 3·리카야/비кая/이까랑 각 1). 어휘 목록은 하루에
6종이 나와 따라잡을 수 없어, 소리 쪽 증거로 판정하도록 바꿨다.

핵심 계약 셋:
  1. 창 안에서 호출 소리가 나도 **창을 새로 열지 않는다** (뒷말 보존)
  2. 전사가 한 덩어리 짧은 말이면 글자가 무엇이든 호출로 본다
  3. 전사가 길면(명령) 구제하지 않는다
"""
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
    """자유 창을 열고(첫 호출), 그 창 안에서 다시 호출 소리를 낸다.

    프레임 0~1 = 창을 여는 첫 호출(persist=2), 2~6 = 발화, 7~ = 침묵.
    `wake_frames` 는 **창 안**에서 호출 소리가 난 프레임이다.
    """
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
    # 프레임 0~1 로 호출 발동 → 창 열림 (persist=2)
    for i in range(2):
        m.process_frame(QUIET, now=i * 0.08, vad=False)
    # 발화 5프레임 + 침묵 11프레임 → 말끝 판정으로 창이 닫히고 전사가 나온다.
    # 최소 개방(2.5초)을 넘기려고 침묵을 넉넉히 준다.
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


class TestRescue:
    def test_misheard_call_is_rescued_by_sound(self):
        """'비켜야'로 적혀도 호출 소리를 들었으면 호출이다."""
        out, texts, wakes, states = _run("비켜야.", wake_frames={2, 3})
        assert out == "user_text"
        assert texts == [WAKE_WORD_TEXT]
        assert any(s.startswith("wake-rescue") for s in states)

    def test_rescue_does_not_open_a_new_window(self):
        """창 안 호출은 창을 새로 열지 않는다 — on_wake 는 첫 호출 1회뿐.

        열어 버리면 이어지는 말이 통째로 날아간다("비카야 화장실로 가자").
        """
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
    """창 안 관찰용 게이트는 **별도 객체**다.

    gate_a 를 그대로 먹이면 쿨다운이 소모돼, 창이 닫힌 직후의 진짜 호출이
    막힌다 — 사용자가 다시 불러도 반응이 없는 회귀가 된다.
    """
    m = WakewordMonitor(on_emergency=lambda e: None,
                        on_user_text=lambda t: None,
                        predict=lambda f: {"a": 0.0, "b": 0.0},
                        transcribe=lambda a: "")
    assert m.gate_a is not m.gate_a_listen
    assert m.gate_a_listen.cooldown_sec == 0.0
    assert m.gate_a_listen.threshold == m.gate_a.threshold

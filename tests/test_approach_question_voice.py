"""접근 질문('안내를 받으시겠어요?') 중 음성 쪽 (2026-10-09 사용자 결정, run82 실기).

1. 귀: 미션이 이 질문의 답을 기다리는 동안(dialog_state=awaiting_user) '비카야'를 호출로 확정하지 않는다 —
   말 끊기·"네?"·호출 창 바꾸기·미션 판정 대기가 모두 없다. 긴급어(멈춰)는 그대로 듣는다.
2. LLM: 이 동안은 질문(question)에만 소리 내어 답한다. 되묻기(clarify)·"네?"·"네, 말씀하세요."·
   확인 질문은 지운다 — 다시 묻기("안내를 받으시겠어요?")는 미션이 한다(목소리 하나).
3. LLM 지시문: 그 질문의 답은 한 문장으로 답만 하고 "?"로 끝내지 않는다(18:12 실기 — "…비카입니다.
   말씀해 주세요?"·"잠깐만요, 안내를 시작할까요?" 뒤에 미션이 또 물어 질문이 겹쳤다, 사용자 결정 1번).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src import langchain_intent_parser as parser
from src.mission_question import quiet_for_mission
from src.replies import CANCEL_CONFIRM, RESUME_CONFIRM, WAKE_GREETING
from src.schema import DestinationData, VicaIntent
from src.wakeword_monitor import POST_ROLL_FRAMES, WakewordMonitor

LOUD = np.full(1280, 3000, dtype=np.int16)
FRAME_SEC = 0.08
ROOT = Path(__file__).resolve().parents[1]


class Fake:
    def __init__(self, scores, text=""):
        self.scores = list(scores)
        self.text = text

    def predict(self, _frame):
        a, b = self.scores.pop(0) if self.scores else (0.0, 0.0)
        return {"a": a, "b": b}

    def transcribe(self, _audio):
        return self.text


def _make(fake, events=None):
    log = {"wakes": []}
    m = WakewordMonitor(
        on_emergency=(events.append if events is not None else lambda e: None),
        on_user_text=lambda t: None,
        on_wake=lambda: log["wakes"].append(1),
        predict=fake.predict,
        transcribe=fake.transcribe,
    )
    return m, log


def _feed(m, frames, t0):
    return [m.process_frame(f, now=t0 + i * FRAME_SEC, vad=v) for i, (f, v) in enumerate(frames)]


# ---- 1. 귀 -----------------------------------------------------------------------------

def test_call_is_not_confirmed_while_the_approach_question_waits():
    m, log = _make(Fake([(0.9, 0)] * 3))
    m.set_wake_suppressed(True)
    assert "wake" not in _feed(m, [(LOUD, None)] * 3, 0.0)
    assert log["wakes"] == []


def test_call_works_again_after_the_question():
    m, log = _make(Fake([(0.9, 0)] * 3))
    m.set_wake_suppressed(True)
    m.set_wake_suppressed(False)
    assert "wake" in _feed(m, [(LOUD, None)] * 3, 0.0)


def test_call_heard_inside_the_answer_window_is_not_rescued():
    m, _ = _make(Fake([(0.9, 0), (0.9, 0)] + [(0.0, 0)] * 3 + [(0.9, 0), (0.9, 0)]))
    _feed(m, [(LOUD, None)] * 2, 0.0)            # 창을 연다(질문 전의 진짜 호출이라고 치자)
    m.set_wake_suppressed(True)                   # 미션이 접근 질문을 시작했다
    _feed(m, [(LOUD, True)] * 5, 0.3)
    assert m._listen_heard_wake is False


def test_stop_word_still_works_while_calls_are_off():
    events = []
    fake = Fake([(0, 0.9), (0, 0.9)] + [(0, 0)] * POST_ROLL_FRAMES, text="멈춰!")
    m, _ = _make(fake, events)
    m.set_wake_suppressed(True)
    out = _feed(m, [(LOUD, None)] * (2 + POST_ROLL_FRAMES), 0.0)
    assert out[-1] == "emergency"
    assert len(events) == 1


def test_node_turns_calls_off_from_the_mission_state():
    src = (ROOT / "src" / "ros_wakeword_node.py").read_text(encoding="utf-8")
    assert '"/vica/robot_state"' in src
    assert 'set_wake_suppressed(msg.dialog_state == "awaiting_user")' in src


# ---- 2. LLM 대답 -------------------------------------------------------------------------

def _i(kind, reply, need_confirm=False, safety="normal"):
    return VicaIntent(intent=kind, reply=reply, need_confirm=need_confirm, safety_flag=safety)


def test_only_a_question_is_answered_aloud():
    out = quiet_for_mission(_i("question", "저는 안내 로봇 비카예요."), "awaiting_user", "", 1.0)
    assert out.reply == "저는 안내 로봇 비카예요."


def test_other_replies_are_left_to_the_mission():
    for intent in (_i("clarify", "어느 곳으로 안내를 원하시나요?"),   # run82 17:19
                   _i("unknown", WAKE_GREETING),
                   _i("unknown", "네, 말씀하세요."),
                   _i("resume", RESUME_CONFIRM, need_confirm=True),
                   _i("cancel", CANCEL_CONFIRM, need_confirm=True)):
        out = quiet_for_mission(intent, "awaiting_user", "", 1.0)
        assert out.reply == "", intent.intent
        assert out.intent == intent.intent


def test_emergency_words_are_kept():
    out = quiet_for_mission(_i("unknown", "멈추겠습니다.", safety="emergency"), "awaiting_user", "", 1.0)
    assert out.reply == "멈추겠습니다."


def test_other_states_are_unchanged():
    assert quiet_for_mission(_i("clarify", "어느 화장실이요?"), "idle", "", 99.0).reply == "어느 화장실이요?"
    assert quiet_for_mission(_i("unknown", WAKE_GREETING), "idle", "", 99.0).reply == WAKE_GREETING


def test_resume_goes_to_the_mission_so_it_can_reask():
    """"다시 가자"는 평소 음성이 쥐고 확인을 묻지만, 접근 질문 중에는 미션에 바로 보내 미션이 다시 묻게 한다."""
    from src.schema import should_forward_intent
    out = quiet_for_mission(_i("resume", RESUME_CONFIRM, need_confirm=True), "awaiting_user", "", 1.0)
    assert (out.reply, out.need_confirm) == ("", False)
    assert should_forward_intent(out)


# ---- 3. LLM 지시문 — 접근 질문 중 답은 한 문장, 되묻지 않는다 --------------------------------------

DEST = DestinationData(id="x", name="식당", confirm_prompt="식당으로 안내해드릴까요?")
APPROACH_RULE = "한 문장으로 답만 하고 \"?\"로 끝내지 않는다"


def test_audio_prompt_answers_the_approach_question_without_asking_back():
    text = parser.build_audio_prompt([DEST])
    assert "인사 답 대기" in text
    assert APPROACH_RULE in text
    assert "말씀해 주세요" in text and "안내를 시작할까요?" in text     # 실기에 붙었던 말을 금지 예로 든다


def test_audio_prompt_question_mark_rule_names_the_exception():
    """[말투]의 '답을 들어야 하는 말은 "?"로 끝낸다'와 부딪치지 않게 예외를 적는다."""
    text = parser.build_audio_prompt([DEST])
    assert "접근 질문 중 question 답은 예외" in text


def test_text_prompt_has_the_same_rule():
    text = parser._build_system_prompt([DEST])
    assert "인사 답 대기" in text
    assert APPROACH_RULE in text

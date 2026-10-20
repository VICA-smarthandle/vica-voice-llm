"""접근 질문('안내를 받으시겠어요?') 중 음성 쪽 (2026-10-09 사용자 결정, run82 실기)."""
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
    _feed(m, [(LOUD, None)] * 2, 0.0)
    m.set_wake_suppressed(True)
    _feed(m, [(LOUD, True)] * 5, 0.3)
    assert m._listen_heard_wake is False


def test_stop_word_still_works_while_calls_are_off():
    events = []
    fake = Fake([(0, 0.9), (0, 0.9)] + [(0, 0)] * POST_ROLL_FRAMES, text="멈춰!")
    m, _ = _make(fake, events)
    m.set_wake_suppressed(True)
    out = _feed(m, [(LOUD, None)] * (1 + POST_ROLL_FRAMES), 0.0)
    assert out[-1] == "emergency"
    assert len(events) == 1


def test_node_turns_calls_off_from_the_mission_state():
    src = (ROOT / "src" / "ros_wakeword_node.py").read_text(encoding="utf-8")
    assert '"/vica/robot_state"' in src
    assert 'set_wake_suppressed(msg.dialog_state == "awaiting_user")' in src


def _i(kind, reply, need_confirm=False, safety="normal"):
    return VicaIntent(intent=kind, reply=reply, need_confirm=need_confirm, safety_flag=safety)


def test_only_a_question_is_answered_aloud():
    out = quiet_for_mission(_i("question", "저는 안내 로봇 비카예요."), "awaiting_user", "", 1.0)
    assert out.reply == "저는 안내 로봇 비카예요."


def test_other_replies_are_left_to_the_mission():
    for intent in (_i("clarify", "어느 곳으로 안내를 원하시나요?"),
                   _i("unknown", WAKE_GREETING),
                   _i("unknown", "네, 말씀하세요."),
                   _i("resume", RESUME_CONFIRM, need_confirm=True),
                   _i("cancel", CANCEL_CONFIRM, need_confirm=True)):
        out = quiet_for_mission(intent, "awaiting_user", "", 1.0)
        assert out.reply == "", intent.intent
        assert out.intent == intent.intent


def test_question_answer_that_asks_back_is_not_spoken():
    """10-10 12:07 — "그래"를 question 으로 잘못 받아 "어디로 안내해드릴까요?"를 말했고, 2초 뒤 미션이"""
    out = quiet_for_mission(_i("question", "어디로 안내해드릴까요?"), "awaiting_user", "", 1.0)
    assert out.reply == ""
    assert out.intent == "question"


def test_answer_with_a_trailing_question_is_not_spoken():
    out = quiet_for_mission(_i("question", "저는 시각장애인 안내로봇 비카입니다. 말씀해 주세요?"),
                            "awaiting_user", "", 1.0)
    assert out.reply == ""


def test_answer_inviting_speech_is_not_spoken():
    """대답을 청하는 끝말("말씀해 주세요.")도 되묻기다 — 귀가 이미 '대답을 청하는 말'로 보는 꼴(expects_answer)."""
    out = quiet_for_mission(_i("question", "네, 말씀해 주세요."), "awaiting_user", "", 1.0)
    assert out.reply == ""


def test_plain_answer_is_still_spoken():
    out = quiet_for_mission(_i("question", "저는 시각장애인 안내로봇 비카입니다."), "awaiting_user", "", 1.0)
    assert out.reply == "저는 시각장애인 안내로봇 비카입니다."


def test_questions_outside_the_approach_question_are_unchanged():
    out = quiet_for_mission(_i("question", "화장실로 가시려는 건가요?"), "idle", "", 99.0)
    assert out.reply == "화장실로 가시려는 건가요?"


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


DEST = DestinationData(id="x", name="식당", confirm_prompt="식당으로 안내해드릴까요?")
APPROACH_RULE = "한 문장으로 답만 하고 \"?\"로 끝내지 않는다"


def test_audio_prompt_answers_the_approach_question_without_asking_back():
    text = parser.build_audio_prompt([DEST])
    assert "인사 답 대기" in text
    assert APPROACH_RULE in text
    assert "말씀해 주세요" in text and "안내를 시작할까요?" in text


def test_audio_prompt_question_mark_rule_names_the_exception():
    """[말투]의 '답을 들어야 하는 말은 "?"로 끝낸다'와 부딪치지 않게 예외를 적는다."""
    text = parser.build_audio_prompt([DEST])
    assert "접근 질문 중 question 답은 예외" in text


def test_text_prompt_has_the_same_rule():
    text = parser._build_system_prompt([DEST])
    assert "인사 답 대기" in text
    assert APPROACH_RULE in text

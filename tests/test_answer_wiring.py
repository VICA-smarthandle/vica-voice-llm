"""대답 칸 배선(3-나-2 1단계) — ros_node 소스 구조로 확인한다. 1단계는 기록만 한다."""
from __future__ import annotations

import ast
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from src.langchain_intent_parser import _shortcut_intent, parse_intent
from src.one_voice import hand_over_question, mission_asks_resume, voice_may_speak
from src.question_record import OpenQuestion
from src.replies import COMMAND_DECLINED, RESUME_CONFIRM
from src.schema import DestinationData, VicaIntent, should_forward_intent
from tests.test_one_voice_wiring import _branch_tests, _calls, _method

SRC = Path(__file__).resolve().parents[1] / "src"


def _kw(call: ast.Call) -> dict:
    return {k.arg: ast.unparse(k.value) for k in call.keywords}


def test_text_path_passes_the_open_question():
    (call,) = _calls(_method("_on_user_text"), "parse_intent")
    assert _kw(call).get("question") == "self._question"


def test_audio_path_passes_the_question_it_started_with():
    fn = _method("_on_user_audio")
    (call,) = _calls(fn, "parse_intent_audio")
    assert _kw(call).get("question") == "question"
    snapshot = [u for u in _calls(fn, "self._audio_turn.update") if "question" in _kw(u)]
    assert snapshot and snapshot[0].lineno < call.lineno      # 그림자 갈래가 같은 질문으로 규칙을 본다


def test_time_based_awaiting_line_is_replaced_only_when_on():
    assert "answer_field.on()" in ast.unparse(_method("_situation_block"))


def test_answer_is_logged_after_publishing():
    fn = _method("_publish_intent")
    (log,) = _calls(fn, "answer_field.log_line")
    assert log.lineno > max(c.lineno for c in _calls(fn, "self._intent_pub.publish"))


def test_transcript_rule_is_only_logged():
    fn = _method("_log_transcript_rule")
    assert _calls(fn, "answer_field.short_answer")
    assert not _calls(fn, "self._intent_pub.publish") and not _calls(fn, "self._publish_intent")
    assert _calls(_method("_on_user_text"), "self._log_transcript_rule")


def test_answer_fields_go_to_the_mission_when_the_message_has_them():
    text = (SRC / "ros_convert.py").read_text()
    assert 'hasattr(msg, "question_id")' in text and "msg.answer_value" in text
    # 미션 질문 판단(mission_question)은 대답 칸을 모른다. one_voice 는 Task 3.4 부터 스위치를 본다 — 켜지면 "다시 출발할까요?"를
    # 미션이 묻는다(미션 파라미터 answer_field, 3.3).
    assert "answer_field" not in (SRC / "mission_question.py").read_text()


# ---- Task 3.4: "다시 출발할까요?"를 미션이 묻는다(원칙 3 · 9, 11-18 설계 4.4·4.5) ------------------------------------------
# 켜지면(VICA_ANSWER=on) 음성은 재개 제안을 쥐지 않고 미션에 보낸다 — 문장은 넘기지도 말하지도 않는다(미션이 MSG_RESUME_CONFIRM 을
# 묻는다). 그 질문의 "아니"는 미션이 받고 MSG_PAUSE_KEPT 를 말한다(두 목소리 금지). 꺼지면 지금 그대로(음성이 쥔 질문).
_DESTS = [DestinationData(id="elev", name="엘리베이터")]
_RESUME_Q = OpenQuestion(id="q7", kind="yesno", text=RESUME_CONFIRM, option_ids=(), opened_at=0.0)
_ASKED = [HumanMessage("다시 가자"), AIMessage(RESUME_CONFIRM)]


def test_resume_proposals_reach_the_mission_when_answers_are_on(monkeypatch):
    held = VicaIntent(intent="resume", need_confirm=True)
    assert should_forward_intent(held) is False
    monkeypatch.setenv("VICA_ANSWER", "on")
    assert should_forward_intent(held) is True


def test_p9_resume_proposal_goes_to_the_mission_without_a_sentence_when_on(monkeypatch):
    proposal = VicaIntent(intent="resume", need_confirm=True, reply=RESUME_CONFIRM)
    off = hand_over_question(proposal, _DESTS)
    assert (off.ask_text, off.ask_kind, off.reply) == (RESUME_CONFIRM, "yesno", "")    # 꺼짐 — 음성이 쥔 질문 그대로
    assert mission_asks_resume(proposal) is False
    monkeypatch.setenv("VICA_ANSWER", "on")
    on = hand_over_question(proposal, _DESTS)
    assert (on.intent, on.need_confirm, on.reply, on.ask_text, on.ask_kind) == ("resume", True, "", "", "")
    assert should_forward_intent(on) is True and mission_asks_resume(proposal) is True
    confirmed = VicaIntent(intent="resume", need_confirm=False, reply="")
    assert mission_asks_resume(confirmed) is False and hand_over_question(confirmed, _DESTS) is confirmed


def test_p9_voice_stays_quiet_while_the_missions_resume_question_is_open(monkeypatch):
    declined = VicaIntent(intent="unknown", reply=COMMAND_DECLINED)
    answer = VicaIntent(intent="question", reply="화장실은 2층에 있어요.")
    emergency = VicaIntent(intent="unknown", safety_flag="emergency", reply="")
    assert [voice_may_speak(i, _RESUME_Q) for i in (declined, answer, emergency)] == [True, True, True]   # 꺼짐 — 지금 그대로
    monkeypatch.setenv("VICA_ANSWER", "on")
    assert [voice_may_speak(i, _RESUME_Q) for i in (declined, answer, emergency)] == [False, True, True]


def test_p9_no_to_the_resume_question_leaves_the_voice_nothing_to_say_when_on(monkeypatch):
    off = _shortcut_intent("아니", _ASKED, _DESTS)
    assert (off.intent, off.reply) == ("unknown", COMMAND_DECLINED)                  # 꺼짐 — 지금 그대로
    monkeypatch.setenv("VICA_ANSWER", "on")
    on = _shortcut_intent("아니", _ASKED, _DESTS)
    assert (on.intent, on.reply, on.need_confirm) == ("deny", "", False)
    yes = _shortcut_intent("네", _ASKED, _DESTS)                                       # 승낙 단축은 그대로
    assert (yes.intent, yes.need_confirm, yes.reply) == ("resume", False, "")


def test_p6_resume_question_answers_carry_the_answer_field_when_on(monkeypatch):
    monkeypatch.setenv("VICA_ANSWER", "on")
    no = parse_intent("아니", _DESTS, history=_ASKED, question=_RESUME_Q)
    assert (no.intent, no.reply, no.answer, no.question_id) == ("deny", "", "no", "q7")
    yes = parse_intent("네", _DESTS, history=_ASKED, question=_RESUME_Q)
    assert (yes.intent, yes.need_confirm, yes.answer, yes.question_id) == ("resume", False, "yes", "q7")


def test_p9_held_resume_paths_are_skipped_when_answers_are_on():
    fn = _method("_publish_intent")
    (reask,) = _calls(fn, "reask_held_resume")
    assert ("not answer_field.on()", True) in _branch_tests(fn, reask)
    (carrier,) = _calls(fn, "held_resume_carrier")
    assert ("should_forward_intent(intent)", False) in _branch_tests(fn, carrier)   # 켜지면 should_forward 가 늘 참
    silence = _method("_on_listen_state_for_resume")
    gates = [n for n in ast.walk(silence) if isinstance(n, ast.If) and ast.unparse(n.test) == "answer_field.on()"]
    assert gates and isinstance(gates[0].body[0], ast.Return)
    assert gates[0].lineno < _calls(silence, "should_reask_resume_on_silence")[0].lineno
    assert gates[0].lineno > _calls(silence, "self._reservation.on_silence")[0].lineno   # 예약 대답 없음은 그대로


def test_p3_resume_proposal_is_not_written_to_history_when_on():
    """미션이 실제로 물었는지 음성은 모른다(미션 answer_field 꺼짐이면 묻지 않는다) — 글자 모드 기록에 "다시 출발할까요?"를
    적으면 다음 "다시 가자"가 확정 재개가 된다(_finalize 의 pending_command). 소리 모드는 미션이 실제로 말한 줄이 들어온다."""
    fn = _method("_publish_intent")
    ifs = [n for n in ast.walk(fn) if isinstance(n, ast.If) and ast.unparse(n.test) == "mission_asks_resume(intent)"]
    assert ifs and ast.unparse(ifs[0].body[0]) == "history_line = ''"
    (extend,) = [c for c in _calls(fn, "self._history.extend") if "AIMessage(history_line)" in ast.unparse(c)]
    assert ifs[0].lineno < extend.lineno

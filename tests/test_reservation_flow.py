"""예약 진행 상태(설계 2026-11-18 5.2·5.3)."""
import json
from datetime import datetime

import pytest

from src import reservation_phrases as p
from src.question_record import OpenQuestion
from src.reservation_flow import (FlowOutput, ReservationFlow, ReserveCall, created_event, start_gate, to_intent)
from src.reservation_store import Facility, ReservationStore
from src.schema import VicaIntent

NOW = datetime(2026, 11, 18, 14, 3, 21)        # 수요일
FACILITIES = [Facility("세미나실", ("세미나룸",), 2, "09:00", "18:00"), Facility("상담실", (), 2, "09:00", "18:00")]
DEMO = "김토니 님 이름으로 내일 오후 두 시부터 네 시까지 세미나실 예약할까요?"
YES = VicaIntent(intent="affirm", need_confirm=False)
NO = VicaIntent(intent="deny", need_confirm=False)


def q(text, kind="open"):
    return OpenQuestion(id="q1", kind=kind, text=text, option_ids=(), opened_at=0.0)


@pytest.fixture
def flow(tmp_path):
    return ReservationFlow(ReservationStore(tmp_path / "reservations.json"), FACILITIES, clock=lambda: NOW)


def call(flow, text, **kw):
    return flow.on_call(ReserveCall(**kw), text, "ok")


def first(flow, text="내일 오후 두 시에 세미나실 예약해 줘.", date="내일"):
    return call(flow, text, facility="세미나실", date=date, start_time="14:00")


def test_demo_script(flow):
    assert first(flow) == FlowOutput(p.ASK_NAME, "open")
    assert call(flow, "김토니", name="김토니") == FlowOutput(DEMO, "yesno")
    out = flow.on_general(YES)
    assert out.say == p.DONE and out.kind == "open"
    assert (out.saved.facility, out.saved.date, out.saved.start, out.saved.end, out.saved.name) == (
        "세미나실", "2026-11-19", "14:00", "16:00", "김토니")
    assert flow.store.list() == [out.saved] and not flow.active


def test_values_not_in_the_words_are_dropped(flow):
    out = call(flow, "세미나실 예약해 줘.", facility="세미나실", date="오늘", start_time="14:00", name="홍길동")
    assert out == FlowOutput(p.ASK_WHEN, "open")
    assert call(flow, "내일 오후 두 시", date="내일", start_time="14:00") == FlowOutput(p.ASK_NAME, "open")
    assert call(flow, "음", name="홍길동") == FlowOutput(p.ASK_NAME, "open")     # 글자에 없는 이름 — 다시 묻는다
    assert call(flow, "제 이름은 김토니예요.", name="김토니") == FlowOutput(DEMO, "yesno")


def test_the_models_time_is_used(flow):
    call(flow, "상담실 오늘 오후 네 시에 쓸 수 있어?", facility="상담실", date="오늘", start_time="16:00")
    out = call(flow, "이서연입니다.", name="이서연")
    assert out.say == "이서연 님 이름으로 오늘 오후 네 시부터 여섯 시까지 상담실 예약할까요?"


def test_end_time_only_when_spoken(flow):
    call(flow, "내일 두 시부터 세 시까지 세미나실 예약해 줘.", facility="세미나실", date="내일",
         start_time="14:00", end_time="15:00")
    assert "오후 두 시부터 세 시까지" in call(flow, "김토니", name="김토니").say
    flow.drop()
    call(flow, "모레 오후 두 시에 한 시간만 세미나실 쓸게요.", facility="세미나실", date="모레",
         start_time="14:00", end_time="15:00")
    assert "오후 두 시부터 세 시까지" in call(flow, "김토니", name="김토니").say
    flow.drop()
    call(flow, "내일 오후 두 시에 세미나실 예약해 줘.", facility="세미나실", date="내일", start_time="14:00", end_time="17:00")
    assert "오후 두 시부터 네 시까지" in call(flow, "김토니", name="김토니").say     # 지어낸 끝은 버리고 기본 2시간


def test_empty_call_asks_facility_first(flow):
    assert call(flow, "예약 좀 하려고.") == FlowOutput(p.ASK_WHERE, "open")
    assert call(flow, "세미나실이요.", facility="세미나실") == FlowOutput(p.ASK_WHEN, "open")


def test_taken_slot_then_another_time(flow):
    flow.store.add("세미나실", "2026-11-19", "14:00", "16:00", "이서연", NOW)
    assert first(flow) == FlowOutput(p.TAKEN, "open")
    assert call(flow, "오후 네 시", start_time="16:00") == FlowOutput(p.ASK_NAME, "open")   # 날짜(내일)는 쥐고 있다
    assert "오후 네 시부터 여섯 시까지" in call(flow, "김토니", name="김토니").say


def test_out_of_hours_and_past_are_refused(flow):
    late = call(flow, "내일 오후 다섯 시에 세미나실 예약해 줘.", facility="세미나실", date="내일", start_time="17:00")
    assert late == FlowOutput(p.BAD_TIME, "open")
    flow.drop()
    past = call(flow, "오늘 오전 아홉 시에 세미나실 예약해 줘.", facility="세미나실", date="오늘", start_time="09:00")
    assert past == FlowOutput(p.BAD_TIME, "open") and flow.store.list() == []


def test_gate(flow):
    assert flow.on_call(ReserveCall(facility="세미나실"), "세미나실 예약해 줘.", "busy") == FlowOutput(p.UNAVAILABLE, "")
    assert flow.on_call(ReserveCall(facility="세미나실"), "세미나실 예약해 줘.", "mute") is None
    assert not flow.active
    first(flow)
    assert flow.on_call(ReserveCall(name="김토니"), "김토니", "busy") == FlowOutput(DEMO, "yesno")   # 진행 중이면 문을 다시 보지 않는다
    assert start_gate("idle", None, "") == "ok" and start_gate("idle", q("네?"), "") == "ok"
    assert start_gate("idle", q("화장실로 안내해드릴까요?", "confirm"), "") == "busy"
    assert start_gate("idle", q(DEMO, "yesno"), DEMO) == "ok"
    assert start_gate("navigating", None, "") == "busy" and start_gate("", None, "") == "busy"
    assert start_gate("awaiting_user", None, "") == "mute" and start_gate("patrolling", None, "") == "mute"


def test_confirm_no_and_second_yes_does_not_save_twice(flow):
    first(flow)
    call(flow, "김토니", name="김토니")
    assert flow.on_general(YES).saved is not None
    assert flow.on_general(YES) is None and len(flow.store.list()) == 1
    assert first(flow, "모레 오후 두 시에 세미나실 예약해 줘.", date="모레") == FlowOutput(p.ASK_NAME, "open")
    assert call(flow, "김토니", name="김토니").kind == "yesno"
    assert flow.on_general(NO) == FlowOutput(p.NOT_RESERVED, "") and not flow.active and len(flow.store.list()) == 1


def test_save_failure_says_unavailable_not_done(flow):
    first(flow)
    call(flow, "김토니", name="김토니")
    flow.store.path.write_text('[{"id": "r1", "facil', encoding="utf-8")
    assert flow.on_general(YES) == FlowOutput(p.UNAVAILABLE, "") and not flow.active


def test_general_result_while_active(flow):
    first(flow)                                                   # 이름을 기다리는 중
    for kind in ("unknown", "question", "clarify", "affirm"):
        assert flow.on_general(VicaIntent(intent=kind, need_confirm=False)) is None and flow.active
    assert flow.on_general(VicaIntent(intent="navigate", destination_candidate="화장실")) is None and not flow.active
    first(flow)
    assert flow.on_general(VicaIntent(intent="cancel", need_confirm=False)) == FlowOutput(p.NOT_RESERVED, "")


def test_question_closing_drops_the_draft(flow):
    first(flow)
    flow.on_question_record("")                 # 앞 질문("네?")이 닫힘 — 내 질문은 아직 열린 적 없다
    assert flow.active
    flow.on_question_record(p.ASK_NAME)         # 미션이 내 질문을 열었다
    call(flow, "김토니", name="김토니")          # 새 질문(②)을 넘겼다
    flow.on_question_record("")                 # ①이 닫힘 — ②는 아직 안 열렸다
    assert flow.active
    flow.on_question_record(DEMO)
    flow.on_question_record("네?")              # 대답 없이 다시 불렀다
    assert not flow.active


def test_same_question_asked_again_survives_the_old_one_closing(flow):
    first(flow)
    flow.on_question_record(p.ASK_NAME)
    assert call(flow, "음", name="홍길동") == FlowOutput(p.ASK_NAME, "open")   # 같은 문장을 다시 넘겼다
    flow.on_question_record("")                 # 앞의 ①이 닫힘
    assert flow.active
    flow.on_question_record(p.ASK_NAME)         # 새 ①이 열림
    flow.on_question_record("")
    assert not flow.active


def test_same_call_twice_asks_the_same_question(flow):
    assert first(flow) == first(flow) == FlowOutput(p.ASK_NAME, "open")


def test_situation_line(flow):
    assert flow.situation_line() == ""
    first(flow)
    line = flow.situation_line()
    assert line.startswith("[진행 중인 예약]") and "세미나실" in line and "내일" in line and "14:00" in line
    assert "(아직 없음)" in line and p.ASK_NAME in line


def test_to_intent_and_event(flow):
    ask = to_intent(FlowOutput(p.ASK_NAME, "open"))
    assert (ask.intent, ask.reply, ask.ask_text, ask.ask_kind, ask.need_confirm) == ("clarify", "", p.ASK_NAME, "open", False)
    say = to_intent(FlowOutput(p.NOT_RESERVED, ""))
    assert (say.intent, say.reply, say.ask_text, say.need_confirm) == ("question", p.NOT_RESERVED, "", False)
    res = flow.store.add("세미나실", "2026-11-19", "14:00", "16:00", "김토니", NOW)
    event = json.loads(created_event(res, NOW))
    assert event["event"] == "reservation_created" and event["id"] == res.id and event["name"] == "김토니"
    assert set(event) == {"event", "id", "facility", "date", "start", "end", "name", "source", "timestamp"}

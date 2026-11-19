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
DEMO = "김민수 님 이름으로 내일 오후 두 시부터 네 시까지 세미나실 예약할까요?"
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


def opened(flow):
    """미션이 방금 넘긴 질문을 열었다(질문 기록이 왔다). 승낙은 그 뒤에만 저장으로 이어진다."""
    flow.on_question_record(flow.asked_text)


def test_demo_script(flow):
    assert first(flow) == FlowOutput(p.ASK_NAME, "open")
    assert call(flow, "김민수", name="김민수") == FlowOutput(DEMO, "yesno")
    opened(flow)
    out = flow.on_general(YES)
    assert out.say == p.DONE and out.kind == "open"
    assert (out.saved.facility, out.saved.date, out.saved.start, out.saved.end, out.saved.name) == (
        "세미나실", "2026-11-19", "14:00", "16:00", "김민수")
    assert flow.store.list() == [out.saved] and not flow.active


def test_values_not_in_the_words_are_dropped(flow):
    out = call(flow, "세미나실 예약해 줘.", facility="세미나실", date="오늘", start_time="14:00", name="홍길동")
    assert out == FlowOutput(p.ASK_WHEN, "open")
    assert call(flow, "내일 오후 두 시", date="내일", start_time="14:00") == FlowOutput(p.ASK_NAME, "open")
    assert call(flow, "음", name="홍길동") == FlowOutput(p.ASK_NAME, "open")     # 글자에 없는 이름 — 다시 묻는다
    assert call(flow, "제 이름은 김민수예요.", name="김민수") == FlowOutput(DEMO, "yesno")


@pytest.mark.parametrize("name, said", [
    ("이서연", "이서연입니다."), ("박준호", "박준호요."), ("정현우", "정현우라고 합니다."), ("강민지", "제 이름은 강민지예요."),
])
def test_any_ordinary_name_goes_into_the_confirm_sentence(flow, name, said):
    first(flow)
    out = call(flow, said, name=name)
    assert out == FlowOutput(f"{name} 님 이름으로 내일 오후 두 시부터 네 시까지 세미나실 예약할까요?", "yesno")
    opened(flow)
    assert flow.on_general(YES).saved.name == name


def test_the_models_time_is_used(flow):
    call(flow, "상담실 오늘 오후 네 시에 쓸 수 있어?", facility="상담실", date="오늘", start_time="16:00")
    out = call(flow, "이서연입니다.", name="이서연")
    assert out.say == "이서연 님 이름으로 오늘 오후 네 시부터 여섯 시까지 상담실 예약할까요?"


def test_end_time_only_when_spoken(flow):
    call(flow, "내일 두 시부터 세 시까지 세미나실 예약해 줘.", facility="세미나실", date="내일",
         start_time="14:00", end_time="15:00")
    assert "오후 두 시부터 세 시까지" in call(flow, "김민수", name="김민수").say
    flow.drop()
    call(flow, "모레 오후 두 시에 한 시간만 세미나실 쓸게요.", facility="세미나실", date="모레",
         start_time="14:00", end_time="15:00")
    assert "오후 두 시부터 세 시까지" in call(flow, "김민수", name="김민수").say
    flow.drop()
    call(flow, "내일 오후 두 시에 세미나실 예약해 줘.", facility="세미나실", date="내일", start_time="14:00", end_time="17:00")
    assert "오후 두 시부터 네 시까지" in call(flow, "김민수", name="김민수").say     # 지어낸 끝은 버리고 기본 2시간


def test_empty_call_asks_facility_first(flow):
    assert call(flow, "예약 좀 하려고.") == FlowOutput(p.ASK_WHERE, "open")
    assert call(flow, "세미나실이요.", facility="세미나실") == FlowOutput(p.ASK_WHEN, "open")


def test_taken_slot_then_another_time(flow):
    flow.store.add("세미나실", "2026-11-19", "14:00", "16:00", "이서연", NOW)
    assert first(flow) == FlowOutput(p.TAKEN, "open")
    assert call(flow, "오후 네 시", start_time="16:00") == FlowOutput(p.ASK_NAME, "open")   # 날짜(내일)는 쥐고 있다
    assert "오후 네 시부터 여섯 시까지" in call(flow, "김민수", name="김민수").say


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
    assert flow.on_call(ReserveCall(name="김민수"), "김민수", "busy") == FlowOutput(DEMO, "yesno")   # 진행 중이면 문을 다시 보지 않는다
    assert start_gate("idle", None, "") == "ok" and start_gate("idle", q("네?"), "") == "ok"
    assert start_gate("idle", q("화장실로 안내해드릴까요?", "confirm"), "") == "busy"
    assert start_gate("idle", q(DEMO, "yesno"), DEMO) == "ok"
    assert start_gate("navigating", None, "") == "busy" and start_gate("", None, "") == "busy"
    assert start_gate("awaiting_user", None, "") == "mute" and start_gate("patrolling", None, "") == "mute"


def test_confirm_no_and_second_yes_does_not_save_twice(flow):
    first(flow)
    call(flow, "김민수", name="김민수")
    opened(flow)
    assert flow.on_general(YES).saved is not None
    assert flow.on_general(YES) is None and len(flow.store.list()) == 1
    assert first(flow, "모레 오후 두 시에 세미나실 예약해 줘.", date="모레") == FlowOutput(p.ASK_NAME, "open")
    assert call(flow, "김민수", name="김민수").kind == "yesno"
    assert flow.on_general(NO) == FlowOutput(p.NOT_RESERVED, "") and not flow.active and len(flow.store.list()) == 1


def test_save_failure_says_unavailable_not_done(flow):
    first(flow)
    call(flow, "김민수", name="김민수")
    flow.store.path.write_text('[{"id": "r1", "facil', encoding="utf-8")
    opened(flow)
    assert flow.on_general(YES) == FlowOutput(p.UNAVAILABLE, "") and not flow.active


def test_general_result_while_active(flow):
    """예약 단계에서 못 알아들은 말·질문·엉뚱한 승낙은 미션에 가지 않는다 — 같은 질문을 한 번 다시 묻는다(11-19 사용자)."""
    for kind in ("unknown", "question", "clarify", "affirm"):
        first(flow)                                               # 이름을 기다리는 중
        opened(flow)
        assert flow.on_general(VicaIntent(intent=kind, need_confirm=False)) == FlowOutput(p.ASK_NAME, "open")
        assert flow.active
        flow.drop()
    first(flow)
    assert flow.on_general(VicaIntent(intent="navigate", destination_candidate="화장실")) is None and not flow.active
    first(flow)
    assert flow.on_general(VicaIntent(intent="cancel", need_confirm=False)) == FlowOutput(p.NOT_RESERVED, "")


def test_closed_question_keeps_the_draft_but_another_question_drops_it(flow):
    first(flow)
    flow.on_question_record("")                 # 앞 질문("네?")이 닫힘 — 내 질문은 아직 열린 적 없다
    assert flow.active
    flow.on_question_record(p.ASK_NAME)         # 미션이 내 질문을 열었다
    flow.on_question_record("")                 # 내 질문이 닫혔다(침묵·대답) — 버리지 않는다. 다시 물을지는 on_silence·on_general 몫
    assert flow.active
    call(flow, "김민수", name="김민수")          # 새 질문(②)을 넘겼다
    flow.on_question_record(DEMO)
    flow.on_question_record("네?")              # 대답 없이 다시 불렀다 — 다른 질문이 열림
    assert not flow.active


def test_same_question_asked_again_survives_the_old_one_closing(flow):
    first(flow)
    flow.on_question_record(p.ASK_NAME)
    assert call(flow, "음", name="홍길동") == FlowOutput(p.ASK_NAME, "open")   # 같은 문장을 다시 넘겼다
    flow.on_question_record("")                 # 앞의 ①이 닫힘
    assert flow.active
    flow.on_question_record(p.ASK_NAME)         # 새 ①이 열림
    flow.on_question_record("")
    assert flow.active                          # 닫힘만으로는 버리지 않는다(11-19)


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
    res = flow.store.add("세미나실", "2026-11-19", "14:00", "16:00", "김민수", NOW)
    event = json.loads(created_event(res, NOW))
    assert event["event"] == "reservation_created" and event["id"] == res.id and event["name"] == "김민수"
    assert set(event) == {"event", "id", "facility", "date", "start", "end", "name", "source", "timestamp"}


# ----- 해석 결과 가르기(작업 6) ------------------------------------------
from src.reservation_flow import handle_result  # noqa: E402


def test_routing(flow):
    nav = VicaIntent(intent="navigate", destination_candidate="화장실", matched_destination_id="restroom")
    assert handle_result(None, nav, "화장실로 가자.", gate="ok", tools_used=True) == (None, nav)
    out, passed = handle_result(flow, ReserveCall(facility="세미나실", date="내일", start_time="14:00"),
                                "내일 오후 두 시에 세미나실 예약해 줘.", gate="ok", tools_used=True)
    assert out == FlowOutput(p.ASK_NAME, "open") and passed is None
    out, passed = handle_result(flow, ReserveCall(name="김민수"), "김민수", gate="ok", tools_used=True)
    assert out == FlowOutput(DEMO, "yesno") and passed is None
    opened(flow)
    out, passed = handle_result(flow, YES, "그래", gate="ok", tools_used=True)
    assert out.say == p.DONE and passed is None                   # 예약이 받은 승낙은 미션에 보내지 않는다
    assert handle_result(flow, nav, "화장실로 가자.", gate="ok", tools_used=True) == (None, nav)


def test_reservation_word_without_tools_is_refused(flow):
    nav = VicaIntent(intent="navigate", destination_candidate="세미나실")
    assert handle_result(flow, nav, "세미나실 예약해 줘.", gate="ok", tools_used=False) == (FlowOutput(p.UNAVAILABLE, ""), None)
    assert handle_result(flow, nav, "세미나실 예약해 줘.", gate="mute", tools_used=False) == (None, nav)   # 가로채지 않는다
    assert handle_result(flow, nav, "예약은 됐고 화장실 가자.", gate="ok", tools_used=True) == (None, nav)
    assert handle_result(flow, YES, "네 예약이요", gate="ok", tools_used=False) == (None, YES)


def test_emergency_and_switched_off_node_pass_straight_through(flow):
    stop = VicaIntent(intent="unknown", need_confirm=False, safety_flag="emergency")
    assert handle_result(flow, stop, "예약 멈춰", gate="ok", tools_used=False) == (None, stop)
    # 흐름이 없는 노드(스위치 꺼짐)에 예약 호출이 오면 의도인 척 내보내지 않는다 — 아무것도 내지 않는다.
    assert handle_result(None, ReserveCall(facility="세미나실"), "세미나실 예약해 줘.", gate="ok", tools_used=True) == (None, None)


def test_decline_inside_a_draft_is_kept_from_the_mission(flow):
    first(flow)
    out, passed = handle_result(flow, NO, "아니", gate="ok", tools_used=True)
    assert out == FlowOutput(p.NOT_RESERVED, "") and passed is None and not flow.active


def test_snapshot_and_situation_join(flow):
    from src.reservation_flow import join_situation
    assert flow.snapshot() == {"active": False, "facility": "", "date": "", "start": "", "end": "", "name": "", "asked": ""}
    first(flow)
    assert flow.snapshot() == {"active": True, "facility": "세미나실", "date": "2026-11-19", "start": "14:00",
                               "end": "16:00", "name": "", "asked": p.ASK_NAME}
    assert join_situation("- 대화 단계: 쉼\n", "") == "- 대화 단계: 쉼\n"
    assert join_situation("- 대화 단계: 쉼\n", "[진행 중인 예약] 가") == "- 대화 단계: 쉼\n[진행 중인 예약] 가\n"
    assert join_situation("", "[진행 중인 예약] 가") == "[진행 중인 예약] 가\n"


# ----- 독립 검토(11-19) 수리 --------------------------------------------------
def test_yes_before_the_mission_opened_the_confirm_question_does_not_save(flow):
    """I-1: 확인 질문 ②가 미션에 닿지 않았으면(질문 기록이 온 적 없음) '네'로 저장하지 않는다."""
    first(flow)
    call(flow, "김민수", name="김민수")                      # ②를 넘겼지만 미션은 아직 열지 않았다
    out, passed = handle_result(flow, YES, "네", gate="ok", tools_used=True)
    assert out == FlowOutput(DEMO, "yesno") and passed is None and flow.store.list() == []   # 저장 대신 확인을 다시 묻는다
    opened(flow)
    out, passed = handle_result(flow, YES, "네", gate="ok", tools_used=True)
    assert out.say == p.DONE and passed is None and len(flow.store.list()) == 1


def test_question_the_mission_never_opened_expires(tmp_path):
    """I-1: 넘긴 질문이 열리지 않은 채 시간이 지나면 진행 중인 예약을 버린다 — 다음 사람에게 붙지 않게."""
    from datetime import timedelta

    from src.reservation_flow import UNSEEN_DROP_SEC
    clock = [NOW]
    flow = ReservationFlow(ReservationStore(tmp_path / "r.json"), FACILITIES, clock=lambda: clock[0])
    first(flow)
    clock[0] = NOW + timedelta(seconds=UNSEEN_DROP_SEC - 1)
    assert flow.situation_line() and flow.active
    clock[0] = NOW + timedelta(seconds=UNSEEN_DROP_SEC + 1)
    assert flow.situation_line() == "" and not flow.active
    from src.reservation_flow import IDLE_DROP_SEC
    first(flow)                                              # 열린 질문도 오래 아무 일이 없으면 말없이 버린다
    opened(flow)
    asked = clock[0]
    clock[0] = asked + timedelta(seconds=IDLE_DROP_SEC - 1)
    assert flow.situation_line() and flow.active
    clock[0] = asked + timedelta(seconds=IDLE_DROP_SEC + 1)
    assert flow.on_general(YES) is None and not flow.active


def test_reservation_word_without_tools_only_blocks_reservation_requests(flow):
    """I-3: 도구 없는 경로의 '예약' 낱말 규칙이 예약이 아닌 요청을 삼키지 않는다."""
    go = VicaIntent(intent="navigate", destination_candidate="화장실", matched_destination_id="restroom")
    assert handle_result(flow, go, "예약했는데 화장실 가자.", gate="ok", tools_used=False) == (None, go)
    for kind in ("wait", "finish", "help"):
        said = VicaIntent(intent=kind)
        assert handle_result(flow, said, "예약 시간까지 좀 기다려 줘.", gate="ok", tools_used=False) == (None, said)
    seminar = VicaIntent(intent="navigate", destination_candidate="세미나실")
    assert handle_result(flow, seminar, "세미나룸 예약해 줘.", gate="ok", tools_used=False) == (FlowOutput(p.UNAVAILABLE, ""), None)
    ask = VicaIntent(intent="unknown")
    assert handle_result(flow, ask, "예약 좀 하고 싶은데.", gate="ok", tools_used=False) == (FlowOutput(p.UNAVAILABLE, ""), None)
    # 말하지 않는 상태(접근 질문·순찰)에서는 가로채지 않는다 — 그 말은 예전처럼 미션에 간다
    assert handle_result(flow, seminar, "세미나실 예약해 줘.", gate="mute", tools_used=False) == (None, seminar)


NOISE = VicaIntent(intent="unknown", need_confirm=False)


def test_question_twice_then_cancelled(flow):
    """I-2(11-19 사용자): 한 질문은 두 번까지 묻는다. 그래도 못 알아들으면 "예약이 취소되었습니다."."""
    assert first(flow) == FlowOutput(p.ASK_NAME, "open")
    opened(flow)
    out, passed = handle_result(flow, NOISE, "어 음", gate="ok", tools_used=True)
    assert out == FlowOutput(p.ASK_NAME, "open") and passed is None and flow.active     # 미션에 보내지 않고 한 번 더
    flow.on_question_record("")                              # 미션이 앞 질문을 닫고
    opened(flow)                                             # 같은 질문을 다시 열었다
    out, passed = handle_result(flow, NOISE, "음", gate="ok", tools_used=True)
    assert out == FlowOutput(p.CANCELLED, "") and passed is None
    assert not flow.active and flow.store.list() == []


def test_retry_count_starts_over_with_each_new_question(flow):
    first(flow)
    opened(flow)
    assert flow.on_general(NOISE) == FlowOutput(p.ASK_NAME, "open")           # ① 두 번째
    opened(flow)
    assert call(flow, "김민수", name="김민수") == FlowOutput(DEMO, "yesno")    # 대답함 → ②
    opened(flow)
    assert flow.on_general(NOISE) == FlowOutput(DEMO, "yesno")                # ②는 새 질문이라 다시 한 번
    opened(flow)
    assert flow.on_general(YES).say == p.DONE


def test_reserve_call_without_a_new_value_counts_as_not_understood(flow):
    first(flow)
    opened(flow)
    assert call(flow, "음", name="홍길동") == FlowOutput(p.ASK_NAME, "open")    # 지어낸 이름 — 받은 값이 없다
    opened(flow)
    assert call(flow, "어", name="홍길동") == FlowOutput(p.CANCELLED, "") and not flow.active


def test_silence_follows_the_same_rule(flow):
    """듣기 창이 빈손으로 닫혀도(대답 없음·잡음 기각) 같은 규칙이다."""
    first(flow)
    assert flow.on_silence("ok") is None and flow.active     # 미션이 아직 묻지 않았다 — 그 질문의 침묵이 아니다
    opened(flow)
    assert flow.on_silence("ok") == FlowOutput(p.ASK_NAME, "open")
    opened(flow)
    assert flow.on_silence("ok") == FlowOutput(p.CANCELLED, "") and not flow.active
    assert flow.on_silence("ok") is None                     # 진행 중인 예약이 없으면 아무것도 하지 않는다
    first(flow)
    opened(flow)
    assert flow.on_silence("busy") is None and not flow.active   # 미션이 쉬는 상태를 떠났다 — 말없이 버린다


def test_leaving_idle_ends_the_stage_silently(flow):
    """진행 중에 미션이 쉬는 상태를 떠났거나 다른 질문이 열렸으면(gate) 예약 단계는 끝났다 — 그 말은 평소대로 간다."""
    first(flow)
    opened(flow)
    out, passed = handle_result(flow, NOISE, "어", gate="busy", tools_used=True)
    assert out is None and passed == NOISE and not flow.active
    first(flow)
    opened(flow)
    stop = VicaIntent(intent="unknown", need_confirm=False, safety_flag="emergency")
    assert flow.on_general(stop) is None and flow.active     # 긴급은 건드리지 않는다


def test_finish_inside_a_draft_means_no_reservation(flow):
    """11-19 gpt 재생 F4: 이름 질문에 "아니, 됐어."를 LLM 이 finish 로 읽었다. 예약 단계의 finish 는 예약을 그만두는 말이다."""
    first(flow)
    opened(flow)
    out, passed = handle_result(flow, VicaIntent(intent="finish", need_confirm=False), "아니, 됐어.", gate="ok", tools_used=True)
    assert out == FlowOutput(p.NOT_RESERVED, "") and passed is None and not flow.active

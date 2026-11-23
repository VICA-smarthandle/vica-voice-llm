"""예약 묶음 재생의 채점(설계 2026-11-18 7절). LLM 없이 가짜 해석기로 시험한다."""
from src import reservation_phrases as p
from src.ledger_view import FLOOR_LABEL
from src.reservation_flow import ReserveCall
from src.schema import VicaIntent
from tools.reserve_cases import CASES
from tools.reserve_replay import run_case, summarize


def by_id(case_id):
    return next(c for c in CASES if c.id == case_id)


def fake(result):
    return lambda text, history, situation, *_: (result, 0.5)


def test_case_set_shape():
    kinds = [c.id[0] for c in CASES]
    assert len(CASES) == 43 and len({c.id for c in CASES}) == 43
    assert kinds.count("R") == 12 and kinds.count("E") == 3 and kinds.count("N") == 13 and kinds.count("F") == 15
    assert all(c.setup for c in CASES if c.id.startswith("F")) and not any(c.setup for c in CASES if c.id[0] in "REN")


def test_summary_counts():
    rows = [{"id": "R1", "ok": True, "reserve_called": True, "wall_s": 1.0},
            {"id": "E1", "ok": False, "reserve_called": True, "wall_s": 1.0},
            {"id": "N3", "ok": False, "reserve_called": True, "wall_s": 1.0},
            {"id": "F1", "ok": True, "reserve_called": True, "wall_s": 1.0},
            {"id": "F8", "ok": False, "reserve_called": False, "wall_s": 1.0}]
    s = summarize(rows)
    assert (s["first_ok"], s["first_total"], s["stray_reserve"], s["follow_ok"], s["follow_total"]) == (1, 2, 1, 1, 2)


def test_first_request_is_graded_by_what_the_flow_keeps():
    case = by_id("R1")
    row = run_case(case, fake(ReserveCall(facility="세미나실", date="내일", start_time="14:00")))
    assert row["ok"] and row["reserve_called"] and row["say"] == p.ASK_NAME
    row = run_case(case, fake(ReserveCall(facility="세미나실", date="내일", start_time="16:00")))
    assert not row["ok"] and "start" in row["why"]
    row = run_case(case, fake(VicaIntent(intent="navigate", destination_candidate="세미나실")))
    assert not row["ok"] and not row["reserve_called"]


def test_spoken_end_time_is_part_of_the_answer():
    case = by_id("E1")
    assert run_case(case, fake(ReserveCall(facility="세미나실", date="내일", start_time="14:00", end_time="15:00")))["ok"]
    assert not run_case(case, fake(ReserveCall(facility="세미나실", date="내일", start_time="14:00")))["ok"]


def test_follow_up_gets_the_draft_line_and_the_asked_question():
    seen = {}

    def interpret(text, history, situation, *_):
        seen.update(text=text, history=history, situation=situation)
        return ReserveCall(name="김민수"), 0.1

    row = run_case(by_id("F1"), interpret)
    assert row["ok"] and row["say"].startswith("김민수 님 이름으로")
    assert "[진행 중인 예약]" in seen["situation"] and FLOOR_LABEL in seen["situation"]
    assert seen["history"][-1].content == p.ASK_NAME and seen["history"][-1].type == "ai"
    assert seen["text"] == "김민수"


def test_confirm_yes_must_end_in_a_saved_reservation():
    yes = VicaIntent(intent="affirm", need_confirm=False)
    assert run_case(by_id("F8"), fake(yes))["ok"]
    assert not run_case(by_id("F8"), fake(ReserveCall()))["ok"]
    assert not run_case(by_id("F9"), fake(yes))["ok"]


def test_other_requests_must_not_call_the_reserve_tool():
    row = run_case(by_id("N3"), fake(ReserveCall(facility="세미나실")))
    assert row["reserve_called"] and not row["ok"]
    nav = VicaIntent(intent="navigate", destination_candidate="세미나실", matched_destination_id="seminar")
    assert run_case(by_id("N3"), fake(nav))["ok"]


def test_moving_on_drops_the_draft_and_passes_the_intent():
    nav = VicaIntent(intent="navigate", destination_candidate="화장실", matched_destination_id="restroom")
    row = run_case(by_id("F5"), fake(nav))
    assert row["ok"] and row["passed"] == "navigate"
    assert not run_case(by_id("F5"), fake(VicaIntent(intent="unknown")))["ok"]


def test_llm_failure_is_an_error_row_not_a_wrong_answer():
    from src.replies import LLM_UNAVAILABLE
    row = run_case(by_id("R1"), fake(VicaIntent(intent="unknown", reply=LLM_UNAVAILABLE)))
    assert row["error"] and not row["ok"]
    assert summarize([row])["errors"] == 1


def test_flow_checks_values_against_the_heard_text():
    """Realtime 재생: 값의 근거는 넣은 글자가 아니라 모델이 적은 들린 말(heard_text)에서 찾는다 — 운영과 같다."""
    assert run_case(by_id("F1"), lambda t, h, s, *_: (ReserveCall(name="김민수"), 0.3, "김민수"))["ok"]
    row = run_case(by_id("F1"), lambda t, h, s, *_: (ReserveCall(name="김민수"), 0.3, "네"))
    assert not row["ok"] and row["heard"] == "네"


def test_interpreter_exception_is_an_error_row():
    def boom(text, history, situation, *_):
        raise TimeoutError("Realtime 응답 8초 초과")

    row = run_case(by_id("R1"), boom)
    assert row["error"] and not row["ok"] and "TimeoutError" in row["why"]


def test_realtime_replay_gets_the_tool_trail():
    seen = {}

    def interpret(text, history, situation, trail):
        seen["trail"] = trail
        return ReserveCall(name="김민수"), 0.2

    run_case(by_id("F1"), interpret)
    assert [t["asked"] for t in seen["trail"]] == [p.ASK_NAME]

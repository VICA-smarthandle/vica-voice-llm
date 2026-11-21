"""재생 도구 — 경로별 기준 점수 (설계 2026-11-14 5절, 계획 Task 4). LLM 없이 가짜 interpret 로 시험한다."""
from __future__ import annotations

from datetime import datetime

import pytest

from src.replies import LLM_UNAVAILABLE
from src.schema import DestinationData, VicaIntent
from tools import answer_replay
from tools.answer_replay import call_with_retry, clip_for, parsed_text_for, path_env, replay, text_for


def test_call_with_retry_retries_rate_limit_then_succeeds():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("Realtime 응답 failed: rate_limit_exceeded")
        return "ok"
    assert call_with_retry(flaky, tries=3, wait_s=0) == "ok" and len(calls) == 3


def test_call_with_retry_gives_up_and_other_errors_pass_through():
    def always():
        raise RuntimeError("rate_limit_exceeded")
    with pytest.raises(RuntimeError):
        call_with_retry(always, tries=2, wait_s=0)

    def other():
        raise ValueError("bad args")
    with pytest.raises(ValueError):
        call_with_retry(other, tries=3, wait_s=0)


@pytest.fixture
def dests():
    return [DestinationData(id="d1", name="사회복지"), DestinationData(id="d2", name="화장실")]


@pytest.fixture
def rows_labeled():
    base = {"date": "11-11", "time": "21:00:00", "dest_yaml": "x", "stt": ""}
    return [
        {**base, "ts": "1.0", "kind": "yesno", "question": "여기서 대기할까요?", "answer_text": "네",
         "label_answer": "yes", "label_value": ""},
        {**base, "ts": "2.0", "kind": "minutes", "question": "몇 분쯤 걸리실까요?", "answer_text": "십 분",
         "label_answer": "minutes", "label_value": "10"},
        {**base, "ts": "3.0", "kind": "confirm", "question": "사회복지로 안내해드릴까요?", "answer_text": "아니",
         "label_answer": "no", "label_value": ""},
    ]


def test_replay_counts_and_kinds(rows_labeled, dests):
    out = replay(rows_labeled, lambda row: (VicaIntent(intent="affirm"), 1.2), lambda _p: dests)
    assert set(out["counts"]) <= {"ok", "reask", "wrong", "skip", "error"} and out["p90"] >= 1.2
    assert out["counts"]["ok"] == 1 and out["counts"]["wrong"] == 2   # 네=ok, 십 분·아니 → yes 는 wrong
    assert out["by_kind"]["yesno"]["ok"] == 1


def test_missing_destination_file_is_skip(rows_labeled):
    out = replay(rows_labeled[:1], lambda r: (VicaIntent(intent="affirm"), 1.0), lambda _p: None)
    assert out["rows"][0]["status"] == "skip"


def test_llm_unavailable_is_error_and_stops_after_five(rows_labeled, dests):
    down = lambda r: (VicaIntent(intent="unknown", reply=LLM_UNAVAILABLE), 0.1)  # noqa: E731
    out = replay(rows_labeled * 3, down, lambda _p: dests)
    assert out["counts"]["error"] == 5 and out["stopped"] is True


def test_exception_in_interpret_is_error(rows_labeled, dests):
    def boom(_row):
        raise TimeoutError("realtime")
    out = replay(rows_labeled[:1], boom, lambda _p: dests)
    assert out["rows"][0]["status"] == "error"


def test_no_clip_is_skip(rows_labeled, dests):
    out = replay(rows_labeled[:1], lambda r: (None, 0.0), lambda _p: dests)
    assert out["rows"][0]["status"] == "skip"


def test_empty_answer_never_calls_llm(dests):
    called = []
    row = {"ts": "1.0", "answer_text": "", "kind": "yesno", "question": "여기서 대기할까요?", "label_answer": "unclear",
           "label_value": "", "dest_yaml": "x"}
    out = replay([row], lambda r: called.append(r) or (None, 0.0), lambda _p: dests)
    assert called == [] and out["rows"][0]["status"] == "ok"


def test_clip_for_matches_time_and_text(tmp_path):
    clips = [{"time": "2026-11-11T21:47:44.346", "file": "a.wav", "result": "text", "text": "그래."}]
    ts = datetime(2026, 11, 11, 21, 47, 45, 800000).timestamp()
    assert clip_for({"ts": ts, "stt": "그래."}, clips, tmp_path) == tmp_path / "a.wav"
    assert clip_for({"ts": ts + 10, "stt": "그래."}, clips, tmp_path) is None
    assert clip_for({"ts": ts, "stt": "아니."}, clips, tmp_path) is None


def test_text_path_reads_the_transcript_not_realtime_heard():
    """검토 I5: 글자 경로의 입력은 받아쓴 글자(설계 5절). 없으면 들은 글자."""
    assert text_for({"answer_text": "네", "stt": "바이바이."}) == "바이바이."
    assert text_for({"answer_text": "네", "stt": ""}) == "네"


def test_path_env_matches_deployed_switches():
    """검토 C1: 오프라인 Mi:dm 은 로컬 규칙 켬(⑬ 와 같게), gpt 는 ⑫ 예비처럼 끔."""
    midm = path_env("midm")
    assert midm["VICA_LOCAL_RULES"] == "1" and midm["VICA_INTENT_INPUT"] == "text"
    assert midm["VICA_LLM_PROVIDER"] == "ollama" and midm["VICA_LLM_FALLBACK_MODEL"] == ""
    gpt = path_env("gpt")
    assert gpt["VICA_LOCAL_RULES"] == "0" and gpt["VICA_LLM_PROVIDER"] == "openai"
    assert path_env("realtime")["VICA_INTENT_INPUT"] == "audio"
    assert path_env("midm", local_rules=False)["VICA_LOCAL_RULES"] == "0"


def test_mismatch_is_counted_apart_from_wrong(dests):
    row = {"ts": "1.0", "answer_text": "아니야 마이너스", "kind": "yesno", "question": "여기서 대기할까요?",
           "label_answer": "not_for_robot", "label_value": "", "dest_yaml": "x"}
    out = replay([row], lambda r: (VicaIntent(intent="question"), 1.0), lambda _p: dests)
    assert out["rows"][0]["status"] == "mismatch" and out["counts"].get("wrong", 0) == 0


def test_text_path_strips_the_wake_echo_like_the_node():
    """글자 경로 재생은 음성 노드처럼 호출 응답 "네?"의 메아리를 뗀 글자를 해석기에 넣는다(3-나-1, 기준 점수 11-18)."""
    row = {"question": "네?", "kind": "open", "dialog_state": "idle", "stt": "네 나 배 아파.", "answer_text": "네 나 배 아파"}
    assert parsed_text_for(row) == "나 배 아파."
    assert parsed_text_for({**row, "dialog_state": "asking_next"}) == "네 나 배 아파."      # 물어 둔 질문을 쥔 단계
    assert parsed_text_for({**row, "question": "화장실로 안내해드릴까요?", "kind": "confirm"}) == "네 나 배 아파."
    assert parsed_text_for({**row, "stt": "네."}) == "네."
    assert parsed_text_for({"stt": "그래.", "answer_text": "그래"}) == "그래."               # 옛 행(칸 없음)


def test_text_interpreter_uses_the_stripped_text():
    import inspect
    source = inspect.getsource(answer_replay.main)
    assert "parse_intent(parsed_text_for(row)" in source
    assert "parse_intent(text_for(row)" not in source


def test_reserve_call_on_a_labeled_row_counts_as_wrong(rows_labeled, dests):
    """정답 208행에는 예약 발화가 없다 — 도구 출력에서 reserve_facility 가 나오면 잘못된 행동이다."""
    from src.reservation_flow import ReserveCall
    run = replay(rows_labeled, lambda row: (ReserveCall(facility="세미나실"), 0.1), lambda _p: dests)
    assert run["counts"].get("wrong", 0) == len(rows_labeled)
    assert all(r["why"] == "reserve_facility" for r in run["rows"])


def test_tools_switch_is_set_before_the_parser_is_imported():
    assert answer_replay.tools_env(True) == {"VICA_TOOLS_TEXT": "on", "VICA_TOOLS_AUDIO": "on"}
    assert answer_replay.tools_env(False) == {"VICA_TOOLS_TEXT": "off", "VICA_TOOLS_AUDIO": "off"}


def test_answer_switch_env():
    assert answer_replay.answer_env(True) == {"VICA_ANSWER": "on"}
    assert answer_replay.answer_env(False) == {"VICA_ANSWER": "off"}


def test_question_for_maps_the_row(dests):
    row = {"kind": "other", "question": "무엇을 도와드릴까요?"}
    asked = answer_replay.question_for(row, dests)
    assert (asked.id, asked.kind, asked.text, asked.option_ids) == ("replay", "open", "무엇을 도와드릴까요?", ())
    confirm = answer_replay.question_for({"kind": "confirm", "question": "사회복지로 안내해드릴까요?"}, dests)
    assert confirm.option_ids == ("d1",)
    choice = answer_replay.question_for({"kind": "choice", "question": "사회복지와 화장실 중 어디로 갈까요?"}, dests)
    assert choice.option_ids == ("d1", "d2")


def test_replay_grades_by_the_answer_field(rows_labeled, dests):
    said_yes = VicaIntent(intent="navigate", matched_destination_id="d1", answer="yes")
    run = replay(rows_labeled[:1], lambda row: (said_yes, 0.5), lambda _p: dests)
    assert run["rows"][0]["status"] == "ok" and run["rows"][0]["pred"] == "yes"


def test_p6_audio_path_grades_what_the_parser_returned():
    """Task 3.5b: 짧은 대답 규칙은 운영 코드(_deliver_realtime_result)가 쓴다 — 재생 도구가 Realtime 결과를 따로 덮지 않는다
    (측정 = 운영)."""
    import inspect
    assert not hasattr(answer_replay, "with_transcript_rule")
    source = inspect.getsource(answer_replay.main)
    assert "with_transcript_rule" not in source and "return intent, latency" in source


def test_rows_keep_both_readings_and_the_source(monkeypatch, rows_labeled, dests):
    """유료 측정 한 번으로 두 채점(대답 칸 · 의도 칸)이 다 나오게 행마다 둘 다 남긴다(검토 I1)."""
    monkeypatch.setenv("VICA_ANSWER", "on")
    said = VicaIntent(intent="navigate", matched_destination_id="d1", answer="yes")
    by_rule = replay(rows_labeled[:1], lambda row: (said, 0.5), lambda _p: dests)["rows"][0]      # "네" — 규칙이 정한 행
    assert (by_rule["pred"], by_rule["pred_intent"], by_rule["pred_intent_value"]) == ("yes", "destination", "d1")
    assert (by_rule["answer"], by_rule["answer_value"], by_rule["answer_src"]) == ("yes", "", "rule")
    free = {**rows_labeled[2], "answer_text": "화장실로 가자"}
    moved = VicaIntent(intent="navigate", matched_destination_id="d2", answer="destination", answer_value="d2")
    by_llm = replay([free], lambda row: (moved, 0.5), lambda _p: dests)["rows"][0]
    assert (by_llm["answer"], by_llm["answer_src"]) == ("destination", "llm")
    plain = replay([free], lambda row: (VicaIntent(intent="deny"), 0.5), lambda _p: dests)["rows"][0]
    assert (plain["pred"], plain["pred_intent"], plain["answer"], plain["answer_src"]) == ("no", "no", "", "")


def test_transcript_for_reads_only_the_local_transcript(rows_labeled):
    """transcript_for 는 그때의 받아쓴 글자(stt)만 읽는다 — 없으면 들은 글자(answer_text)로 채우지 않고 빈 글자(검토 M1)."""
    row = {**rows_labeled[1], "stt": "", "answer_text": "1분"}       # kind=minutes
    assert answer_replay.transcript_for(row) == "" and answer_replay.transcript_for({**row, "stt": " 삼십분. "}) == "삼십분."


def test_rows_keep_the_reply_sentence(rows_labeled, dests):
    """되묻기가 어떤 문장이었는지 보려면 행에 문장이 남아야 한다(11-19 사용자 "다시 물을 때 어떤 대사를 하는지도")."""
    asked = VicaIntent(intent="clarify", reply="어느 화장실로 갈까요?", need_confirm=False)
    row = replay(rows_labeled[:1], lambda row: (asked, 0.5), lambda _p: dests)["rows"][0]
    assert row["reply"] == "어느 화장실로 갈까요?"
    empty = replay([{**rows_labeled[0], "answer_text": ""}], lambda row: (asked, 0.5), lambda _p: dests)["rows"][0]
    assert empty["reply"] == ""


def test_model_tag_for_file_names():
    """--model 로 다른 로컬 모델을 잴 때 결과 파일 이름에 붙일 꼬리표(11-19 사용자 "이미 설치된 gemma4로")."""
    assert answer_replay.model_tag("midm2-mini") == "midm2-mini"
    assert answer_replay.model_tag("hf.co/unsloth/gemma-4-E2B-it-GGUF:Q4_K_M") == "gemma-4-E2B-it-GGUF-Q4_K_M"
    assert answer_replay.model_tag("") == ""


# ---- 새 답지 채점(설계 2026-11-20 G2 · 8절, 계획 Task 1-4) ------------------------------------------------------------
KEY_ROW = {"date": "11-04", "time": "10:00:00", "dest_yaml": "x", "stt": "", "ts": "9.0", "kind": "open",
           "question": "네?", "answer_text": "화장실 가자", "label_answer": "destination", "label_value": "d2"}


def _key(action: str, set_: str = "seen", **more) -> dict:
    return {"action": action, "target": "", "minutes": "", "set": set_, "source": "rule", "drop": "", **more}


def test_key_grading_uses_what_the_mission_receives(dests):
    """확인 질문 없이 나온 확정 navigate 는 음성 노드가 제안으로 낮춘다(demote_unasked_confirm) — 반응은 go 가 아니라 confirm."""
    confirmed = VicaIntent(intent="navigate", matched_destination_id="d2", need_confirm=False)
    run = replay([KEY_ROW], lambda row: (confirmed, 0.5), lambda _p: dests, key={"9.0": _key("confirm", target="d2")})
    row = run["rows"][0]
    assert row["reaction"]["action"] == "confirm" and row["key_status"] == "ok" and row["set"] == "seen"
    assert run["key"]["seen"] == {"ok": 1}


def test_key_grading_reads_the_handed_over_question(dests):
    """되묻는 문장은 미션에 넘어간다(hand_over_question) — 넘어간 문장으로 which · reask 를 가른다."""
    which = VicaIntent(intent="clarify", need_confirm=False, reply="사회복지와 화장실 중 어디로 갈까요?")
    run = replay([KEY_ROW], lambda row: (which, 0.5), lambda _p: dests, key={"9.0": _key("which")})
    assert run["rows"][0]["ask_text"] == "사회복지와 화장실 중 어디로 갈까요?" and run["rows"][0]["key_status"] == "ok"


def test_key_summary_counts_dropped_requests_and_principles(dests):
    rows = [{**KEY_ROW, "ts": "1.0"},
            {**KEY_ROW, "ts": "2.0", "kind": "confirm", "question": "사회복지로 안내해드릴까요?", "answer_text": "응"}]
    key = {"1.0": _key("confirm", "heldout", target="d2"), "2.0": _key("yes", "heldout")}
    run = replay(rows, lambda row: (VicaIntent(intent="unknown"), 0.5), lambda _p: dests, key=key)
    assert run["key"]["heldout"] == {"wrong": 1, "reask": 1}       # "네?" 뒤의 말을 버림 · 확인 질문은 미션이 다시 물음
    assert run["key_dropped"] == {"heldout": 1} and run["key_go"] == {}
    assert run["key_principles"]["heldout"] == {5: 1, 6: 1}


def test_key_counts_departure_without_confirm_only_when_it_reaches_the_mission(dests):
    asked = {**KEY_ROW, "kind": "confirm", "question": "화장실로 안내해드릴까요?", "answer_text": "사회복지로 가자"}
    other = VicaIntent(intent="navigate", matched_destination_id="d1", need_confirm=False)   # 묻지 않은 곳으로 확정
    run = replay([asked], lambda row: (other, 0.5), lambda _p: dests, key={"9.0": _key("confirm", target="d1")})
    assert run["rows"][0]["reaction"]["action"] == "confirm" and run["key_go"] == {}          # 음성 노드가 낮춘다


def test_rows_without_a_key_row_are_not_graded(dests):
    run = replay([KEY_ROW], lambda row: (VicaIntent(intent="affirm"), 0.5), lambda _p: dests, key={})
    assert run["rows"][0]["key_status"] == "nokey" and run["key"] == {}


def test_without_a_key_the_result_is_as_before(rows_labeled, dests):
    out = replay(rows_labeled, lambda row: (VicaIntent(intent="affirm"), 1.2), lambda _p: dests)
    assert "key" not in out and "key_status" not in out["rows"][0]


def test_select_rows_by_set():
    rows = [{"ts": "1.0"}, {"ts": "2.0"}, {"ts": "3.0"}]
    key = {"1.0": _key("yes", "seen"), "2.0": _key("yes", "heldout")}
    assert answer_replay.select_rows(rows, key, "seen") == [{"ts": "1.0"}]
    assert answer_replay.select_rows(rows, key, "heldout") == [{"ts": "2.0"}]
    assert answer_replay.select_rows(rows, key, "all") == rows


def test_heldout_rows_are_counted_but_not_listed_in_the_report(dests):
    rows = [{**KEY_ROW, "ts": "1.0", "answer_text": "본 말 문장"},
            {**KEY_ROW, "ts": "2.0", "date": "10-21", "answer_text": "안 본 말 문장"}]
    key = {"1.0": _key("silent", "seen"), "2.0": _key("silent", "heldout")}
    moved = VicaIntent(intent="navigate", matched_destination_id="d2", need_confirm=True)
    run = replay(rows, lambda row: (moved, 0.5), lambda _p: dests, key=key)
    md = answer_replay._report_md("gpt", [run], setup="")
    assert "본 말 문장" in md and "안 본 말 문장" not in md
    assert "heldout" in md and "원칙 7" in md


def test_result_name_keeps_runs_on_different_label_files_apart():
    a = answer_replay.result_name("gpt", "", False, False, False, "labels_208_confirmed", True, "20261121-1100")
    b = answer_replay.result_name("gpt", "", False, False, False, "people_talk_labels", True, "20261121-1100")
    assert a != b and a.startswith("replay_gpt_key-labels_208_confirmed")
    assert answer_replay.result_name("gpt", "", False, True, False, "labeled", False, "20261121-1100") == "replay_gpt_answer_20261121-1100"


def test_regrade_uses_the_saved_reaction_with_a_newer_key(dests):
    """답지가 고쳐지면(사용자 확인 · 새 행) 저장된 결과를 LLM 호출 없이 다시 채점한다."""
    proposal = VicaIntent(intent="navigate", matched_destination_id="d2", need_confirm=True)
    run = replay([KEY_ROW], lambda row: (proposal, 0.5), lambda _p: dests, key={"9.0": _key("etc", source="pending")})
    assert run["rows"][0]["key_status"] == "pending"
    again = answer_replay.regrade(run, {"9.0": _key("confirm", target="d2", source="user")})
    assert again["rows"][0]["key_status"] == "ok" and again["key"] == {"seen": {"ok": 1}}
    assert run["rows"][0]["key_status"] == "pending"            # 원본은 그대로 둔다


def test_regrade_keeps_rows_that_were_never_interpreted(rows_labeled):
    run = replay(rows_labeled[:1], lambda r: (VicaIntent(intent="affirm"), 1.0), lambda _p: None, key={})
    assert answer_replay.regrade(run, {"1.0": _key("yes")})["rows"][0]["status"] == "skip"


def test_regrade_rereads_saved_call_answers_with_the_current_rule(dests):
    """예전 채점기로 저장된 결과 — 호출에 답한 "네, 말씀하세요."가 reask 로 적혀 있어도 다시 채점하면 wake 로 읽는다."""
    saved = {"rows": [{**KEY_ROW, "intent": "unknown", "reaction": {"action": "reask", "target": "", "minutes": "",
                                                                     "say": "네, 말씀하세요."}}]}
    again = answer_replay.regrade(saved, {"9.0": _key("wake")})
    assert again["rows"][0]["reaction"]["action"] == "wake" and again["rows"][0]["key_status"] == "ok"
    kept = {"rows": [{**KEY_ROW, "intent": "clarify", "reaction": {"action": "reask", "target": "", "minutes": "",
                                                                    "say": "네, 말씀하세요."}}]}
    assert answer_replay.regrade(kept, {"9.0": _key("wake")})["rows"][0]["reaction"]["action"] == "reask"


def test_recut_switch_goes_into_the_environment_before_the_parser_loads():
    """--recut: 재단 스위치(2026-11-21). 운영 .env 값과 무관하게 이 값으로 돈다 — 해석기는 부를 때마다 환경변수를 읽는다."""
    assert answer_replay.recut_env(True) == {"VICA_RECUT": "on"}
    assert answer_replay.recut_env(False) == {"VICA_RECUT": "off"}
    assert "_recut_" in answer_replay.result_name("gpt", "", False, False, True, "labels", False, "20261121-1300")


def test_regrade_rereads_saved_dismiss_the_way_the_mission_takes_it(dests):
    """예전 채점기로 저장된 dismiss 는 bye 로 적혀 있다 — 미션 질문이 열려 있던 행은 거절로 다시 읽는다."""
    bye = {"action": "bye", "target": "", "minutes": "", "say": ""}
    asked = {**KEY_ROW, "kind": "yesno", "question": "여기서 기다릴까요?", "intent": "dismiss", "reaction": bye}
    assert answer_replay.regrade({"rows": [asked]}, {"9.0": _key("no")})["rows"][0]["reaction"]["action"] == "no"
    called = {**KEY_ROW, "kind": "open", "question": "네?", "intent": "dismiss", "reaction": bye}
    assert answer_replay.regrade({"rows": [called]}, {"9.0": _key("no")})["rows"][0]["key_status"] == "ok"

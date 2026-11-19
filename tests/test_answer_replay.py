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


def test_transcript_rule_wins_on_the_audio_path(monkeypatch, rows_labeled, dests):
    from src.reservation_flow import ReserveCall
    monkeypatch.setenv("VICA_ANSWER", "on")
    row = {**rows_labeled[1], "stt": "삼십분."}                 # kind=minutes
    asked = answer_replay.question_for(row, dests)
    heard = VicaIntent(intent="wait", wait_minutes=15, answer="minutes", answer_value="15")
    fixed = answer_replay.with_transcript_rule(heard, row, asked)
    assert (fixed.answer, fixed.answer_value, fixed.wait_minutes) == ("minutes", "30", 15)
    long = {**rows_labeled[1], "stt": "화장실 가고 싶어"}
    assert answer_replay.with_transcript_rule(heard, long, asked) is heard
    call = ReserveCall(facility="세미나실")
    assert answer_replay.with_transcript_rule(call, row, asked) is call


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


def test_audio_rule_needs_a_real_transcript(monkeypatch, rows_labeled, dests):
    """받아쓴 글자(stt)가 없던 발화에 Realtime 이 들은 글자로 규칙을 대지 않는다 — 운영에는 없던 글자다(검토 M1)."""
    monkeypatch.setenv("VICA_ANSWER", "on")
    row = {**rows_labeled[1], "stt": "", "answer_text": "1분"}       # kind=minutes
    asked = answer_replay.question_for(row, dests)
    heard = VicaIntent(intent="wait", wait_minutes=15, answer="minutes", answer_value="15")
    assert answer_replay.with_transcript_rule(heard, row, asked) is heard
    assert answer_replay.transcript_for(row) == "" and answer_replay.transcript_for({**row, "stt": " 삼십분. "}) == "삼십분."


def test_audio_minutes_rule_does_not_override_a_refusal(monkeypatch, rows_labeled, dests):
    monkeypatch.setenv("VICA_ANSWER", "on")
    row = {**rows_labeled[1], "stt": "십 분 말고 그냥 안내 끝내 줘"}
    asked = answer_replay.question_for(row, dests)
    refused = VicaIntent(intent="deny", answer="no")
    assert answer_replay.with_transcript_rule(refused, row, asked) is refused


def test_rows_keep_the_reply_sentence(rows_labeled, dests):
    """되묻기가 어떤 문장이었는지 보려면 행에 문장이 남아야 한다(11-19 사용자 "다시 물을 때 어떤 대사를 하는지도")."""
    asked = VicaIntent(intent="clarify", reply="어느 화장실로 갈까요?", need_confirm=False)
    row = replay(rows_labeled[:1], lambda row: (asked, 0.5), lambda _p: dests)["rows"][0]
    assert row["reply"] == "어느 화장실로 갈까요?"
    empty = replay([{**rows_labeled[0], "answer_text": ""}], lambda row: (asked, 0.5), lambda _p: dests)["rows"][0]
    assert empty["reply"] == ""

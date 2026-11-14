"""재생 도구 — 경로별 기준 점수 (설계 2026-11-14 5절, 계획 Task 4). LLM 없이 가짜 interpret 로 시험한다."""
from __future__ import annotations

from datetime import datetime

import pytest

from src.replies import LLM_UNAVAILABLE
from src.schema import DestinationData, VicaIntent
from tools.answer_replay import call_with_retry, clip_for, path_env, replay, text_for


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

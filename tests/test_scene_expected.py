"""대화 장면 분석기 — 설계상 대답 (설계 2026-11-20 5.2). API·Ollama 없이 interpret 를 주입한다."""
import pytest

from src.schema import DestinationData, VicaIntent
from tools.scene.events import Event
from tools.scene.expected import Expected, reask, rule_expected, sentence_for
from tools.scene.scenes import Scene

DESTS = [DestinationData(id="h1", name="현장 안전 지원 휴머노이드")]


def _scene(user="아니", mode="text", local=True, state="confirming", sid="20261120-100000-wake"):
    evs = [Event(0.5, "llm", "startup", {"text": ""}, f"의도 입력 모드: {mode} | 로컬 규칙: {'켬' if local else '꺼짐'}"),
           Event(1.0, "mis", "state", {"from": "idle", "to": state, "via": "mission"}, ""),
           Event(2.0, "tts", "tts_play", {"priority": "normal", "text": "현장 안전 지원 휴머노이드로 안내해드릴까요?"}, ""),
           Event(5.0, "wake", "user_text", {"text": user}, "")]
    return Scene(sid, 0.5, 6.0, "wake", evs, "time", None)


def test_sentence_navigate_confirm():
    assert sentence_for("navigate", "h1", True, "", DESTS) == "현장 안전 지원 휴머노이드로 안내해드릴까요?"


def test_sentence_cancel_and_unknown():
    assert sentence_for("cancel", None, True, "", DESTS) == "안내를 취소할까요?"
    assert sentence_for("clarify", None, False, "", DESTS) is None


def test_rule_deny_short_answer():
    e = rule_expected(_scene("아니"), DESTS)
    assert e is not None and e.by == "규칙" and e.intent == "deny"


def test_rule_none_for_destination_request_and_audio():
    assert rule_expected(_scene("화장실 가자", state="idle"), DESTS) is None
    assert rule_expected(_scene("아니", mode="audio"), DESTS) is None


def _fake(calls):
    def interpret(scene, path, destinations, state):
        calls.append(path)
        return VicaIntent(intent="navigate", matched_destination_id="h1", need_confirm=True, reply="제안")
    return interpret


def test_reask_uses_cache(tmp_path):
    calls = []
    a = reask(_scene(), "gpt", _fake(calls), destinations=DESTS, cache_dir=tmp_path)
    b = reask(_scene(), "gpt", _fake(calls), destinations=DESTS, cache_dir=tmp_path)
    assert calls == ["gpt"] and a == b
    assert a.sentence == "현장 안전 지원 휴머노이드로 안내해드릴까요?"
    assert (tmp_path / "20261120-100000-wake_gpt.json").exists()


def test_reask_midm_refused_when_driving(tmp_path):
    calls = []
    with pytest.raises(RuntimeError, match="주행 중이라 로컬 다시 묻기 거부"):
        reask(_scene(), "midm", _fake(calls), destinations=DESTS, cache_dir=tmp_path, is_driving=lambda: True)
    assert calls == []
    assert reask(_scene(), "midm", _fake(calls), destinations=DESTS, cache_dir=tmp_path,
                 is_driving=lambda: False).by == "LLM 추정"
    # 클라우드 경로는 주행 중에도 허용
    assert reask(_scene(sid="x"), "gpt", _fake(calls), destinations=DESTS, cache_dir=tmp_path,
                 is_driving=lambda: True).by == "LLM 추정"


def test_expected_marks_estimate(tmp_path):
    e = reask(_scene(), "gpt", _fake([]), destinations=DESTS, cache_dir=tmp_path)
    assert "다시 물은 결과 — 그때와 다를 수 있음" in e.note


def test_reask_no_clip_is_none(tmp_path):
    e = reask(_scene(), "realtime", lambda *a: None, destinations=DESTS, cache_dir=tmp_path)
    assert e.by == "없음" and "소리 없음" in e.note


def test_deny_in_asking_next_asks_entrance():
    """11-04 15:07:19 — '여기서 대기할까요?'에 '아니.' → '여기까지 안내를 마칠까요?'"""
    sc = _scene("아니.", state="asking_next")
    sc.events[2] = Event(2.0, "tts", "tts_play", {"priority": "normal", "text": "여기서 대기할까요?"}, "")
    e = rule_expected(sc, DESTS)
    assert e.intent == "deny" and e.sentence == "여기까지 안내를 마칠까요?"


def test_affirm_after_entrance_question_finishes():
    sc = _scene("네", state="asking_next")
    sc.events[2] = Event(2.0, "tts", "tts_play", {"priority": "normal", "text": "여기까지 안내를 마칠까요?"}, "")
    assert rule_expected(sc, DESTS).sentence == "안내를 종료합니다."


def test_affirm_in_confirming_starts_destination():
    e = rule_expected(_scene("네", state="confirming"), DESTS)
    assert e.sentence == "현장 안전 지원 휴머노이드로 안내를 시작합니다."


def test_unlisted_row_is_none_with_note():
    e = rule_expected(_scene("아니", state="confirming"), DESTS)
    assert e.sentence is None and "문장 표 없음" in e.note

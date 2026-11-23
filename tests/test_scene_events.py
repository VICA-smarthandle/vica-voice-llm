"""대화 장면 분석기 1단계 — 로그 줄 분류와 읽기 (설계 2026-11-20 5.1)."""
import os
import shutil
import time
from pathlib import Path

import pytest

from tools.scene import events
from tools.scene.events import Event, classify, read_events

FIXTURE = Path(__file__).parent / "fixtures" / "scene" / "1104_sample.log"
WAKE = "vica_wakeword_node"


def test_wake_near_and_fire():
    e = classify(1.0, WAKE, "🎯 호출 점수 놓침: 0.83 (문턱 0.70)")
    assert e.kind == "wake_near" and e.data == {"peak": 0.83, "threshold": 0.70}
    assert e.node == "wake" and e.t == 1.0
    f = classify(1.0, WAKE, "🎯 호출 점수 확정: 0.99 (문턱 0.70)")
    assert f.kind == "wake_fire" and f.data["peak"] == 0.99
    assert classify(1.0, WAKE, "🎯 호출 점수 자기 말 무시: 0.91 (문턱 0.70)").kind == "wake_self"


def test_ghost_negative_speech():
    e = classify(1.0, WAKE, "청취 기각: empty:ghost speech=-0.00s rms=0.0051")
    assert e.kind == "listen_ghost" and e.data["rms"] == 0.0051 and e.data["speech"] <= 0


def test_listen_window_kinds():
    assert classify(1.0, WAKE, "청취 기각: empty:wake-ignored").kind == "wake_ignored"
    r = classify(1.0, WAKE, "청취 기각: empty:short-reject '비.'")
    assert r.kind == "listen_reject" and r.data["state"] == "empty:short-reject '비.'"
    assert classify(1.0, WAKE, "청취 창 빈손 종료 (발화 없음/빈 전사)").kind == "listen_empty_free"
    assert classify(1.0, WAKE, "질문 예약 — TTS 종료 후 재청취 창을 연다").kind == "listen_reserve"
    o = classify(1.0, WAKE, "🙋 비카야 호출 #3 — 청취 창 열림, 미션 판정 대기")
    assert o.kind == "wake_open" and o.data["n"] == 3
    assert classify(1.0, WAKE, "🙋 창 안 호출 (소리로 구제): wake-rescue '디카야.'").kind == "wake_open"


def test_user_text_quality_timing():
    u = classify(1.0, WAKE, "🗣️ 호출 발화 -> /vica/user_text: '그래.'")
    assert u.kind == "user_text" and u.data["text"] == "그래."
    a = classify(1.0, WAKE, "🎧 발화 소리 -> /vica/user_audio (3.60s)")
    assert a.kind == "user_audio" and a.data == {"sec": 3.6}
    q = classify(1.0, WAKE, "수음 품질: rms 0.1261 · peak 1.00 · clip 0.8%")
    assert q.kind == "listen_quality" and q.data == {"rms": 0.1261, "peak": 1.0, "clip": 0.8}
    t = classify(1.0, WAKE, "계측: 대기 -0.02s · 발화 8.01s · 말끝판정 0.88s · STT 1.63s")
    assert t.kind == "listen_timing"
    assert t.data == {"wait": -0.02, "speech": 8.01, "endpoint": 0.88, "stt": 1.63}


def test_forward_answer_line():
    e = classify(1.0, WAKE, "긴급 관문 발동 → STT 기각 (전사: '아니.') · 관문 0.61 · 관문→판정 0.80 s "
                            "(STT 0.71 s) · 긴급 아님 — 질문 답으로 넘김")
    assert e.kind == "forward_answer" and e.data["text"] == "아니."
    assert classify(1.0, WAKE, "긴급 관문 발동 → STT 기각 (전사: '좋아.') · 관문 0.5 · 청취 창 안 — 긴급 아님, 듣기 이어감") is None


def test_llm_lines():
    i = classify(1.0, "vica_llm_intent_node", "입력='그래.' -> intent=affirm matched=None safety=normal")
    assert i.kind == "llm_input" and i.data["text"] == "그래." and i.data["intent"] == "affirm"
    d = classify(1.0, "vica_llm_intent_node", "재청취 기각(무의미): '듀모노이드 보러 가자.' intent=clarify")
    assert d.kind == "relisten_drop" and d.data == {"text": "듀모노이드 보러 가자.", "intent": "clarify"}
    rt = classify(1.0, "vica_llm_intent_node",
                  "[RT] heard='그래' intent=affirm dest=- nc=False conf=0.9 reply='' src=llm dt=1.2s tokens=3")
    assert rt.kind == "llm_rt" and rt.data["intent"] == "affirm" and rt.data["dt"] == 1.2
    qv = classify(1.0, "vica_llm_intent_node", "[질문] id=q3 kind=confirm 나이=2.5s")
    assert qv.kind == "q_voice" and qv.data == {"id": "q3", "kind": "confirm", "age": 2.5}


def test_mission_say_is_mission_source():
    e = classify(1.0, "vica_mission_manager", "TTS[response]: 안내를 취소할까요?")
    assert e.kind == "mis_say" and e.data == {"priority": "response", "text": "안내를 취소할까요?"}
    assert e.node == "mis"


def test_mission_request_wake_state_question():
    r = classify(1.0, "vica_mission_manager",
                 "음성 요청 intent=cancel dest=- confirm=True: navigating -> navigating 말=['안내를 취소할까요?']")
    assert r.kind == "mis_req" and r.data["intent"] == "cancel" and r.data["confirm"] is True
    assert r.data["from"] == "navigating" and r.data["to"] == "navigating"
    w = classify(1.0, "vica_mission_manager", "'비카야'(대답 안 함): estopped -> estopped")
    assert w.kind == "mis_wake" and w.data == {"heard": False, "from": "estopped", "to": "estopped"}
    s = classify(1.0, "vica_mission_manager", "중앙 estop=True -> state=estopped")
    assert s.kind == "state" and s.data["to"] == "estopped"
    h = classify(1.0, "vica_mission_manager", "홈 복귀 시작: idle -> returning")
    assert h.kind == "state" and (h.data["from"], h.data["to"]) == ("idle", "returning")
    o = classify(1.0, "vica_mission_manager", "[질문] 열림 id=q3 kind=confirm 후보=1 요청=y 글자=안내해드릴까요?")
    assert o.kind == "q_open" and o.data["id"] == "q3"
    c = classify(1.0, "vica_mission_manager", "[질문] 닫힘 id=q3 사유=answer 열린 시간=4.2s")
    assert c.kind == "q_close" and c.data["reason"] == "answer" and c.data["open_s"] == 4.2


def test_tts_play_usb_startup_and_noise():
    p = classify(1.0, "vica_tts_node", "재생[response]: 네?")
    assert p.kind == "tts_play" and p.data == {"priority": "response", "text": "네?"}
    assert classify(1.0, WAKE, "[USB] reSpeaker 소리가 12 s 째 안 들어옴 — 끊김 의심").kind == "usb"
    assert classify(1.0, WAKE, "귀 감시가 멈췄습니다 — 노드를 끝냅니다: SystemExit('x')").kind == "usb"
    assert classify(1.0, WAKE, "VICA 웨이크워드 감시 시작 (발행: /vica/emergency)").kind == "startup"
    assert classify(1.0, "vica_tts_node", "VICA TTS node 시작 (구독: /vica/tts_request)").kind == "startup"
    assert classify(1.0, WAKE, "🧭 호출 방향 188° (방향 회전 꺼짐 — 미션에 안 보냄)") is None
    assert classify(1.0, "controller_server", "재생[response]: 네?") is None


def test_spk_line():
    e = classify(1.0, WAKE, "[화자] fail 점수 0.31/0.45 target 1.20s 방향 안 목소리 미달(대기) — 버림")
    assert e.kind == "spk" and e.data["verdict"] == "fail" and e.data["score"] == 0.31
    assert e.data["threshold"] == 0.45 and e.data["dropped"] is True
    assert e.data["reason"] == "목소리 미달(대기)"


def test_llm_fail_from_stderr():
    e = classify(1.0, "vica_llm", "[ERROR] [LLM] 호출 실패(모델 midm2-mini): timed out")
    assert e.kind == "llm_fail" and e.node == "llm"
    assert "timed out" in e.data["text"]


def _node_log(path: Path, lines: list[str]):
    path.write_text("".join(l + "\n" for l in lines), encoding="utf-8")


def test_truncated_last_line_skipped(tmp_path):
    _node_log(tmp_path / "python3_1_1.log", [
        "[INFO] [1000.5] [vica_wakeword_node]: 🎯 호출 점수 확정: 0.99 (문턱 0.70)",
        "[INFO] [17921",
    ])
    out = read_events(0, 2e9, log_dir=tmp_path, relaunch_dir=tmp_path / "none", journal=False)
    assert [e.kind for e in out] == ["wake_fire"]


def test_duplicate_lines_merged(tmp_path):
    line = "[INFO] [1000.5] [vica_wakeword_node]: 🎯 호출 점수 확정: 0.99 (문턱 0.70)"
    _node_log(tmp_path / "python3_1_1.log", [line])
    rl = tmp_path / "relaunch"
    rl.mkdir()
    _node_log(rl / "voice_1104_100000.log", ["[vica_wakeword-1] " + line])
    out = read_events(0, 2e9, log_dir=tmp_path, relaunch_dir=rl, journal=False)
    assert [e.kind for e in out] == ["wake_fire"]


def test_identical_lines_within_one_source_kept(tmp_path):
    rl = tmp_path / "relaunch"
    rl.mkdir()
    fail = "[vica_llm-1] [ERROR] [LLM] 호출 실패(클라우드만, 폴백 꺼짐): timed out"
    _node_log(rl / "voice_1011_175409.log", [
        "[vica_llm-1] [INFO] [1500.25] [vica_llm_intent_node]: 입력='그래.' -> intent=affirm matched= safety=normal",
        fail, fail])
    out = read_events(0, 2e9, log_dir=tmp_path / "none", relaunch_dir=rl, journal=False)
    assert [e.kind for e in out] == ["llm_input", "llm_fail", "llm_fail"]


def test_journal_same_second_failures_kept(tmp_path, monkeypatch):
    line = "2026-11-20T07:52:05+0900 Z-jet run_llm.sh[1]: [vica_llm-1] [ERROR] [LLM] 호출 실패(모델 m): timed out"
    monkeypatch.setattr(events, "_journal_lines", lambda s, u: [line, line])
    out = read_events(0, 2e9, log_dir=tmp_path / "a", relaunch_dir=tmp_path / "b", journal=True)
    assert [e.kind for e in out] == ["llm_fail", "llm_fail"]


def test_relaunch_stderr_gets_time_of_previous_line(tmp_path):
    rl = tmp_path / "relaunch"
    rl.mkdir()
    _node_log(rl / "voice_1011_175409.log", [
        "[vica_llm-1] [INFO] [1500.25] [vica_llm_intent_node]: 입력='그래.' -> intent=affirm matched= safety=normal",
        "[vica_llm-1] [ERROR] [LLM] 호출 실패(클라우드만, 폴백 꺼짐): timed out",
    ])
    out = read_events(0, 2e9, log_dir=tmp_path / "none", relaunch_dir=rl, journal=False)
    assert [(e.kind, e.t) for e in out] == [("llm_input", 1500.25), ("llm_fail", 1500.25)]


def test_journal_lines_injected(tmp_path, monkeypatch):
    monkeypatch.setattr(events, "_journal_lines", lambda since, until: [
        "2026-11-20T07:52:05+0900 Z-jet run_llm.sh[10100]: [vica_llm-1] [ERROR] [LLM] 호출 실패(모델 midm2-mini): timed out",
        "2026-11-20T07:52:06+0900 Z-jet run_llm.sh[10100]: [vica_llm-1] [LLM] 로컬 모델 예열 완료: midm2-mini",
    ])
    out = read_events(0, 2e9, log_dir=tmp_path / "a", relaunch_dir=tmp_path / "b", journal=True)
    assert [e.kind for e in out] == ["llm_fail"]
    assert out[0].t == pytest.approx(1795128725.0)


def test_since_until_and_mtime_filter(tmp_path):
    old = tmp_path / "python3_1_1.log"
    _node_log(old, ["[INFO] [100.0] [vica_wakeword_node]: 🎯 호출 점수 확정: 0.99 (문턱 0.70)"])
    os.utime(old, (100, 100))
    new = tmp_path / "python3_2_2.log"
    _node_log(new, [
        "[INFO] [5000.0] [vica_wakeword_node]: 🎯 호출 점수 확정: 0.99 (문턱 0.70)",
        "[INFO] [9000.0] [vica_wakeword_node]: 🎯 호출 점수 확정: 0.98 (문턱 0.70)",
    ])
    os.utime(new, (9000, 9000))
    out = read_events(4000, 6000, log_dir=tmp_path, relaunch_dir=tmp_path / "x", journal=False)
    assert [e.t for e in out] == [5000.0]


def test_fixture_counts(tmp_path):
    shutil.copy(FIXTURE, tmp_path / "python3_1_1.log")
    out = read_events(0, 2e9, log_dir=tmp_path, relaunch_dir=tmp_path / "x", journal=False)
    assert [e.t for e in out] == sorted(e.t for e in out)
    c = {}
    for e in out:
        c[e.kind] = c.get(e.kind, 0) + 1
    assert c["wake_fire"] == 3 and c["wake_near"] == 1
    assert c["relisten_drop"] == 4
    assert c["forward_answer"] == 1
    assert c["mis_say"] == 19
    assert c["listen_ghost"] == 4 and c["wake_ignored"] == 2 and c["listen_empty_free"] == 2
    assert c["tts_play"] == 19 and c["listen_reserve"] == 11

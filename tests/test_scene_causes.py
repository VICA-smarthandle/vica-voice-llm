"""대화 장면 분석기 — 원인 판정 (설계 2026-11-20 5.3)."""
import datetime as dt
import shutil
from pathlib import Path

import pytest

from tools.scene.causes import Cause, flagged, judge, make_ctx
from tools.scene.events import Event, read_events
from tools.scene.scenes import Scene, build_scenes

FIXTURE = Path(__file__).parent / "fixtures" / "scene" / "1104_sample.log"
CTX = {"wake_threshold": 0.70, "listen_min_rms": 0.02, "approach_min_rms": 0.014, "llm_timeout": 15.0}


@pytest.fixture(scope="module")
def events(tmp_path_factory):
    d = tmp_path_factory.mktemp("log")
    shutil.copy(FIXTURE, d / "python3_1_1.log")
    return read_events(0, 2e9, log_dir=d, relaunch_dir=d / "none", journal=False)


def _t(hms):
    h, m, s = hms.split(":")
    ref = dt.datetime.fromtimestamp(1793753928)
    return ref.replace(hour=int(h), minute=int(m), second=0, microsecond=0).timestamp() + int(s)


def scene(events, hms):
    t = _t(hms)
    hits = [s for s in build_scenes(events) if s.start - 0.5 <= t <= s.end + 0.5]
    assert hits, hms
    return hits[0]


def pairs(s, ctx=CTX):
    return {(c.name, c.status) for c in judge(s, ctx)}


def names(s, ctx=CTX):
    return {c.name for c in judge(s, ctx)}


def ev(t, kind, node="wake", **d):
    return Event(t, node, kind, d, "")


def mk(evs):
    return Scene("x", evs[0].t, evs[-1].t, "wake", evs, "time", None)


def test_1507_double_forward(events):
    assert names(scene(events, "15:07:19")) >= {"같은 말 두 번 전달"}


def test_1120_relisten(events):
    assert ("재청취 기각", "확정") in pairs(scene(events, "11:20:38"))


def test_1118_question_overlap(events):
    assert ("질문 겹침", "확정") in pairs(scene(events, "11:18:54"))


def test_1118_only_one_overlap(events):
    n = sum(1 for s in build_scenes(events) for c in judge(s, CTX) if c.name == "질문 겹침")
    assert n == 1


def test_1022_timeout_then_drop(events):
    c = pairs(scene(events, "10:22:43"))
    assert ("LLM 시간 초과", "추정") in c and ("재청취 기각", "확정") in c


def test_1022_1013_timeout(events):
    assert ("LLM 시간 초과", "추정") in pairs(scene(events, "10:22:13"))


def test_timeout_confirmed_with_stderr(events):
    s = scene(events, "10:22:43")
    s.events.append(ev(_t("10:22:43") - 5, "llm_fail", node="llm", text="[LLM] 호출 실패"))
    assert ("LLM 시간 초과", "확정") in pairs(s)


def test_1109_wake_near():
    s = mk([ev(100.0, "wake_near", peak=0.83, threshold=0.70)])
    assert ("호출 놓침", "추정") in pairs(s)


def test_near_without_any_threshold_is_not_wake_miss():
    c = judge(mk([ev(100.0, "wake_near", peak=0.83)]), {"wake_threshold": None})
    assert "호출 놓침" not in {x.name for x in c} and ("원인 미분류", "확인 필요") in {(x.name, x.status) for x in c}


def test_below_threshold_near_ignored():
    assert not judge(mk([ev(100.0, "wake_near", peak=0.5, threshold=0.70)]), CTX)


def test_near_within_2s_after_fire_is_normal():
    s = mk([ev(100.0, "wake_fire", peak=0.99, threshold=0.7), ev(101.5, "wake_near", peak=0.8, threshold=0.7)])
    c = judge(s, CTX)
    assert [(x.name, x.status) for x in c] == [("정상: 확정 직후 같은 말", "정상")] and not flagged(c)


def test_usage_guide_two_sentences_not_overlap():
    s = mk([ev(10.0, "mis_say", node="mis", priority="response", text="검지손가락으로 스위치를."),
            ev(10.001, "mis_say", node="mis", priority="response", text="어디로 가고 싶으신가요?"),
            ev(10.002, "listen_reserve")])
    assert "질문 겹침" not in names(s)


def test_ghost_quiet_vs_short():
    quiet = mk([ev(1.0, "listen_ghost", speech=0.0, rms=0.0068)])
    short = mk([ev(1.0, "listen_ghost", speech=0.03, rms=0.03)])
    assert ("목소리 작음", "확정") in pairs(quiet) and ("말이 너무 짧아 버림", "확정") in pairs(short)


def test_ghost_uses_approach_threshold_in_awaiting_user():
    st = ev(0.5, "state", node="mis", to="awaiting_user", via="mission")
    g = ev(1.0, "listen_ghost", speech=0.0, rms=0.016)
    assert ("말이 너무 짧아 버림", "확정") in pairs(mk([st, g]))
    assert ("목소리 작음", "확정") in pairs(mk([g]))


def test_wake_ignored_is_normal():
    c = judge(mk([ev(1.0, "wake_ignored")]), CTX)
    assert c and not flagged(c)


def test_empty_free_and_silence():
    assert ("말소리 안 잡힘", "확정") in pairs(mk([ev(1.0, "listen_empty_free")]))
    assert ("말소리 안 잡힘", "확정") in pairs(mk([ev(1.0, "q_close", node="mis", reason="silence")]))
    s = mk([ev(1.0, "listen_ghost", speech=0, rms=0.001), ev(2.0, "listen_empty_free")])
    assert "말소리 안 잡힘" not in names(s)


def test_speaker_and_usb_and_slow():
    spk = mk([ev(1.0, "spk", verdict="fail", reason="목소리 미달(대기)", dropped=False)])
    assert ("대기 중 본인 확인 미달", "확정") in pairs(spk)
    assert ("화자 판정으로 버림", "확정") in pairs(mk([ev(1.0, "spk", verdict="fail", reason="x", dropped=True)]))
    assert ("USB 끊김", "확정") in pairs(mk([ev(1.0, "usb")]))
    s = mk([ev(1.0, "user_text", text="가자"), ev(8.0, "tts_play", node="tts", priority="response", text="네")])
    assert ("느린 응답", "확정") in pairs(s)


def test_unclassified_needs_check_without_spk():
    s = mk([ev(1.0, "listen_reject", state="empty:x"), ev(2.0, "user_text", text="가자")])
    assert ("원인 미분류", "확인 필요") in pairs(s)
    assert flagged(judge(s, CTX))


def test_clean_scene_has_no_causes():
    s = mk([ev(1.0, "wake_fire", peak=0.99, threshold=0.7), ev(1.1, "wake_open", n=1),
            ev(3.0, "user_text", text="그래."), ev(3.2, "llm_input", node="llm", text="그래."),
            ev(3.3, "tts_play", node="tts", priority="response", text="네")])
    assert judge(s, CTX) == []


def test_flagged_empty_and_normal():
    assert not flagged([]) and not flagged([Cause("a", "정상")]) and flagged([Cause("a", "추정")])


def test_make_ctx_from_startup():
    e = [Event(1.0, "wake", "startup", {"text": "호출 문턱 0.70 · 말 시작 기다림 6.0s · 수음 최소 0.0200 · 짧은 답 최소 0.0200 · 접근 대답 최소 0.0140"}),
         Event(5.0, "wake", "startup", {"text": "호출 문턱 0.60 · 수음 최소 0.0300"})]
    c = make_ctx(e, 3.0)
    assert (c["wake_threshold"], c["listen_min_rms"], c["approach_min_rms"], c["llm_timeout"]) == (0.70, 0.02, 0.014, 15.0)
    assert make_ctx(e, 9.0)["wake_threshold"] == 0.60

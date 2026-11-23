"""대화 장면 분석기 — 집계 (설계 2026-11-20 6절)."""
from tools.scene.causes import Cause
from tools.scene.events import Event
from tools.scene.tally import ctx_at, tally


def E(t, kind, node="wake", **d):
    return Event(t, node, kind, d, "")


def run(events, causes=None, scenes=None):
    return tally(sorted(events, key=lambda e: e.t), scenes or [], causes or {})


def test_wake_counts():
    ev = [E(10, "wake_fire", peak=0.9, threshold=0.7),
          E(11, "wake_near", peak=0.8, threshold=0.7),
          E(30, "wake_near", peak=0.75, threshold=0.7),
          E(40, "wake_near", peak=0.5, threshold=0.7),
          E(50, "wake_near", peak=0.45, threshold=0.4)]
    r = run(ev)
    assert r["wake_fire"] == 1
    assert r["wake_near_over"] == 3 and r["wake_near_over_after_fire"] == 1


def test_after_wake_buckets():
    ev = [E(100, "wake_open", n=1), E(102, "user_text", text="a"),
          E(200, "wake_open", n=2), E(201, "wake_ignored"),
          E(300, "wake_open", n=3), E(310, "listen_empty_free"), E(310.5, "listen_ghost", speech=0.0, rms=0.0),
          E(400, "wake_open", n=4), E(410, "listen_empty_free"), E(415, "listen_ghost", speech=0.1, rms=0.0),
          E(500, "wake_open", n=5),
          E(600, "wake_open", n=6), E(630, "user_text", text="늦음"),
          E(700, "wake_open", n=7), E(710, "wake_open", rescue=True, text="x"), E(711, "user_text", text="b"),
          E(800, "wake_open", n=8), E(805, "wake_open", n=9), E(806, "user_text", text="c")]
    a = run(ev)["after_wake"]
    assert a == {"heard": 3, "locate": 1, "quiet": 1, "silent": 1, "none": 3}


def test_ghost_split_followup_after_wake():
    ev = [E(10, "wake_open", n=1), E(12, "listen_empty_free"), E(12.4, "listen_ghost", speech=0.0, rms=0.0),
          E(100, "listen_ghost", speech=0.3, rms=0.01), E(101, "listen_ghost", speech=0.2, rms=0.01)]
    r = run(ev)
    assert (r["ghost"], r["ghost_after_wake"], r["ghost_followup"]) == (3, 1, 2)


def test_relisten_overlap_flagged():
    from tools.scene.scenes import Scene
    s1 = Scene("s1", 1, 2, "wake", [], "time", None)
    s2 = Scene("s2", 3, 4, "wake", [], "time", None)
    s3 = Scene("s3", 5, 6, "wake", [], "time", None)
    causes = {"s1": [Cause("질문 겹침", "확정")], "s2": [Cause("정상: 확정 직후 같은 말", "정상")], "s3": []}
    r = tally([E(1, "relisten_drop", node="llm"), E(2, "relisten_drop", node="llm")], [s1, s2, s3], causes)
    assert r["relisten_drop"] == 2 and r["overlap"] == 1 and r["flagged"] == ["s1"]


def test_ctx_at_uses_closest_startup_line_else_defaults():
    from src import wakeword_monitor as wm
    a = E(10, "startup", text="호출 문턱 0.70 · 수음 최소 0.0300 · 접근 대답 최소 0.0220 · 끝")
    b = E(20, "startup", text="호출 문턱 0.60 · 수음 최소 0.0200 · 접근 대답 최소 0.0140 · 끝")
    c = E(25, "startup", text="AGC 목표 레벨 설정: 0.007")
    evs = [a, b, c]
    assert ctx_at(evs, 15)["wake_threshold"] == 0.70 and ctx_at(evs, 15)["listen_min_rms"] == 0.03
    x = ctx_at(evs, 30)
    assert (x["wake_threshold"], x["listen_min_rms"], x["approach_min_rms"]) == (0.60, 0.02, 0.014)
    d = ctx_at(evs, 5)
    assert d["wake_threshold"] == wm.WAKE_THRESHOLD_DEFAULT and d["listen_min_rms"] == wm.LISTEN_MIN_RMS

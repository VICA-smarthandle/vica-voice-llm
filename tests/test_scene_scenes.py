"""대화 장면 분석기 — 장면 묶기와 대화 단계 시계열 (설계 2026-11-20 5.1)."""
import shutil
from pathlib import Path

import pytest

from tools.scene.events import Event, read_events
from tools.scene.scenes import Scene, build_scenes, state_at

FIXTURE = Path(__file__).parent / "fixtures" / "scene" / "1104_sample.log"


@pytest.fixture(scope="module")
def old_events(tmp_path_factory):
    d = tmp_path_factory.mktemp("log")
    shutil.copy(FIXTURE, d / "python3_1_1.log")
    return read_events(0, 2e9, log_dir=d, relaunch_dir=d / "none", journal=False)


def _scene_with(scenes, kind, pred):
    hits = [s for s in scenes if any(e.kind == kind and pred(e) for e in s.events)]
    assert len(hits) == 1, [s.id for s in hits]
    return hits[0]


def test_scene_1507_contains_both_forwards(old_events):
    s = _scene_with(build_scenes(old_events), "forward_answer", lambda e: True)
    users = [e for e in s.events if e.kind == "user_text" and e.data["text"] == "아니."]
    plays = [e for e in s.events if e.kind == "tts_play" and e.data["text"] == "여기까지 안내를 마칠까요?"]
    assert len(users) == 2 and len(plays) == 2


def test_scene_1118_has_two_mission_questions(old_events):
    s = _scene_with(build_scenes(old_events), "mis_say", lambda e: e.data["text"] == "안내를 취소할까요?")
    says = [e.data["text"] for e in s.events if e.kind == "mis_say"]
    assert "안내를 취소할까요?" in says
    assert any(t.startswith("현장 안전 지원 휴머노이드에 도착했습니다.") for t in says)


def test_old_logs_linked_by_time(old_events):
    scenes = build_scenes(old_events)
    assert scenes and all(s.linked_by == "time" and s.question_id is None for s in scenes)
    assert all(s.trigger in ("wake", "question") for s in scenes)
    assert all(s.start < s.end or s.start == s.end for s in scenes)
    assert [s.start for s in scenes] == sorted(s.start for s in scenes)


def test_scene_ids_format_and_unique(old_events):
    scenes = build_scenes(old_events)
    import re
    assert all(re.fullmatch(r"\d{8}-\d{6}-(wake|question)(-\d+)?", s.id) for s in scenes)
    assert len({s.id for s in scenes}) == len(scenes)


def _ev(t, node, kind, **data):
    return Event(t, node, kind, data, "")


def test_question_id_link():
    t = 1_800_000_000.0
    evs = [_ev(t, "wake", "listen_reserve"),
           _ev(t + 0.5, "mis", "q_open", id="q7", qkind="confirm", candidates=1, requested=True, text="x"),
           _ev(t + 3, "llm", "q_voice", id="q7", qkind="answer", age=1.0),
           _ev(t + 4, "mis", "q_close", id="q7", reason="answer", open_s=3.5)]
    scenes = build_scenes(evs)
    assert len(scenes) == 1                      # 예약과 q_open 은 한 장면
    s = scenes[0]
    assert s.linked_by == "question_id" and s.question_id == "q7" and s.trigger == "question"
    assert len(s.events) == 4


def test_reserve_and_q_open_far_apart_make_two_scenes():
    t = 1_800_000_000.0
    evs = [_ev(t, "wake", "listen_reserve"), _ev(t + 5, "mis", "q_open", id="q1", qkind="c", candidates=0,
                                                   requested=False, text="")]
    assert len(build_scenes(evs)) == 2


def test_wake_open_backs_up_to_wake_fire_and_ends_at_next_start():
    t = 1_800_000_000.0
    evs = [_ev(t, "wake", "wake_fire", peak=0.9, threshold=0.7), _ev(t + 0.001, "wake", "wake_open", n=1),
           _ev(t + 2, "wake", "user_text", text="a"),
           _ev(t + 20, "wake", "wake_open", n=2), _ev(t + 21, "wake", "user_text", text="b")]
    a, b = build_scenes(evs)
    assert a.trigger == "wake" and a.start == t and a.end == t + 20 and len(a.events) == 3
    assert b.start == t + 20 and b.events[0].kind == "wake_open"


def test_events_before_first_start_dropped():
    t = 1_800_000_000.0
    evs = [_ev(t, "tts", "startup"), _ev(t + 10, "wake", "wake_open", n=1)]
    assert [len(s.events) for s in build_scenes(evs)] == [1]
    assert build_scenes([]) == []


def test_state_at_asking_next(old_events):
    t_arrive = next(e.t for e in old_events if e.kind == "state" and e.data.get("to") == "asking_next")
    assert state_at(old_events, t_arrive + 0.001) == "asking_next"
    assert state_at(old_events, t_arrive + 0.5) == "asking_next"
    first = next(e.t for e in old_events if e.kind == "state" and e.data.get("via") in ("mission", "handle"))
    assert state_at(old_events, first - 1) == ""
    assert state_at([], 5.0) == ""


def test_state_at_ignores_estop_central_states():
    evs = [_ev(1.0, "mis", "state", to="navigating", via="mission", **{"from": "idle"}),
           _ev(2.0, "mis", "state", to="ESTOP_LATCHED", via="estop")]
    assert state_at(evs, 3.0) == "navigating"


def _qo(t, qid, text=""):
    return _ev(t, "mis", "q_open", id=qid, qkind="c", candidates=0, requested=False, text=text)


def test_different_q_open_ids_split_even_after_user_text():
    t = 1_800_000_000.0
    evs = [_qo(t, "q1"), _ev(t + 10, "wake", "user_text", text="a"), _qo(t + 30, "q2"),
           _ev(t + 35, "wake", "user_text", text="b")]
    a, b = build_scenes(evs)
    assert (a.question_id, b.question_id) == ("q1", "q2")
    assert a.linked_by == b.linked_by == "question_id"
    assert [e.data.get("text") for e in a.events if e.kind == "user_text"] == ["a"]


def test_wake_with_q_open_merges_and_keeps_wake_trigger():
    t = 1_800_000_000.0
    evs = [_ev(t, "wake", "wake_open", n=1), _qo(t + 0.01, "q3", "네?"), _ev(t + 3, "wake", "user_text", text="a")]
    (s,) = build_scenes(evs)
    assert s.trigger == "wake" and s.question_id == "q3" and s.linked_by == "question_id"


def test_wake_scene_without_q_open_stays_time_linked():
    t = 1_800_000_000.0
    (s,) = build_scenes([_ev(t, "wake", "wake_open", n=1), _ev(t + 3, "wake", "user_text", text="a")])
    assert s.linked_by == "time" and s.question_id is None


def test_old_log_reserve_after_15s_splits():
    t = 1_800_000_000.0
    evs = [_ev(t, "wake", "listen_reserve"), _ev(t + 5, "wake", "user_text", text="a"),
           _ev(t + 25, "wake", "listen_reserve")]
    assert len(build_scenes(evs)) == 2
    evs[2] = _ev(t + 12, "wake", "listen_reserve")
    assert len(build_scenes(evs)) == 1


def test_reserve_right_after_q_open_does_not_split():
    t = 1_800_000_000.0
    evs = [_ev(t, "wake", "wake_open", n=1), _qo(t + 30, "q9"), _ev(t + 30.01, "wake", "listen_reserve")]
    assert len(build_scenes(evs)) == 2

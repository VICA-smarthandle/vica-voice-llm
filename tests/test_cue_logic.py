"""청각 안내 판정 로직 검증 (소리·ROS 없이)."""
from __future__ import annotations

import json

from src.cue_logic import (
    DIRECTION_LEFT,
    DIRECTION_NONE,
    DIRECTION_RIGHT,
    GUIDANCE_ARRIVED_EVENT,
    PHASE_COMPLETE,
    PHASE_NOW,
    PHASE_PREPARE,
    TurnAnnouncer,
    parse_goal_event,
)
from src.replies import TURN_LEFT, TURN_RIGHT


# ---- 회전 안내 (신중 모드: 지속 확인 + 도착 근접 억제) ---------------------------
#
# 실주행(2026-08-26)에서 잔 보정 흔들림마다 "우회전할게요"가 나와 부정확한
# 정보가 쌓였다. 신호를 받아도 hold_sec 동안 회전이 계속될 때만 말하고,
# 잔여거리 5 m 이하(신선한 값)에서는 도착 시퀀스와 겹치지 않게 침묵한다.


def _speak_after_hold(ann, seq=1, direction=DIRECTION_LEFT, t0=0.0):
    ann.on_turn(direction, PHASE_NOW, seq, now=t0)
    return ann.poll(t0 + ann.hold_sec + 0.05)


def test_sustained_turn_speaks_once_after_hold():
    ann = TurnAnnouncer(hold_sec=0.8)
    ann.on_turn(DIRECTION_LEFT, PHASE_NOW, 1, now=0.0)
    assert ann.poll(0.5) is None                 # 아직 확신 없음
    assert ann.poll(0.9) == TURN_LEFT            # 0.8초 지속 -> 발화
    assert ann.poll(1.2) is None                 # 같은 회전은 한 번만


def test_brief_correction_stays_silent():
    """잔 보정: 신호 직후 COMPLETE 가 오면 회전이 아니었던 것 — 침묵."""
    ann = TurnAnnouncer(hold_sec=0.8)
    ann.on_turn(DIRECTION_RIGHT, PHASE_NOW, 3, now=0.0)
    ann.on_turn(DIRECTION_RIGHT, PHASE_COMPLETE, 3, now=0.4)
    assert ann.poll(0.9) is None
    # 같은 회차의 늦은 신호가 되살아나면 안 된다
    ann.on_turn(DIRECTION_RIGHT, PHASE_NOW, 3, now=1.0)
    assert ann.poll(1.9) is None


def test_near_goal_is_silent():
    """잔여 5 m 이하(신선)면 침묵 — 도착 정렬 회전이 도착 멘트를 밀어내지 않게."""
    ann = TurnAnnouncer(hold_sec=0.8)
    ann.set_distance(4.0, now=0.0)
    assert _speak_after_hold(ann, seq=1, t0=0.1) is None
    # 그 회차는 소진 — 거리값이 낡아져도 같은 회전을 뒤늦게 말하지 않는다
    assert ann.poll(10.0) is None


def test_far_goal_speaks():
    ann = TurnAnnouncer(hold_sec=0.8)
    ann.set_distance(12.0, now=0.0)
    assert _speak_after_hold(ann, seq=1, t0=0.1) == TURN_LEFT


def test_stale_distance_does_not_suppress():
    """거리값이 3초 넘게 낡았으면 게이트를 접는다 — 모르는 값으로 침묵하지 않는다."""
    ann = TurnAnnouncer(hold_sec=0.8)
    ann.set_distance(4.0, now=0.0)
    assert _speak_after_hold(ann, seq=1, t0=10.0) == TURN_LEFT


def test_unknown_distance_speaks():
    ann = TurnAnnouncer(hold_sec=0.8)
    assert _speak_after_hold(ann, seq=1) == TURN_LEFT


def test_only_now_phase_arms():
    """PREPARE 는 2단계 몫, COMPLETE 는 끝났다는 뜻 — 발화를 걸지 않는다."""
    ann = TurnAnnouncer(hold_sec=0.8)
    ann.on_turn(DIRECTION_LEFT, PHASE_PREPARE, 1, now=0.0)
    ann.on_turn(DIRECTION_LEFT, PHASE_COMPLETE, 2, now=0.0)
    assert ann.poll(5.0) is None


def test_stale_source_is_ignored_but_not_consumed():
    """/odom stale 신호는 무시하되 회차를 소진하지 않는다 — 회복되면 말해야 한다."""
    ann = TurnAnnouncer(hold_sec=0.8)
    ann.on_turn(DIRECTION_LEFT, PHASE_NOW, 3, now=0.0, source_stale=True)
    assert ann.poll(1.0) is None
    ann.on_turn(DIRECTION_LEFT, PHASE_NOW, 3, now=1.0)
    assert ann.poll(1.9) == TURN_LEFT


def test_direction_none_is_ignored():
    ann = TurnAnnouncer(hold_sec=0.8)
    ann.on_turn(DIRECTION_NONE, PHASE_NOW, 1, now=0.0)
    assert ann.poll(1.0) is None


def test_right_turn_speaks_right():
    ann = TurnAnnouncer(hold_sec=0.8)
    assert _speak_after_hold(ann, seq=1, direction=DIRECTION_RIGHT) == TURN_RIGHT


def test_reset_allows_same_sequence_again():
    ann = TurnAnnouncer(hold_sec=0.8)
    assert _speak_after_hold(ann, seq=1) == TURN_LEFT
    ann.reset()
    assert _speak_after_hold(ann, seq=1, t0=5.0) == TURN_LEFT


# ---- /vica_goal_event 파싱 -----------------------------------------------------


def test_parse_goal_event_extracts_event_from_mission_json():
    """정본(mission_manager_node._publish_goal_event)과 같은 모양의 payload."""
    payload = json.dumps(
        {
            "event": "goal_succeeded",
            "map_id": "vica_map_0815",
            "location_id": "room",
            "destination_id": "room",
            "name": "방",
            "x": 1.0,
            "y": 2.0,
            "yaw": 90.0,
            "reason": "",
            "timestamp": "2026-08-16T12:00:00",
        },
        ensure_ascii=False,
    )
    assert parse_goal_event(payload) == GUIDANCE_ARRIVED_EVENT


def test_parse_goal_event_rejects_plain_event_name():
    """옛 평문 형식은 계약이 아니다 — 몰래 허용하면 계약이 다시 흐려진다."""
    assert parse_goal_event("goal_succeeded") is None


def test_parse_goal_event_returns_none_for_malformed_payloads():
    """잘못된 payload 는 무시가 정답이다. 예외로 노드를 죽이면 안 된다."""
    for payload in (
        "",
        "{broken",
        "[1, 2]",
        '"goal_succeeded"',
        '{"no_event": 1}',
        '{"event": 3}',
        '{"event": null}',
    ):
        assert parse_goal_event(payload) is None, payload


def test_parse_goal_event_passes_through_any_event_name():
    """goal_sent 같은 무관 이벤트의 필터링은 호출자 몫이다."""
    assert parse_goal_event('{"event": "goal_sent"}') == "goal_sent"

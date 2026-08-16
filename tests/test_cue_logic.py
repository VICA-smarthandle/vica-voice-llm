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
    GreetingState,
    TurnAnnouncer,
    parse_goal_event,
)
from src.replies import TURN_LEFT, TURN_RIGHT


# ---- 회전 안내 ----------------------------------------------------------------


def test_left_and_right_are_announced():
    ann = TurnAnnouncer()
    assert ann.on_turn(DIRECTION_LEFT, PHASE_NOW, sequence_id=1) == TURN_LEFT
    assert ann.on_turn(DIRECTION_RIGHT, PHASE_NOW, sequence_id=2) == TURN_RIGHT


def test_same_turn_is_announced_once():
    """한 번 도는 동안 신호가 여러 번 와도 말은 한 번만."""
    ann = TurnAnnouncer()
    assert ann.on_turn(DIRECTION_LEFT, PHASE_NOW, sequence_id=7) == TURN_LEFT
    assert ann.on_turn(DIRECTION_LEFT, PHASE_NOW, sequence_id=7) is None
    assert ann.on_turn(DIRECTION_LEFT, PHASE_NOW, sequence_id=7) is None
    # 다음 회전은 다시 말한다
    assert ann.on_turn(DIRECTION_LEFT, PHASE_NOW, sequence_id=8) == TURN_LEFT


def test_only_now_phase_is_announced():
    """PREPARE 는 2단계(경로 예고) 몫이고, COMPLETE 는 끝났다는 뜻이다."""
    ann = TurnAnnouncer()
    assert ann.on_turn(DIRECTION_LEFT, PHASE_PREPARE, sequence_id=1) is None
    assert ann.on_turn(DIRECTION_LEFT, PHASE_COMPLETE, sequence_id=1) is None


def test_stale_source_is_not_announced():
    """/odom 미수신이면 방향이 판단 불가다. 틀린 방향을 말하느니 침묵한다."""
    ann = TurnAnnouncer()
    assert ann.on_turn(DIRECTION_LEFT, PHASE_NOW, 1, source_stale=True) is None


def test_direction_none_is_not_announced():
    ann = TurnAnnouncer()
    assert ann.on_turn(DIRECTION_NONE, PHASE_NOW, sequence_id=1) is None


def test_reset_allows_same_sequence_again():
    """안내가 끝나면 sequence_id 가 1부터 다시 시작할 수 있다."""
    ann = TurnAnnouncer()
    assert ann.on_turn(DIRECTION_LEFT, PHASE_NOW, sequence_id=1) == TURN_LEFT
    ann.reset()
    assert ann.on_turn(DIRECTION_LEFT, PHASE_NOW, sequence_id=1) == TURN_LEFT


def test_stale_does_not_consume_the_sequence():
    """stale 로 걸러진 회전은 '말했다'로 치지 않는다 — 회복되면 말해야 한다."""
    ann = TurnAnnouncer()
    assert ann.on_turn(DIRECTION_LEFT, PHASE_NOW, 3, source_stale=True) is None
    assert ann.on_turn(DIRECTION_LEFT, PHASE_NOW, 3) == TURN_LEFT


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


# ---- 첫 호출 인사 --------------------------------------------------------------


def test_first_wake_greets():
    g = GreetingState()
    assert g.on_wake(0.0) is True


def test_second_wake_in_same_guidance_does_not_greet():
    g = GreetingState()
    assert g.on_wake(0.0) is True
    g.on_user_spoke(1.0)              # 발화가 이어져 인사가 성립
    assert g.on_wake(5.0) is False


def test_wake_without_speech_does_not_count_as_greeting():
    """잘못 부르고 떠난 사람 뒤의 다음 사용자도 인사를 받아야 한다."""
    g = GreetingState()
    assert g.on_wake(0.0) is True     # A 가 부르고
    # A 가 아무 말 없이 떠남 (on_user_spoke 없음)
    assert g.on_wake(5.0) is True     # B 가 불렀다 — 다시 인사


def test_guidance_end_restores_greeting():
    g = GreetingState()
    g.on_wake(0.0)
    g.on_user_spoke(1.0)
    assert g.on_wake(2.0) is False
    g.on_guidance_ended()             # 도착·취소·실패
    assert g.on_wake(3.0) is True     # 다음 사용자


def test_long_silence_restores_greeting():
    """안내가 시작되지 않은 채 방치된 경우의 안전망 (대화 끊김 판정)."""
    g = GreetingState(idle_reset_sec=180.0)
    g.on_wake(0.0)
    g.on_user_spoke(1.0)
    assert g.on_wake(100.0) is False   # 아직 같은 대화
    assert g.on_wake(300.0) is True    # 3분 넘게 조용했다 = 떠났다


def test_activity_extends_the_session():
    """대화가 이어지는 동안에는 끊김 판정이 걸리지 않는다."""
    g = GreetingState(idle_reset_sec=180.0)
    g.on_wake(0.0)
    g.on_user_spoke(1.0)
    for t in (100.0, 200.0, 300.0):    # 100초 간격으로 계속 대화
        assert g.on_wake(t) is False
        g.on_user_spoke(t + 1.0)

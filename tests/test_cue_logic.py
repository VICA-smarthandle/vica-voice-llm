"""청각 안내 판정 로직 검증 (소리·ROS 없이)."""
from __future__ import annotations

import json

from src.cue_logic import GUIDANCE_ARRIVED_EVENT, parse_goal_event


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

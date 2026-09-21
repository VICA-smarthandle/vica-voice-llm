"""pydantic 스키마 <-> ROS2 커스텀 메시지(vica_interfaces) 변환.

이 모듈은 vica_interfaces 를 import 하므로 ROS2 노드에서만 사용한다.
사용 전 `source ../vica_ros2_ws/install/setup.bash` 가 필요하다
(정본 메시지 패키지 — docs/ros2-interface.md 3절).
CLI(main.py)는 이 모듈을 import 하지 않는다.
"""
from __future__ import annotations

from vica_interfaces.msg import EmergencyEvent as EmergencyEventMsg
from vica_interfaces.msg import RobotState as RobotStateMsg
from vica_interfaces.msg import VicaIntent as VicaIntentMsg

from .schema import EmergencyEvent, RobotState, VicaIntent


def emergency_to_msg(event: EmergencyEvent) -> EmergencyEventMsg:
    """pydantic EmergencyEvent -> ROS2 EmergencyEvent 메시지."""
    msg = EmergencyEventMsg()
    msg.keyword = event.keyword
    msg.source_text = event.source_text
    msg.detected_at = float(event.detected_at)
    return msg


def intent_to_msg(intent: VicaIntent) -> VicaIntentMsg:
    """pydantic VicaIntent -> ROS2 VicaIntent 메시지. (None 은 빈 문자열로)"""
    msg = VicaIntentMsg()
    msg.intent = intent.intent
    msg.destination_candidate = intent.destination_candidate or ""
    msg.matched_destination_id = intent.matched_destination_id or ""
    msg.confidence = float(intent.confidence)
    msg.need_confirm = intent.need_confirm
    msg.reply = intent.reply
    msg.safety_flag = intent.safety_flag
    msg.wait_minutes = int(intent.wait_minutes)
    return msg


def msg_to_intent(msg: VicaIntentMsg) -> VicaIntent:
    """ROS2 VicaIntent 메시지 -> pydantic VicaIntent. (빈 문자열은 None 으로)"""
    return VicaIntent(
        intent=msg.intent,
        destination_candidate=msg.destination_candidate or None,
        matched_destination_id=msg.matched_destination_id or None,
        confidence=msg.confidence,
        need_confirm=msg.need_confirm,
        reply=msg.reply,
        safety_flag=msg.safety_flag,
        wait_minutes=msg.wait_minutes,
    )


def msg_to_robot_state(msg: RobotStateMsg) -> RobotState:
    """ROS2 RobotState 메시지 -> pydantic RobotState. (-1 층은 '알 수 없음' = None)

    대장 칸(P1)은 옛 미션 메시지에 없을 수 있어 getattr 기본값으로 받는다 — 그러면
    ledger_view 가 빈 블록을 돌려주고 노드는 goal-event 상황판으로 폴백한다.
    """
    return RobotState(
        current_floor=None if msg.current_floor < 0 else msg.current_floor,
        current_building=msg.current_building,
        is_moving=msg.is_moving,
        is_paused=msg.is_paused,
        dialog_state=str(getattr(msg, "dialog_state", "") or ""),
        place_here=str(getattr(msg, "place_here", "") or ""),
        place_here_dist_m=float(getattr(msg, "place_here_dist_m", -1.0)),
        active_destination=str(getattr(msg, "active_destination", "") or ""),
        last_destination=str(getattr(msg, "last_destination", "") or ""),
        last_arrived_age_sec=int(getattr(msg, "last_arrived_age_sec", -1)),
        aborted_destination=str(getattr(msg, "aborted_destination", "") or ""),
        wait_minutes=int(getattr(msg, "wait_minutes", -1)),
        wait_left_sec=int(getattr(msg, "wait_left_sec", -1)),
        battery_pct=int(getattr(msg, "battery_pct", -1)),
    )


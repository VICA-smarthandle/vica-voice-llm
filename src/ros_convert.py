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
    )


def msg_to_robot_state(msg: RobotStateMsg) -> RobotState:
    """ROS2 RobotState 메시지 -> pydantic RobotState. (-1 층은 '알 수 없음' = None)"""
    return RobotState(
        current_floor=None if msg.current_floor < 0 else msg.current_floor,
        current_building=msg.current_building,
        is_moving=msg.is_moving,
    )


"""pydantic 스키마 <-> ROS2 커스텀 메시지(vica_interfaces) 변환."""
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
    if hasattr(msg, "speaker_verdict"):
        msg.speaker_verdict = intent.speaker_verdict
        msg.speaker_score = float(intent.speaker_score)
        msg.speaker_score_joined = float(intent.speaker_joined)
    if supports_ask_fields(msg):
        msg.ask_text = intent.ask_text
        msg.ask_kind = intent.ask_kind
        msg.ask_option_ids = list(intent.ask_option_ids)
    if hasattr(msg, "question_id"):
        msg.question_id = intent.question_id
        msg.answer = intent.answer
        msg.answer_value = intent.answer_value
    return msg


def supports_ask_fields(msg) -> bool:
    """이 VicaIntent 메시지에 질문 넘김 칸이 있나 — 없으면(옛 설치본) 음성이 질문을 직접 말한다."""
    return hasattr(msg, "ask_text")


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
    """ROS2 RobotState 메시지 -> pydantic RobotState. (-1 층은 '알 수 없음' = None)"""
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
        door_side=str(getattr(msg, "door_side", "") or ""),
        wait_place=str(getattr(msg, "wait_place", "") or ""),
    )

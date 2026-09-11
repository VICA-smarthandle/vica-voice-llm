"""VICA 음성/LLM 파이프라인의 데이터 모양(스키마) 정의."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class DestinationPose(BaseModel):
    """목적지의 지도상 위치/방향. 실제 좌표는 calibration 단계에서 채운다."""

    frame_id: str = "map"
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


class DestinationData(BaseModel):
    """목적지 1개 정보. config/destinations.yaml 한 항목에 대응한다."""

    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    category1: str = ""
    category2: str = ""
    building: str = ""
    floor: int = 0
    room: Optional[str] = None
    owner: str = ""
    authorization: str = "public"
    is_approachable: bool = True
    unavailable_reason: str = ""
    pose: DestinationPose = Field(default_factory=DestinationPose)
    confirm_prompt: str = ""
    arrival_message: str = ""


VicaIntentType = Literal[
    "navigate", "question", "clarify", "unknown", "cancel", "pause", "resume",
    "affirm", "deny",
    "wait", "finish",
]
SafetyFlag = Literal["normal", "emergency"]


class VicaIntent(BaseModel):
    """LLM 파이프라인의 최종 출력. state machine에 전달되는 '제안'이다."""

    intent: VicaIntentType
    destination_candidate: Optional[str] = None
    matched_destination_id: Optional[str] = None
    confidence: float = 0.0
    need_confirm: bool = True
    reply: str = ""
    safety_flag: SafetyFlag = "normal"
    wait_minutes: int = -1


class EmergencyEvent(BaseModel):
    """상시 긴급어 감지 이벤트. Safety Supervisor / State Machine 에 전달된다."""

    keyword: str
    source_text: str
    detected_at: float


class RobotState(BaseModel):
    """로봇의 현재 상태. ROS2 연결 전에는 더미 값을 쓴다."""

    current_floor: Optional[int] = None
    current_building: str = ""
    is_moving: bool = False


def should_forward_intent(intent) -> bool:
    """이 intent 를 /vica/intent 로 미션에 보낼 것인가 (ROS 무관 순수 판정)."""
    return not (
        getattr(intent, "intent", "") == "resume"
        and bool(getattr(intent, "need_confirm", False))
    )

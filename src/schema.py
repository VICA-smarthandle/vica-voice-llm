"""VICA 음성/LLM 파이프라인의 데이터 모양(스키마) 정의.

이 파일은 '데이터의 모양'만 정의한다. 실제 로직은 다른 모듈에 둔다.
- DestinationPose / DestinationData: 목적지 1개 정보 (입력 계약)
- VicaIntent: LLM 파이프라인의 최종 출력 (출력 계약)

자세한 설계는 docs/design.md 참고.
"""
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


# intent 종류와 안전 플래그는 정해진 값만 허용한다 (오타/임의값 방지).
# cancel/pause/resume 은 '실행'이 아니라 '제안'이다 — 코드가 확인 질문("네")을
# 검증한 뒤에만 need_confirm=False 로 내려가고, 실제 실행은 Mission Manager 의
# MissionCommand 서비스(관리자 앱과 같은 경로)가 상태를 보고 최종 판정한다.
# 즉시 정지(멈춰·정지)는 여기 없다 — 긴급어 필터가 LLM 이전에 처리한다.
VicaIntentType = Literal[
    "navigate", "question", "clarify", "unknown", "cancel", "pause", "resume",
    # 로봇이 직전에 던진 질문에 대한 짧은 답 (2026-08-25, 사람 접근 인수).
    # 어느 질문의 답인지는 담지 않는다 — Mission 이 상태로 판정한다.
    # 계약 정본: vica_ros2_ws vica_interfaces/msg/VicaIntent.msg 의 affirm/deny 절.
    "affirm", "deny",
]
SafetyFlag = Literal["normal", "emergency"]


class VicaIntent(BaseModel):
    """LLM 파이프라인의 최종 출력. state machine에 전달되는 '제안'이다.

    LLM은 로봇을 직접 제어하지 않는다. 이 JSON은 이동 명령이 아니라 해석 결과다.

    - destination_candidate: LLM이 채운다 (사용자가 말한 목적지 표현)
    - matched_destination_id: 파이썬 코드가 채운다 (실제 목적지 id)
    """

    intent: VicaIntentType
    destination_candidate: Optional[str] = None
    matched_destination_id: Optional[str] = None
    confidence: float = 0.0
    need_confirm: bool = True
    reply: str = ""
    safety_flag: SafetyFlag = "normal"


class EmergencyEvent(BaseModel):
    """상시 긴급어 감지 이벤트. Safety Supervisor / State Machine 에 전달된다.

    LLM 을 거치지 않는 안전 경로의 출력이다 (CLAUDE.md Phase 4).
    """

    keyword: str  # 매칭된 긴급어 (예: "멈춰")
    source_text: str  # STT 가 인식한 원본 텍스트
    detected_at: float  # 감지 시각 (time.time())


class RobotState(BaseModel):
    """로봇의 현재 상태. ROS2 연결 전에는 더미 값을 쓴다.

    question intent 답변(예: "지금 몇 층이야?")에 활용한다.
    나중에 ROS2 가 실제 값(층, 건물, 이동 여부 등)을 채운다.
    """

    current_floor: Optional[int] = None
    current_building: str = ""
    is_moving: bool = False

"""VICA 청각 안내 노드 — 회전·도착을 소리와 말로 알린다.

구독: /vica/turn_guide   (vica_interfaces/TurnGuide) — 회전 감지
      /vica_goal_event   (std_msgs/String)          — 안내 시작·끝
발행: /vica/tts_request  (std_msgs/String)          — 회전 안내 문구

회전 안내는 **음 + 말** 이다. 음이 주의를 끌고 말이 뜻을 전한다. 음만으로는 처음
쓰는 사용자가 무슨 뜻인지 모르고, 말만으로는 앞부분을 놓친다.

문구를 축약하지 않는 이유: 방향 안내는 사용자의 안전과 직결된다. 매번 같은
문장이어야 매번 같은 판단을 할 수 있다 (2026-08-05 사용자 결정).

우선순위를 response 로 두는 이유: narration 은 큐 정원 초과 시 가장 먼저 버려진다
(tts_queue._trim). 방향 안내가 사라지면 사용자는 어디로 가는지 모른 채 끌려간다.

안전 원칙: 알리기만 한다. /cmd_vel*, Nav2 goal, CAN 없음.

실행:
    source /opt/ros/humble/setup.bash && source ../vica_ros2_ws/install/setup.bash
    .venv/bin/python -m src.ros_audio_cue_node
"""
from __future__ import annotations

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String
from vica_interfaces.msg import TurnGuide

from . import audio_cue
from .cue_logic import (
    DIRECTION_LEFT,
    GUIDANCE_ARRIVED_EVENT,
    GUIDANCE_END_EVENTS,
    TurnAnnouncer,
)
from .replies import TURN_LEFT
from .tts_queue import RESPONSE, build_request


class AudioCueNode(Node):
    def __init__(self) -> None:
        super().__init__("vica_audio_cue")
        self._announcer = TurnAnnouncer()
        self._tts_pub = self.create_publisher(String, "/vica/tts_request", 10)
        self.create_subscription(TurnGuide, "/vica/turn_guide", self._on_turn, 10)
        self.create_subscription(String, "/vica_goal_event", self._on_goal_event, 10)
        self.get_logger().info(
            "VICA 청각 안내 시작 (구독: /vica/turn_guide, /vica_goal_event)")

    def _on_turn(self, msg: TurnGuide) -> None:
        text = self._announcer.on_turn(
            direction=msg.direction,
            phase=msg.phase,
            sequence_id=msg.sequence_id,
            source_stale=msg.source_stale,
        )
        if text is None:
            return

        # 음이 먼저 — 말이 시작되기 전에 귀를 연다.
        audio_cue.play(
            audio_cue.turn_left() if text == TURN_LEFT else audio_cue.turn_right()
        )
        self._tts_pub.publish(String(data=build_request(RESPONSE, text)))
        self.get_logger().info(f"🔀 회전 안내: {text} (seq={msg.sequence_id})")

    def _on_goal_event(self, msg: String) -> None:
        event = (msg.data or "").strip()
        if event == GUIDANCE_ARRIVED_EVENT:
            # 도착 안내 멘트는 Mission Manager 가 말한다. 여기서는 그 앞에 음만
            # 붙여 "이제 안내가 나온다" 를 알린다.
            audio_cue.play(audio_cue.arrived())
            self.get_logger().info("🏁 도착음")
        if event in GUIDANCE_END_EVENTS:
            # 다음 안내의 첫 회전을 놓치지 않도록 중복 억제를 푼다.
            self._announcer.reset()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AudioCueNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

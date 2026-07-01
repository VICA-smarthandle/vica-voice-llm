"""VICA TTS ROS2 노드 (/vica/intent 를 듣고 음성으로 재생).

구독: /vica/intent (std_msgs/String, JSON) - VicaIntent 결과
동작: reply 텍스트를 supertonic 으로 합성해 스피커로 재생한다.

실행:
    source /opt/ros/jazzy/setup.bash
    .venv/bin/python -m src.ros_tts_node
"""
from __future__ import annotations

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from vica_interfaces.msg import VicaIntent as VicaIntentMsg

from .tts import VicaTTS


class TtsNode(Node):
    def __init__(self) -> None:
        super().__init__("vica_tts_node")
        self.get_logger().info("TTS 모델 로드 중...")
        self._tts = VicaTTS()
        self.create_subscription(VicaIntentMsg, "/vica/intent", self._on_intent, 10)
        self.get_logger().info("VICA TTS node 시작 (구독: /vica/intent)")

    def _on_intent(self, msg: VicaIntentMsg) -> None:
        """VicaIntent 메시지의 reply 를 음성으로 재생한다."""
        reply = (msg.reply or "").strip()
        if not reply:
            return
        self.get_logger().info(f"재생: {reply}")
        self._tts.speak(reply)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TtsNode()
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

"""VICA TTS ROS2 노드 (/vica/intent 를 듣고 음성으로 재생).

구독: /vica/intent (std_msgs/String, JSON) - VicaIntent 결과
발행: /vica/tts_state (std_msgs/Bool) - 재생 중 표시.
      웨이크워드 노드가 자기 목소리를 오인하지 않도록 재생 구간을 알린다
      (AEC 배선 전 임시 장치 — 통합 설계 D3/D6).
      토픽 이름 정본은 docs/ros2-interface.md 의 /vica/tts_state 다 — dev 브랜치의
      긴급어 감시(ros_emergency_node)도 같은 이름을 구독한다.
동작: reply 텍스트를 supertonic 으로 합성해 스피커로 재생한다.

실행:
    source /opt/ros/humble/setup.bash
    .venv/bin/python -m src.ros_tts_node
"""
from __future__ import annotations

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool
from vica_interfaces.msg import VicaIntent as VicaIntentMsg

from .tts import VicaTTS


class TtsNode(Node):
    def __init__(self) -> None:
        super().__init__("vica_tts_node")
        self.get_logger().info("TTS 모델 로드 중...")
        self._tts = VicaTTS()
        self._state_pub = self.create_publisher(Bool, "/vica/tts_state", 10)
        self.create_subscription(VicaIntentMsg, "/vica/intent", self._on_intent, 10)
        self.get_logger().info(
            "VICA TTS node 시작 (구독: /vica/intent | 발행: /vica/tts_state)")

    def _publish_state(self, speaking: bool) -> None:
        msg = Bool()
        msg.data = speaking
        self._state_pub.publish(msg)

    def _on_intent(self, msg: VicaIntentMsg) -> None:
        """VicaIntent 메시지의 reply 를 음성으로 재생한다."""
        reply = (msg.reply or "").strip()
        if not reply:
            return
        self.get_logger().info(f"재생: {reply}")
        self._publish_state(True)
        try:
            self._tts.speak(reply)
        finally:
            self._publish_state(False)  # 예외에도 반드시 해제 (웨이크워드 fail-safe 와 이중 방어)


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

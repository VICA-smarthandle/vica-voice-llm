"""VICA TTS ROS2 노드 (/vica/intent 를 듣고 음성으로 재생).

구독: /vica/intent (std_msgs/String, JSON) - VicaIntent 결과
발행: /vica/tts_active (std_msgs/Bool) - 재생 중 표시.
      웨이크워드 노드가 자기 목소리를 오인하지 않도록 재생 구간을 알린다
      (AEC 배선 전 임시 장치 — 통합 설계 D3/D6).
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
        self._pub_active = self.create_publisher(Bool, "/vica/tts_active", 10)
        self.create_subscription(VicaIntentMsg, "/vica/intent", self._on_intent, 10)
        self.get_logger().info("VICA TTS node 시작 (구독: /vica/intent)")

    def _set_active(self, active: bool) -> None:
        msg = Bool()
        msg.data = active
        self._pub_active.publish(msg)

    def _on_intent(self, msg: VicaIntentMsg) -> None:
        """VicaIntent 메시지의 reply 를 음성으로 재생한다."""
        reply = (msg.reply or "").strip()
        if not reply:
            return
        self.get_logger().info(f"재생: {reply}")
        self._set_active(True)
        try:
            self._tts.speak(reply)
        finally:
            self._set_active(False)   # 예외에도 반드시 해제 (웨이크워드 fail-safe 와 이중 방어)


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

"""VICA STT ROS2 노드 (마이크 -> /vica/user_text). push-to-talk 개발용."""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .replies import RETRY_PROMPT
from .stt import VicaSTT
from .tts_queue import RESPONSE, build_request


class SttNode(Node):
    def __init__(self) -> None:
        super().__init__("vica_stt_node")
        self._pub = self.create_publisher(String, "/vica/user_text", 10)
        self._tts_pub = self.create_publisher(String, "/vica/tts_request", 10)

    def publish_text(self, text: str) -> None:
        msg = String()
        msg.data = text
        self._pub.publish(msg)
        self.get_logger().info(f"발행 /vica/user_text: '{text}'")

    def ask_retry(self) -> None:
        """인식 결과가 없을 때 다시 말해 달라고 알린다."""
        self._tts_pub.publish(String(data=build_request(RESPONSE, RETRY_PROMPT)))
        self.get_logger().warn("인식 결과 없음 -> 재발화 안내")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SttNode()
    node.get_logger().info("STT 모델 로드 중...")
    stt = VicaSTT()
    node.get_logger().info(
        "VICA STT node 시작 (push-to-talk: 엔터로 녹음, 발행: /vica/user_text)"
    )

    try:
        while rclpy.ok():
            try:
                input("녹음하려면 엔터 (종료 Ctrl+C) > ")
            except EOFError:
                break
            text = stt.listen().strip()
            if not text:
                node.ask_retry()
                continue
            node.publish_text(text)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

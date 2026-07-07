"""VICA STT ROS2 노드 (마이크 -> /vica/user_text).

동작: push-to-talk 로 녹음(엔터 -> 말하기 -> 엔터) -> whisper 로 변환
      -> /vica/user_text 로 발행.

이 노드는 구독이 없으므로 rclpy.spin 대신 입력 루프로 동작한다.

실행:
    source /opt/ros/humble/setup.bash
    .venv/bin/python -m src.ros_stt_node
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .stt import VicaSTT


class SttNode(Node):
    def __init__(self) -> None:
        super().__init__("vica_stt_node")
        self._pub = self.create_publisher(String, "/vica/user_text", 10)

    def publish_text(self, text: str) -> None:
        msg = String()
        msg.data = text
        self._pub.publish(msg)
        self.get_logger().info(f"발행 /vica/user_text: '{text}'")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SttNode()
    node.get_logger().info("STT 모델 로드 중...")
    stt = VicaSTT()
    node.get_logger().info("VICA STT node 시작 (엔터로 녹음, 발행: /vica/user_text)")

    try:
        while rclpy.ok():
            try:
                input("녹음하려면 엔터 (종료 Ctrl+C) > ")
            except EOFError:
                break
            text = stt.listen().strip()
            if not text:
                node.get_logger().warn("인식 결과 없음")
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

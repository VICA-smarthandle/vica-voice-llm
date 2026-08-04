"""VICA 웨이크워드 ROS2 노드 (/vica/wakeword_node) — P1-b.

ros_emergency_node(whisper 상시)와 push-to-talk STT 를 함께 대체하는 마이크 앞단.

발행: /vica/emergency (vica_interfaces/EmergencyEvent)  ← 긴급, LLM 우회 (기존 계약)
      /vica/user_text (std_msgs/String)                 ← 호출 후 발화 (기존 계약)
구독: /vica/tts_state (std_msgs/Bool)                   ← TTS 재생 중 자기 목소리 억제

keyword 는 whisper 전사에서 정확 매칭으로 추출되므로 항상
HARD_EMERGENCY_KEYWORDS 정본 안의 값이다 — 브리지·래치 체인 변경 없음.

안전 원칙: 감지·발행까지만 한다. /cmd_vel*, Nav2 goal, CAN 없음.

실행:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    .venv/bin/python -m src.ros_wakeword_node
"""
from __future__ import annotations

import threading

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String
from vica_interfaces.msg import EmergencyEvent as EmergencyEventMsg

from .ros_convert import emergency_to_msg
from .schema import EmergencyEvent
from .wakeword_monitor import WakewordMonitor


def _ack_beep() -> None:
    """호출 응답음 (시각장애인 사용자용 청각 피드백). 실패해도 감시는 계속된다."""
    try:
        import sounddevice as sd

        t = np.arange(int(0.12 * 44100)) / 44100
        tone = (0.4 * np.sin(2 * np.pi * 880 * t) * np.hanning(len(t))).astype(np.float32)
        sd.play(tone, 44100)
    except Exception:
        pass


class WakewordNode(Node):
    def __init__(self) -> None:
        super().__init__("vica_wakeword_node")
        self._pub_emergency = self.create_publisher(EmergencyEventMsg, "/vica/emergency", 10)
        self._pub_text = self.create_publisher(String, "/vica/user_text", 10)
        self._pub_wake = self.create_publisher(String, "/vica/wake", 10)  # 계측·UI 앵커
        self.create_subscription(Bool, "/vica/tts_state", self._on_tts_state, 10)

        self._monitor = WakewordMonitor(
            on_emergency=self._on_emergency,
            on_user_text=self._on_user_text,
            on_wake=self._on_wake,
        )
        # 마이크 감시 루프는 blocking 이라 별도 스레드 (ros_emergency_node 와 동일 패턴)
        self._thread = threading.Thread(target=self._monitor.run, daemon=True)
        self._thread.start()
        self.get_logger().info(
            "VICA 웨이크워드 감시 시작 (발행: /vica/emergency, /vica/user_text)")

    def _on_emergency(self, event: EmergencyEvent) -> None:
        self._pub_emergency.publish(emergency_to_msg(event))
        self.get_logger().warn(
            f"🚨 긴급 '{event.keyword}' 확정 -> /vica/emergency (인식: {event.source_text!r})")

    def _on_user_text(self, text: str) -> None:
        msg = String()
        msg.data = text
        self._pub_text.publish(msg)
        self.get_logger().info(f"🗣️ 호출 발화 -> /vica/user_text: {text!r}")

    def _on_wake(self) -> None:
        _ack_beep()
        msg = String()
        msg.data = "wake"
        self._pub_wake.publish(msg)
        self.get_logger().info("🙋 비카야 호출 — 청취 창 열림")

    def _on_tts_state(self, msg: Bool) -> None:
        self._monitor.set_muted(bool(msg.data))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WakewordNode()
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

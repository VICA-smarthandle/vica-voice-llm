"""VICA 상시 긴급어 감지 ROS2 노드 (/vica/emergency_monitor_node).

발행: /vica/emergency (vica_interfaces/EmergencyEvent)

마이크를 상시 감시하며 긴급어("멈춰" 등)를 감지해 즉시 발행한다.
LLM 을 거치지 않는 안전 경로다 (CLAUDE.md Phase 4).

안전 원칙:
- 이 노드는 '감지'만 한다. /cmd_vel 발행이나 정지 실행은 하지 않는다.
- 실제 정지는 이 토픽을 구독하는 Safety Supervisor / State Machine 이 한다.

실행:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    .venv/bin/python -m src.ros_emergency_node
"""
from __future__ import annotations

import threading

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from vica_interfaces.msg import EmergencyEvent as EmergencyEventMsg

from .emergency_monitor import EmergencyMonitor
from .ros_convert import emergency_to_msg
from .schema import EmergencyEvent


class EmergencyMonitorNode(Node):
    def __init__(self) -> None:
        super().__init__("vica_emergency_monitor_node")
        self._pub = self.create_publisher(EmergencyEventMsg, "/vica/emergency", 10)
        self._monitor = EmergencyMonitor(on_event=self._on_event)

        # 마이크 감시 루프는 blocking 이라 별도 스레드에서 돌린다.
        self._thread = threading.Thread(target=self._monitor.run, daemon=True)
        self._thread.start()
        self.get_logger().info("VICA 긴급어 상시 감시 시작 (발행: /vica/emergency)")

    def _on_event(self, event: EmergencyEvent) -> None:
        self._pub.publish(emergency_to_msg(event))
        self.get_logger().warn(f"🚨 긴급어 '{event.keyword}' 감지 -> /vica/emergency 발행")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EmergencyMonitorNode()
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

"""VICA 상시 긴급어 감지 ROS2 노드 (/vica/emergency_monitor_node)."""
from __future__ import annotations

import threading

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool
from vica_interfaces.msg import EmergencyEvent as EmergencyEventMsg

from .emergency_monitor import EmergencyMonitor
from .ros_convert import emergency_to_msg
from .schema import EmergencyEvent


class EmergencyMonitorNode(Node):
    def __init__(self) -> None:
        super().__init__("vica_emergency_monitor_node")
        self._pub = self.create_publisher(EmergencyEventMsg, "/vica/emergency", 10)
        self._monitor = EmergencyMonitor(on_event=self._on_event)

        self.create_subscription(Bool, "/vica/tts_state", self._on_tts_state, 10)

        self._thread = threading.Thread(target=self._monitor.run, daemon=True)
        self._thread.start()
        self.get_logger().info(
            "VICA 긴급어 상시 감시 시작 (발행: /vica/emergency, 구독: /vica/tts_state)"
        )

    def _on_tts_state(self, msg: Bool) -> None:
        self._monitor.set_muted(bool(msg.data))

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

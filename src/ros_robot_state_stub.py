"""[개발용 스텁] 로봇 상태 발행 노드.

실제 로봇 대신 더미 RobotState 를 주기적으로 /vica/robot_state 에 발행한다.
로봇 통합 시 이 노드를 '진짜 로봇 상태 퍼블리셔'로 교체한다.

실행:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    .venv/bin/python -m src.ros_robot_state_stub
"""
from __future__ import annotations

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from vica_interfaces.msg import RobotState as RobotStateMsg


class RobotStateStub(Node):
    def __init__(self) -> None:
        super().__init__("vica_robot_state_stub")
        self._pub = self.create_publisher(RobotStateMsg, "/vica/robot_state", 10)
        self.create_timer(2.0, self._tick)  # 2초마다 상태 발행
        self.get_logger().info("[스텁] robot_state 발행 시작 (더미: 별빛관 1층, 정지)")

    def _tick(self) -> None:
        msg = RobotStateMsg()
        msg.current_floor = 1
        msg.current_building = "별빛관"
        msg.is_moving = False
        self._pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RobotStateStub()
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

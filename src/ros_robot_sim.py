"""[SIM ONLY] 가상 로봇 ROS2 노드 — 스텁 2개를 대체하는 주행 시뮬레이터.

로봇 없이 전체 서비스 흐름(호출→LLM→이동→도착 안내→긴급 정지)을 돌리기 위한
개발용 노드다. 실기 통합 시에는 로봇 팀의 state machine / robot_state 로 교체한다.

구독: /vica/intent (이동 판단) · /vica/emergency (즉시 정지+래치)
      /vica/sim/reset (std_msgs/Empty — 래치 해제, 실제의 '관리자 앱 reset' 자리)
발행: /vica/robot_state (1Hz)
      /vica/sim/event (std_msgs/String — move_started:<id> / estopped / arrived:<id>
                        / reset / blocked_estop. 계측 노드의 시각 앵커)
      /vica/intent   (도착 안내 멘트 — TTS 재생용. [SIM ONLY] 실기에서는 로봇 팀
                        경로로 대체. 안내 멘트에 긴급어를 넣지 않는다)

실행:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    .venv/bin/python -m src.ros_robot_sim
"""
from __future__ import annotations

import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Empty, String
from vica_interfaces.msg import EmergencyEvent as EmergencyEventMsg
from vica_interfaces.msg import RobotState as RobotStateMsg
from vica_interfaces.msg import VicaIntent as VicaIntentMsg

from .destination_loader import load_destinations
from .robot_sim import SimRobot
from .ros_convert import msg_to_intent


class RobotSimNode(Node):
    def __init__(self) -> None:
        super().__init__("vica_robot_sim")
        self._sim = SimRobot(load_destinations())
        self.create_subscription(VicaIntentMsg, "/vica/intent", self._on_intent, 10)
        self.create_subscription(EmergencyEventMsg, "/vica/emergency", self._on_emergency, 10)
        self.create_subscription(Empty, "/vica/sim/reset", self._on_reset, 10)
        self._pub_state = self.create_publisher(RobotStateMsg, "/vica/robot_state", 10)
        self._pub_event = self.create_publisher(String, "/vica/sim/event", 10)
        self._pub_intent = self.create_publisher(VicaIntentMsg, "/vica/intent", 10)
        self.create_timer(0.5, self._tick)
        self.create_timer(1.0, self._publish_state)
        self.get_logger().info("[SIM] 가상 로봇 시작 (idle, 별빛관 1층)")

    def _emit(self, event: str) -> None:
        msg = String()
        msg.data = event
        self._pub_event.publish(msg)

    def _on_intent(self, msg: VicaIntentMsg) -> None:
        if not msg.intent:      # 우리가 발행한 도착 안내(reply 전용)는 무시
            return
        result = self._sim.handle_intent(msg_to_intent(msg), time.time())
        if result == "move_started":
            self._emit(f"move_started:{msg.matched_destination_id}")
            self.get_logger().info(f"[SIM] 이동 시작 → {msg.matched_destination_id}")
        elif result == "blocked_estop":
            self._emit("blocked_estop")
            self.get_logger().warn("[SIM] E-stop 래치 중 — 이동 거부 (reset 필요)")

    def _on_emergency(self, msg: EmergencyEventMsg) -> None:
        self._sim.handle_emergency(time.time())
        self._emit("estopped")
        self.get_logger().warn(
            f"[SIM] 🚨 '{msg.keyword}' → 즉시 정지 + 래치 (자동 재개 없음)")
        self._publish_state()

    def _on_reset(self, _msg: Empty) -> None:
        self._sim.reset(time.time())
        self._emit("reset")
        self.get_logger().info("[SIM] reset — idle 복귀 (이전 목적지는 폐기됨)")

    def _tick(self) -> None:
        event = self._sim.tick(time.time())
        if event is None:
            return
        self._emit(f"arrived:{event['dest_id']}")
        self.get_logger().info(f"[SIM] 도착: {event['dest_id']}")
        # 도착 안내 멘트 → TTS. intent 필드를 비워 자기 수신 루프를 막는다.
        out = VicaIntentMsg()
        out.reply = event["message"]
        out.safety_flag = "normal"
        self._pub_intent.publish(out)
        self._publish_state()

    def _publish_state(self) -> None:
        msg = RobotStateMsg()
        msg.current_floor = int(self._sim.floor)
        msg.current_building = self._sim.building
        msg.is_moving = self._sim.is_moving
        self._pub_state.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RobotSimNode()
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

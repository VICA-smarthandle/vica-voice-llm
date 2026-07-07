"""[개발용 스텁] state machine 노드.

/vica/intent 를 받아 '이동 여부 판단' 흐름을 로그로 시뮬레이션한다.

⚠️ 안전 원칙 (CLAUDE.md):
- 이 노드(그리고 LLM)는 /cmd_vel 이나 Nav2 goal 을 직접 보내지 않는다.
- 실제로는 여기서 safety supervisor 확인 후 Nav2 goal 생성 여부를 결정한다.
- 지금은 그 결정 자리를 로그로만 표시한다.

실행:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    .venv/bin/python -m src.ros_state_machine_stub
"""
from __future__ import annotations

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from vica_interfaces.msg import EmergencyEvent as EmergencyEventMsg
from vica_interfaces.msg import VicaIntent as VicaIntentMsg


class StateMachineStub(Node):
    def __init__(self) -> None:
        super().__init__("vica_state_machine_stub")
        self.create_subscription(VicaIntentMsg, "/vica/intent", self._on_intent, 10)
        # 상시 긴급어 감지(Phase 4, LLM 우회 경로)도 구독한다.
        self.create_subscription(EmergencyEventMsg, "/vica/emergency", self._on_emergency, 10)
        self.get_logger().info("[스텁] state machine 시작 (구독: /vica/intent, /vica/emergency)")

    def _on_emergency(self, msg: EmergencyEventMsg) -> None:
        # 실제로는 safety supervisor 가 즉시 정지를 실행한다.
        self.get_logger().warn(
            f"🚨 긴급어 '{msg.keyword}' 수신 -> [실제] 즉시 정지 / safety supervisor override"
        )

    def _on_intent(self, msg: VicaIntentMsg) -> None:
        # 긴급은 최우선. 실제로는 safety supervisor 가 즉시 정지시킨다.
        if msg.safety_flag == "emergency":
            self.get_logger().warn("긴급 감지 -> [실제] 즉시 정지 / safety supervisor override")
            return

        if msg.intent == "navigate" and msg.matched_destination_id and not msg.need_confirm:
            self.get_logger().info(
                f"이동 확정 -> [실제] Nav2 goal 생성 대상: {msg.matched_destination_id}"
            )
        elif msg.intent == "navigate" and msg.need_confirm:
            self.get_logger().info(
                f"확인 대기 -> 사용자 응답 후 진행 예정: {msg.matched_destination_id}"
            )
        else:
            self.get_logger().info(f"이동 없음 (intent={msg.intent})")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StateMachineStub()
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

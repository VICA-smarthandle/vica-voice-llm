"""VICA LLM intent ROS2 노드 (/vica/llm_intent_node).

구독: /vica/user_text   (std_msgs/String)        - 사용자 발화 텍스트(STT 결과 등)
구독: /vica/robot_state (std_msgs/String, JSON)   - 로봇 현재 상태
발행: /vica/intent      (std_msgs/String, JSON)   - VicaIntent 결과(JSON)

안전 원칙 (CLAUDE.md):
- 이 노드는 /cmd_vel 이나 Nav2 goal 을 직접 보내지 않는다.
- VicaIntent 는 state machine 에 전달되는 '제안'일 뿐이다.
- 긴급어는 parse_intent 이전에 emergency_filter 가 처리한다.

실행:
    source /opt/ros/jazzy/setup.bash
    .venv/bin/python -m src.ros_node
"""
from __future__ import annotations

import rclpy
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String
from vica_interfaces.msg import RobotState as RobotStateMsg
from vica_interfaces.msg import VicaIntent as VicaIntentMsg

from .destination_loader import load_destinations
from .emergency_filter import detect_emergency
from .langchain_intent_parser import parse_intent
from .ros_convert import intent_to_msg, msg_to_robot_state
from .schema import RobotState, VicaIntent

MAX_HISTORY = 8  # 최근 메시지만 유지 (대화가 길어져도 프롬프트가 커지지 않게)


class LlmIntentNode(Node):
    def __init__(self) -> None:
        super().__init__("vica_llm_intent_node")
        self._destinations = load_destinations()
        self._robot_state = RobotState()  # robot_state 토픽이 오기 전 기본값
        self._history: list[BaseMessage] = []  # 대화 히스토리 (멀티턴 기억)

        self._intent_pub = self.create_publisher(VicaIntentMsg, "/vica/intent", 10)
        self.create_subscription(String, "/vica/user_text", self._on_user_text, 10)
        self.create_subscription(RobotStateMsg, "/vica/robot_state", self._on_robot_state, 10)

        self.get_logger().info(
            "VICA LLM intent node 시작 (구독: /vica/user_text, /vica/robot_state | 발행: /vica/intent)"
        )

    def _on_robot_state(self, msg: RobotStateMsg) -> None:
        """로봇 상태 메시지를 받아 최신값으로 보관한다."""
        self._robot_state = msg_to_robot_state(msg)

    def _on_user_text(self, msg: String) -> None:
        """발화를 받아 VicaIntent 를 만들어 발행한다."""
        text = msg.data.strip()
        if not text:
            return

        # 1) 긴급어는 LLM 이전에 처리한다 (안전 경로).
        keyword = detect_emergency(text)
        if keyword:
            intent = VicaIntent(
                intent="unknown",
                reply="긴급 정지합니다.",
                need_confirm=False,
                safety_flag="emergency",
            )
            self.get_logger().warn(f"[긴급] '{keyword}' 감지 -> safety_flag=emergency")
        else:
            # 2) 일반 발화는 LLM intent 파서로 해석한다 (대화 히스토리 포함 = 멀티턴).
            intent = parse_intent(
                text,
                self._destinations,
                history=self._history,
                robot_state=self._robot_state,
            )

        # 3) VicaIntent 를 커스텀 메시지로 발행한다 (이동 명령이 아니라 '제안').
        self._intent_pub.publish(intent_to_msg(intent))
        self.get_logger().info(
            f"입력='{text}' -> intent={intent.intent} "
            f"matched={intent.matched_destination_id} safety={intent.safety_flag}"
        )

        # 4) 대화 히스토리를 갱신한다 (다음 발화가 맥락을 기억하도록).
        self._history.append(HumanMessage(text))
        self._history.append(AIMessage(intent.reply))
        self._history[:] = self._history[-MAX_HISTORY:]


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LlmIntentNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():  # 이미 종료된 컨텍스트에 shutdown 을 또 부르지 않는다.
            rclpy.shutdown()


if __name__ == "__main__":
    main()

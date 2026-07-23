"""VICA LLM intent ROS2 노드 (/vica/llm_intent_node).

구독: /vica/user_text   (std_msgs/String)        - 사용자 발화 텍스트(STT 결과 등)
구독: /vica/robot_state (std_msgs/String, JSON)   - 로봇 현재 상태
발행: /vica/intent      (vica_interfaces/VicaIntent) - intent 해석 결과('제안')
발행: /vica/tts_request (std_msgs/String)        - 사용자에게 들려줄 응답

발화 주체를 나누는 이유는 tts_queue.request_for_intent 주석 참고. 요약하면,
navigate 확정 요청의 결과는 Mission Manager 만 알 수 있으므로 그쪽이 말한다.

안전 원칙 (CLAUDE.md):
- 이 노드는 /cmd_vel 이나 Nav2 goal 을 직접 보내지 않는다.
- VicaIntent 는 state machine 에 전달되는 '제안'일 뿐이다.
- 긴급어는 parse_intent 이전에 emergency_filter 가 처리한다.

실행:
    source /opt/ros/humble/setup.bash
    .venv/bin/python -m src.ros_node
"""
from __future__ import annotations

import time

import rclpy
from langchain_core.messages import AIMessage, HumanMessage
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String
from vica_interfaces.msg import RobotState as RobotStateMsg
from vica_interfaces.msg import VicaIntent as VicaIntentMsg

from .destination_loader import load_destinations
from .emergency_filter import EMERGENCY_REPLY, detect_emergency
from .history import ConversationHistory
from .langchain_intent_parser import parse_intent
from .replies import ACK_LISTENING
from .ros_convert import intent_to_msg, msg_to_robot_state
from .schema import RobotState, VicaIntent
from .tts_queue import RESPONSE, build_request, request_for_intent


class LlmIntentNode(Node):
    def __init__(self) -> None:
        super().__init__("vica_llm_intent_node")
        self._destinations = load_destinations()
        self._robot_state = RobotState()  # robot_state 토픽이 오기 전 기본값
        # 멀티턴 기억. 공용 로봇이라 한동안 발화가 없으면 새 대화로 보고 비운다.
        self._history = ConversationHistory()

        self._intent_pub = self.create_publisher(VicaIntentMsg, "/vica/intent", 10)
        self._tts_pub = self.create_publisher(String, "/vica/tts_request", 10)
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

        # 0) 한동안 발화가 없었으면 다른 사용자로 보고 이전 맥락을 버린다.
        #    안 그러면 다음 사람의 "거기로 가줘"가 앞사람 목적지로 해석된다.
        if self._history.begin_turn(time.time()):
            self.get_logger().info("대화가 끊겨 이전 맥락을 비웠다")

        # 1) 긴급어는 LLM 이전에 처리한다 (안전 경로).
        keyword = detect_emergency(text)
        if keyword:
            intent = VicaIntent(
                intent="unknown",
                reply=EMERGENCY_REPLY,
                need_confirm=False,
                safety_flag="emergency",
            )
            self.get_logger().warn(f"[긴급] '{keyword}' 감지 -> safety_flag=emergency")
        else:
            # 1-1) LLM 응답까지는 수 초가 걸린다. 그동안 침묵하면 눈으로 확인할 수
            #      없는 사용자는 로봇이 들었는지 알 수 없다. 먼저 짧게 답한다.
            self._tts_pub.publish(
                String(data=build_request(RESPONSE, ACK_LISTENING))
            )

            # 2) 일반 발화는 LLM intent 파서로 해석한다 (대화 히스토리 포함 = 멀티턴).
            intent = parse_intent(
                text,
                self._destinations,
                history=self._history.messages,
                robot_state=self._robot_state,
            )

        # 3) VicaIntent 를 커스텀 메시지로 발행한다 (이동 명령이 아니라 '제안').
        self._intent_pub.publish(intent_to_msg(intent))

        # 3-1) 이 노드가 말해야 하는 응답만 TTS 로 보낸다.
        #      navigate 확정 요청은 Mission Manager 가 게이트 판단 뒤에 말한다.
        request = request_for_intent(intent)
        if request:
            self._tts_pub.publish(String(data=request))
        self.get_logger().info(
            f"입력='{text}' -> intent={intent.intent} "
            f"matched={intent.matched_destination_id} safety={intent.safety_flag}"
        )

        # 4) 대화 히스토리를 갱신한다 (다음 발화가 맥락을 기억하도록).
        self._history.extend([HumanMessage(text), AIMessage(intent.reply)])


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

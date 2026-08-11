"""VICA LLM intent ROS2 노드 (/vica/llm_intent_node).

구독: /vica/user_text   (std_msgs/String)        - 사용자 발화 텍스트(STT 결과 등)
구독: /vica/robot_state (vica_interfaces/RobotState) - 로봇 현재 상태
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
from pathlib import Path

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
from .mission_command import CancelConfirm
from .replies import ACK_LISTENING
from .ros_convert import intent_to_msg, msg_to_robot_state
from .schema import RobotState, VicaIntent
from .tts_queue import RESPONSE, build_request, request_for_intent


class LlmIntentNode(Node):
    def __init__(self) -> None:
        super().__init__("vica_llm_intent_node")
        self.declare_parameter(
            "destinations_yaml",
            str(
                Path.home()
                / "vica_data"
                / "destinations"
                / "vica_map_0630"
                / "destinations.yaml"
            ),
        )
        self._destinations_path = Path(
            str(self.get_parameter("destinations_yaml").value)
        ).expanduser()
        self._destinations_mtime_ns: int | None = None
        self._destinations = []
        self._reload_destinations_if_changed(force=True)
        self._robot_state = RobotState()  # robot_state 토픽이 오기 전 기본값
        # 멀티턴 기억. 공용 로봇이라 한동안 발화가 없으면 새 대화로 보고 비운다.
        self._history = ConversationHistory()
        # 취소 되묻기 대기. 되묻는 문장은 Mission Manager 가 말하므로 history 로는
        # 알 수 없다. 취소를 내보낸 이쪽이 직접 기억한다.
        self._cancel_confirm = CancelConfirm()

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
        self._reload_destinations_if_changed()

        # 0) 한동안 발화가 없었으면 다른 사용자로 보고 이전 맥락을 버린다.
        #    안 그러면 다음 사람의 "거기로 가줘"가 앞사람 목적지로 해석된다.
        if self._history.begin_turn(time.time()):
            self.get_logger().info("대화가 끊겨 이전 맥락을 비웠다")

        # 0-1) 취소 되묻기에 대한 답이면 LLM 을 거치지 않고 그대로 확정한다.
        #      Mission Manager 는 "안내를 취소할까요?"로 되묻지만 그 문장을 말한 것은
        #      저쪽이라 이 노드의 대화 기록에 없다. 되물을 것을 아는 쪽이 기억한다.
        answer = self._cancel_confirm.take_answer(text, time.time())
        if answer is not None:
            if answer:
                # intent=cancel 을 한 번 더 보낸다. Mission Manager 가 이것을
                # 재확인 긍정으로 읽는다(mission_manager_node 의 cancel 분기).
                self._intent_pub.publish(
                    intent_to_msg(
                        VicaIntent(intent="cancel", confidence=1.0, need_confirm=False)
                    )
                )
                self.get_logger().info(f"취소 재확인 긍정 ('{text}') -> intent=cancel")
            else:
                # 철회. Mission Manager 가 즉시 "안내를 계속하겠습니다"로 답한다.
                # 이 값이 없던 동안에는 저쪽 시한(30초)이 지나야 응답이 나왔고,
                # 눈으로 확인할 수 없는 사용자에게 그 30초가 침묵으로 남았다.
                self._intent_pub.publish(
                    intent_to_msg(
                        VicaIntent(
                            intent="cancel_keep", confidence=1.0, need_confirm=False
                        )
                    )
                )
                self.get_logger().info(
                    f"취소 재확인 부정 ('{text}') -> intent=cancel_keep"
                )
            return

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

        # 2-1) 취소를 내보냈다면 Mission Manager 가 곧 되묻는다. 그 답을 받을
        #      준비를 해 둔다 (0-1 참고).
        if intent.intent == "cancel":
            self._cancel_confirm.on_requested(time.time())

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

    def _reload_destinations_if_changed(self, force: bool = False) -> None:
        """저장 노드가 YAML을 교체하면 다음 발화 전에 public catalog를 갱신한다."""
        try:
            mtime_ns = self._destinations_path.stat().st_mtime_ns
        except FileNotFoundError:
            if force or self._destinations:
                self._destinations = []
                self._destinations_mtime_ns = None
                self.get_logger().warn(
                    f"목적지 catalog가 없어 빈 목록을 사용합니다: "
                    f"{self._destinations_path}"
                )
            return
        if not force and mtime_ns == self._destinations_mtime_ns:
            return
        try:
            loaded = load_destinations(self._destinations_path)
        except Exception as exc:
            self.get_logger().error(
                f"목적지 catalog reload 실패, 이전 목록 유지: {exc}"
            )
            return
        self._destinations = [
            destination
            for destination in loaded
            if destination.authorization == "public"
        ]
        self._destinations_mtime_ns = mtime_ns
        self.get_logger().info(
            f"public 목적지 {len(self._destinations)}개 로드: "
            f"{self._destinations_path}"
        )


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

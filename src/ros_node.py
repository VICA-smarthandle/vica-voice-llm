"""VICA LLM intent ROS2 노드 (/vica/llm_intent_node)."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import rclpy
from langchain_core.messages import AIMessage, HumanMessage
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String
from vica_interfaces.msg import RobotState as RobotStateMsg
from vica_interfaces.msg import VicaIntent as VicaIntentMsg

from .destination_loader import load_destinations
from .emergency_filter import detect_emergency
from .history import ConversationHistory
from .langchain_intent_parser import (
    SHORTCUT_REPLIES, is_instant_utterance, parse_intent)
from .replies import expects_answer
from .ros_convert import intent_to_msg, msg_to_robot_state
from .schema import should_forward_intent, RobotState, VicaIntent
from .tts_queue import request_for_intent


FOLLOWUP_CONTEXT_SEC = 40.0


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
        self._robot_state = RobotState()
        self._history = ConversationHistory()

        self._intent_pub = self.create_publisher(VicaIntentMsg, "/vica/intent", 10)
        self._tts_pub = self.create_publisher(String, "/vica/tts_request", 10)
        self._listen_pub = self.create_publisher(Bool, "/vica/listen_request", 10)
        self._thinking_pub = self.create_publisher(Bool, "/vica/thinking", 10)
        self.create_subscription(String, "/vica/user_text", self._on_user_text, 10)
        self.create_subscription(RobotStateMsg, "/vica/robot_state", self._on_robot_state, 10)
        self._followup_until = 0.0
        self.create_subscription(Bool, "/vica/listen_request", self._on_listen_request, 10)
        self.create_subscription(String, "/vica/wake", self._on_wake_signal, 10)

        self.get_logger().info(
            "VICA LLM intent node 시작 (구독: /vica/user_text, /vica/robot_state | 발행: /vica/intent)"
        )
        threading.Thread(target=self._warmup_llm, daemon=True).start()

    def _warmup_llm(self) -> None:
        started = time.monotonic()
        try:
            parse_intent("워밍업", [], history=[])
            self.get_logger().info(
                f"LLM 워밍업 완료 ({time.monotonic() - started:.1f}초)")
        except Exception as exc:
            self.get_logger().warning(f"LLM 워밍업 실패(무시 가능): {exc}")

    def _on_listen_request(self, msg: Bool) -> None:
        if msg.data:
            self._followup_until = time.time() + FOLLOWUP_CONTEXT_SEC
        else:
            self._followup_until = 0.0

    def _on_wake_signal(self, _msg: String) -> None:
        self._followup_until = 0.0

    def _on_robot_state(self, msg: RobotStateMsg) -> None:
        """로봇 상태 메시지를 받아 최신값으로 보관한다."""
        self._robot_state = msg_to_robot_state(msg)

    def _on_user_text(self, msg: String) -> None:
        """발화를 받아 VicaIntent 를 만들어 발행한다."""
        text = msg.data.strip()
        if not text:
            return
        self._reload_destinations_if_changed()

        if self._history.begin_turn(time.time()):
            self.get_logger().info("대화가 끊겨 이전 맥락을 비웠다")

        keyword = detect_emergency(text)
        if keyword:
            intent = VicaIntent(
                intent="unknown",
                reply="",
                need_confirm=False,
                safety_flag="emergency",
            )
            self.get_logger().warn(f"[긴급] '{keyword}' 감지 -> safety_flag=emergency")
        else:
            thinking = not is_instant_utterance(text)
            if thinking:
                self._thinking_pub.publish(Bool(data=True))

            try:
                intent = parse_intent(
                    text,
                    self._destinations,
                    history=self._history.messages,
                    robot_state=self._robot_state,
                )
            finally:
                if thinking:
                    self._thinking_pub.publish(Bool(data=False))

        if (time.time() < self._followup_until
                and intent.intent in ("unknown", "clarify")
                and not intent.need_confirm
                and intent.reply not in SHORTCUT_REPLIES
                and intent.safety_flag != "emergency"):
            self.get_logger().info(
                f"재청취 기각(무의미): '{text}' intent={intent.intent}")
            return

        if should_forward_intent(intent):
            self._intent_pub.publish(intent_to_msg(intent))
        else:
            self.get_logger().info("resume 확언 대기 — 발행 보류 (질문만 나감)")

        request = request_for_intent(intent)
        if request:
            self._tts_pub.publish(String(data=request))

        if (intent.need_confirm or intent.intent == "clarify"
                or expects_answer(intent.reply)):
            self._listen_pub.publish(Bool(data=True))
        self.get_logger().info(
            f"입력='{text}' -> intent={intent.intent} "
            f"matched={intent.matched_destination_id} safety={intent.safety_flag}"
        )

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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

"""VICA TTS ROS2 노드.

구독: /vica/tts_request (std_msgs/String, "{priority}:{text}")
        Mission Manager 의 안내 멘트(안내 시작·도착·거부 사유 등)와 LLM 응답이
        모두 이 하나의 입구로 들어온다. 이 노드는 무엇을 말할지 판단하지 않고,
        들어온 순서와 우선순위대로 재생만 한다.
발행: /vica/tts_state   (std_msgs/Bool) - 재생 중 여부

/vica/tts_state 를 두는 이유:
    긴급어 상시 감시(ros_emergency_node)는 마이크를 계속 열어 두므로 스피커로 나간
    로봇 자기 목소리도 듣는다. 멘트에 "멈춰"/"정지" 가 없어도 목적지 이름 같은 데서
    걸릴 수 있어(예: "행정지원실" 안의 "정지"), 재생 중에는 감시를 쉬게 한다.

    재생 구간을 짧게 유지하려고 문장 단위로 끊어 재생하고, 문장 사이마다 감시를
    다시 연다. 사용자의 진짜 긴급 발화를 놓치는 창을 줄이기 위해서다.

재생은 별도 스레드에서 돈다. 구독 콜백에서 직접 재생하면 재생이 끝날 때까지
다음 요청을 받지 못해, 긴급 발화가 앞선 안내 뒤에 줄을 서게 된다.

실행:
    source /opt/ros/humble/setup.bash
    .venv/bin/python -m src.ros_tts_node
"""
from __future__ import annotations

import threading
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String

from .tts import VicaTTS
from .tts_queue import TtsQueue, parse_request
from .tts_text import split_sentences

# 재생이 끝난 뒤 감시를 다시 열기까지의 여유. 스피커 잔향과 마이크 입력 지연 때문에
# 0 으로 두면 방금 끝난 로봇 목소리를 그대로 다시 듣는다.
TAIL_SEC = 0.4

# 큐가 비었을 때 재생 스레드가 쉬는 간격.
IDLE_POLL_SEC = 0.05


class TtsNode(Node):
    def __init__(self) -> None:
        super().__init__("vica_tts_node")
        self.get_logger().info("TTS 모델 로드 중...")
        self._tts = VicaTTS()
        self._queue = TtsQueue()
        self._preempt = threading.Event()
        self._running = True

        self._state_pub = self.create_publisher(Bool, "/vica/tts_state", 10)
        self._publish_state(False)  # 시작 상태를 명시적으로 알린다

        self.create_subscription(String, "/vica/tts_request", self._on_request, 10)

        self._worker = threading.Thread(target=self._playback_loop, daemon=True)
        self._worker.start()

        self.get_logger().info(
            "VICA TTS node 시작 (구독: /vica/tts_request | 발행: /vica/tts_state)"
        )

    # -- 입력 ----------------------------------------------------------------

    def _on_request(self, msg: String) -> None:
        priority, text = parse_request(msg.data)
        self._enqueue(priority, text)

    def _enqueue(self, priority: str, text: str) -> None:
        if not text:
            return
        result = self._queue.push(priority, text, now=time.time())
        if not result.accepted:
            self.get_logger().debug(f"발화 무시({result.reason}): {text}")
            return
        if result.preempt:
            self._preempt.set()
            self._tts.stop()
            self.get_logger().warn(f"긴급 발화로 선점: {text}")
        else:
            self.get_logger().info(f"발화 대기[{priority}]: {text}")

    # -- 재생 ----------------------------------------------------------------

    def _publish_state(self, speaking: bool) -> None:
        msg = Bool()
        msg.data = speaking
        self._state_pub.publish(msg)

    def _playback_loop(self) -> None:
        while self._running:
            item = self._queue.pop()
            if item is None:
                time.sleep(IDLE_POLL_SEC)
                continue

            # 이전 선점 신호는 여기서 소비한다 — 새 발화까지 끊기면 안 된다.
            self._preempt.clear()
            self.get_logger().info(f"재생[{item.priority}]: {item.text}")
            self._speak(item.text)

    def _speak(self, text: str) -> None:
        """문장 단위로 끊어 재생하고, 재생 구간마다 감시 억제 신호를 켜고 끈다."""
        for chunk in split_sentences(text):
            if self._preempt.is_set():
                break
            self._publish_state(True)
            try:
                self._tts.speak(chunk)
            finally:
                # 재생이 실패해도 감시는 반드시 다시 열어야 한다.
                time.sleep(TAIL_SEC)
                self._publish_state(False)

    def shutdown(self) -> None:
        self._running = False
        self._tts.stop()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TtsNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

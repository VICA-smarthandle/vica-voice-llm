"""VICA TTS ROS2 노드 — 단일 큐 + 우선순위 재생 (통합 진행순서 ②).

구독:
    /vica/intent      (vica_interfaces/VicaIntent) — reply 를 RESPONSE 우선순위로
    /vica/tts_request (std_msgs/String)            — mission_manager 등의 멘트.
                       접두어(emergency:/narration:/response:)로 우선순위 지정,
                       없으면 내레이션. (src/tts_queue.py 참고)

동작: 재생 전용 워커 스레드가 큐에서 우선순위 순으로 꺼내 supertonic 으로 재생.
긴급 멘트가 들어오면 하위 우선순위 큐를 비우고, 재생 중인 오디오도 중단을
시도한다 (sounddevice.stop, best-effort).

실행:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    .venv/bin/python -m src.ros_tts_node
"""
from __future__ import annotations

import threading

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String
from vica_interfaces.msg import VicaIntent as VicaIntentMsg

from .tts import VicaTTS
from .tts_queue import TtsPriority, TtsQueue, parse_tts_request


class TtsNode(Node):
    def __init__(self) -> None:
        super().__init__("vica_tts_node")
        self.get_logger().info("TTS 모델 로드 중...")
        self._tts = VicaTTS()
        self._queue = TtsQueue()
        self._wake = threading.Event()
        self._shutdown = False

        self.create_subscription(VicaIntentMsg, "/vica/intent", self._on_intent, 10)
        self.create_subscription(String, "/vica/tts_request", self._on_tts_request, 10)

        self._worker = threading.Thread(target=self._playback_loop, daemon=True)
        self._worker.start()
        self.get_logger().info(
            "VICA TTS node 시작 (구독: /vica/intent, /vica/tts_request — 큐+우선순위)"
        )

    # -- 입력 -----------------------------------------------------------------

    def _on_intent(self, msg: VicaIntentMsg) -> None:
        """LLM reply 는 항상 RESPONSE 우선순위."""
        self._enqueue(msg.reply, TtsPriority.RESPONSE)

    def _on_tts_request(self, msg: String) -> None:
        priority, text = parse_tts_request(msg.data)
        if priority == TtsPriority.EMERGENCY:
            self._interrupt_playback()
        self._enqueue(text, priority)

    def _enqueue(self, text: str, priority: TtsPriority) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._queue.put(text, priority)
        self.get_logger().info(f"큐 추가 [{priority.name}] {text} (대기 {len(self._queue)}건)")
        self._wake.set()

    # -- 재생 -----------------------------------------------------------------

    def _playback_loop(self) -> None:
        while not self._shutdown:
            item = self._queue.pop()
            if item is None:
                self._wake.wait(timeout=0.2)
                self._wake.clear()
                continue
            priority, text = item
            self.get_logger().info(f"재생 [{priority.name}] {text}")
            try:
                self._tts.speak(text)
            except Exception as exc:  # 재생 실패가 큐 전체를 죽이면 안 된다
                self.get_logger().error(f"TTS 재생 실패: {exc}")

    def _interrupt_playback(self) -> None:
        """긴급 멘트를 위해 재생 중인 오디오를 중단한다 (best-effort)."""
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass

    def destroy_node(self) -> None:
        self._shutdown = True
        self._wake.set()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TtsNode()
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

"""VICA STT ROS2 노드 (마이크 -> /vica/user_text).

동작 두 갈래:
1. push-to-talk — 녹음(엔터 -> 말하기 -> 엔터) -> whisper -> /vica/user_text.
2. 접근 질문 자동 답변 창 — /vica/tts_done 으로 "안내가 필요하신가요?" 의
   재생 종료를 들으면 엔터 없이 4초 녹음해 같은 경로로 발행한다. 로봇이
   다가가 묻는 상대는 터미널 앞에 없으므로 엔터를 눌러 줄 사람이 없다
   (2026-08-28 실기에서 무응답 "실례했습니다"로 끝난 원인). 재생이 끝난 뒤에만
   마이크를 열므로 로봇 자기 목소리는 녹음되지 않는다.

구독이 생겼으므로 spin 은 데몬 스레드에서 돌리고, 본 스레드는 push-to-talk
입력 루프를 지킨다. 마이크는 Lock 으로 한 번에 한 갈래만 쓴다.

실행:
    source /opt/ros/humble/setup.bash
    .venv/bin/python -m src.ros_stt_node
"""
from __future__ import annotations

import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .auto_answer import RECORD_SECONDS, should_auto_listen
from .replies import RETRY_PROMPT
from .stt import VicaSTT
from .tts_queue import RESPONSE, build_request


class SttNode(Node):
    def __init__(self) -> None:
        super().__init__("vica_stt_node")
        self._pub = self.create_publisher(String, "/vica/user_text", 10)
        self._tts_pub = self.create_publisher(String, "/vica/tts_request", 10)

    def publish_text(self, text: str) -> None:
        msg = String()
        msg.data = text
        self._pub.publish(msg)
        self.get_logger().info(f"발행 /vica/user_text: '{text}'")

    def ask_retry(self) -> None:
        """인식 결과가 없을 때 다시 말해 달라고 알린다.

        침묵으로 두면 눈으로 확인할 수 없는 사용자는 로봇이 못 들은 것인지,
        생각 중인 것인지 구분하지 못해 다시 말할 시점을 잡을 수 없다.
        """
        self._tts_pub.publish(String(data=build_request(RESPONSE, RETRY_PROMPT)))
        self.get_logger().warn("인식 결과 없음 -> 재발화 안내")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SttNode()
    node.get_logger().info("STT 모델 로드 중...")
    stt = VicaSTT()

    # 마이크는 한 갈래만. non-blocking 인 이유: push-to-talk 이 두 번째 엔터를
    # 기다리며 잡고 있을 수 있는데, 자동 창이 거기 줄을 서면 답변 창 8초가
    # 그냥 지나간다 — 그때는 녹음 중인 수동 갈래가 답을 담는다.
    mic_lock = threading.Lock()

    def on_tts_done(msg: String) -> None:
        if not should_auto_listen(msg.data):
            return
        if not mic_lock.acquire(blocking=False):
            node.get_logger().info("수동 녹음 중 — 자동 답변 창 생략")
            return
        try:
            node.get_logger().info(
                f"질문 재생 완료 -> 엔터 없이 {RECORD_SECONDS:.0f}초 자동 녹음"
            )
            audio = stt.record_seconds(RECORD_SECONDS)
            text = stt.transcribe(audio).strip() if audio.size else ""
        finally:
            mic_lock.release()
        if text:
            node.publish_text(text)
        else:
            # 재발화 안내(ask_retry)는 일부러 생략한다 — 안내 재생에 응답 대기
            # 8초를 쓰면 mission 의 무응답 멘트와 겹쳐 두 소리가 충돌한다.
            node.get_logger().warn("자동 답변 창에서 인식 결과 없음")

    node.create_subscription(String, "/vica/tts_done", on_tts_done, 10)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    node.get_logger().info(
        "VICA STT node 시작 (엔터로 녹음 / 접근 질문 뒤엔 자동 녹음, 발행: /vica/user_text)"
    )

    try:
        while rclpy.ok():
            try:
                input("녹음하려면 엔터 (종료 Ctrl+C) > ")
            except EOFError:
                break
            with mic_lock:
                text = stt.listen().strip()
            if not text:
                node.ask_retry()
                continue
            node.publish_text(text)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

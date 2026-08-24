"""VICA 웨이크워드 ROS2 노드 (/vica/wakeword_node) — P1-b.

ros_emergency_node(whisper 상시)와 push-to-talk STT 를 함께 대체하는 마이크 앞단.

발행: /vica/emergency (vica_interfaces/EmergencyEvent)  ← 긴급, LLM 우회 (기존 계약)
      /vica/user_text (std_msgs/String)                 ← 호출 후 발화 (기존 계약)
      /vica/tts_stop  (std_msgs/Empty)                  ← barge-in: 재생 즉시 중단 요청
구독: /vica/tts_state (std_msgs/Bool)                   ← TTS 재생 경계 (뮤트 또는 AEC 모드)
      /vica/listen_request (std_msgs/Bool)              ← 질문 후 재청취 예약 (true=예약)

keyword 는 whisper 전사에서 정확 매칭으로 추출되므로 항상
HARD_EMERGENCY_KEYWORDS 정본 안의 값이다 — 브리지·래치 체인 변경 없음.

안전 원칙: 감지·발행까지만 한다. /cmd_vel*, Nav2 goal, CAN 없음.

실행:
    source /opt/ros/humble/setup.bash && source ../vica_ros2_ws/install/setup.bash
    .venv/bin/python -m src.ros_wakeword_node
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Empty, String
from vica_interfaces.msg import EmergencyEvent as EmergencyEventMsg

from . import audio_cue
from .replies import WAKE_GREETING
from .ros_convert import emergency_to_msg
from .schema import EmergencyEvent
from .tts_queue import RESPONSE, build_request
from .wakeword_monitor import WakewordMonitor

# 첫 호출 인사("네?")를 미리 합성해 둔 파일. 있으면 TTS 큐를 거치지 않고 즉시 난다.
# 호출 응답은 빠를수록 좋다 — 사용자가 "들었나?" 하고 기다리는 순간이다.
# 만드는 법: scripts/make_cue_wavs.py (TTS 가 있는 기기에서 한 번 실행)
GREETING_WAV = Path(__file__).resolve().parent.parent / "assets" / "wake_greeting.wav"


class WakewordNode(Node):
    def __init__(self) -> None:
        super().__init__("vica_wakeword_node")
        self._pub_emergency = self.create_publisher(EmergencyEventMsg, "/vica/emergency", 10)
        self._pub_text = self.create_publisher(String, "/vica/user_text", 10)
        self._pub_wake = self.create_publisher(String, "/vica/wake", 10)  # 계측·UI 앵커
        self._tts_pub = self.create_publisher(String, "/vica/tts_request", 10)
        self._stop_pub = self.create_publisher(Empty, "/vica/tts_stop", 10)
        self._tts_speaking = False
        self.create_subscription(Bool, "/vica/tts_state", self._on_tts_state, 10)
        # 질문을 말한 노드(LLM node, Mission Manager)가 true 를 보내면, 그 질문
        # TTS 가 끝나는 순간 웨이크워드 없이 청취 창을 연다. "안내를 취소할까요?"
        # 에 "아니요" 한마디를 하려고 "비카야"를 다시 부를 필요가 없게 한다.
        self.create_subscription(Bool, "/vica/listen_request", self._on_listen_request, 10)

        # 호출에는 항상 "네?"로 답한다. 짧은 신호음만으로는 언제 말해야
        # 하는지 알 수 없다는 로봇팀 실사용 피드백(2026-08-20)으로, 첫 호출만
        # 인사하던 GreetingState 를 없앴다. 효과음은 참고용일 뿐이다.
        self._greeting_wav = self._load_greeting_wav()

        # TTS 재생 중 마이크 처리. 기본은 뮤트(자기 목소리 오탐 방지).
        # AEC 배선 환경(TTS 가 reSpeaker 재생 경로로 나가는 기기)에서는
        # VICA_TTS_MUTE=off 로 두면 재생 중에도 감시가 계속된다 — 로봇이
        # 말하는 도중의 "멈춰"가 들린다. 문제가 보이면 값 하나로 원복한다.
        self._mute_during_tts = os.environ.get(
            "VICA_TTS_MUTE", "on").strip().lower() not in ("off", "0", "false")

        self._monitor = WakewordMonitor(
            on_emergency=self._on_emergency,
            on_user_text=self._on_user_text,
            on_wake=self._on_wake,
            on_barge_in=self._on_barge_in,
        )
        # 마이크 감시 루프는 blocking 이라 별도 스레드 (ros_emergency_node 와 동일 패턴)
        self._thread = threading.Thread(target=self._monitor.run, daemon=True)
        self._thread.start()
        mode = "뮤트" if self._mute_during_tts else "감시 유지(AEC)"
        self.get_logger().info(
            "VICA 웨이크워드 감시 시작 (발행: /vica/emergency, /vica/user_text | "
            f"TTS 중 {mode})")

    def _on_emergency(self, event: EmergencyEvent) -> None:
        # 긴급이 확정되면 로봇부터 입을 다문다 — 정지 안내(긴급 발화)는
        # 큐에 남아 이어서 나간다 (tts_stop 은 긴급 발화를 버리지 않는다).
        self._stop_pub.publish(Empty())
        self._pub_emergency.publish(emergency_to_msg(event))
        self.get_logger().warn(
            f"🚨 긴급 '{event.keyword}' 확정 -> /vica/emergency (인식: {event.source_text!r})")

    def _on_user_text(self, text: str) -> None:
        msg = String()
        msg.data = text
        self._pub_text.publish(msg)
        self.get_logger().info(f"🗣️ 호출 발화 -> /vica/user_text: {text!r}")
        # 수음 품질을 항상 남긴다 — "잘 못 알아듣는다"의 원인 조사 재료
        stats = self._monitor.last_listen_stats
        if stats:
            self.get_logger().info(
                f"수음 품질: rms {stats['rms']:.4f} · peak {stats['peak']:.2f}"
                f" · clip {stats['clip_ratio']:.1%}")

    def _on_wake(self) -> None:
        # 로봇이 말하는 중에 부른 것이면 하던 말을 끊는다 (barge-in).
        # 뮤트 모드에서는 재생 중 호출이 애초에 들리지 않으므로 영향 없다.
        if self._tts_speaking:
            self._stop_pub.publish(Empty())
            self.get_logger().info("호출 barge-in — 재생 중단 요청")
        self._greet()
        msg = String()
        msg.data = "wake"
        self._pub_wake.publish(msg)
        self.get_logger().info("🙋 비카야 호출 — 청취 창 열림")

    def _on_barge_in(self) -> None:
        """질문 재생 중 사용자가 답을 시작했다 — 하던 말을 끊고 듣는다."""
        self._stop_pub.publish(Empty())
        self.get_logger().info("답변 barge-in — 질문 재생 중단, 청취 시작")

    def _greet(self) -> None:
        """호출 응답 "네?". 미리 만든 음성이 있으면 즉시, 없으면 TTS 로."""
        if self._greeting_wav is not None:
            audio_cue.play(*self._greeting_wav)
            return
        # 폴백: 큐를 거치므로 조금 늦다. 파일을 만들어 두면 즉시 난다.
        self._tts_pub.publish(String(data=build_request(RESPONSE, WAKE_GREETING)))

    def _load_greeting_wav(self):
        """인사 음성 파일을 읽어 둔다. 없으면 None (TTS 폴백)."""
        if not GREETING_WAV.exists():
            self.get_logger().warn(
                f"인사 음성이 없어 TTS 로 대체한다 (조금 늦다): {GREETING_WAV}\n"
                "  만들기: .venv/bin/python scripts/make_cue_wavs.py")
            return None
        try:
            import soundfile as sf

            data, rate = sf.read(str(GREETING_WAV), dtype="float32")
            return data, rate
        except Exception as exc:
            self.get_logger().warn(f"인사 음성을 읽지 못해 TTS 로 대체한다: {exc}")
            return None

    def _on_listen_request(self, msg: Bool) -> None:
        if msg.data:
            self._monitor.arm_followup()
            self.get_logger().info("질문 예약 — TTS 종료 후 재청취 창을 연다")
        else:
            self._monitor.disarm_followup()

    def _on_tts_state(self, msg: Bool) -> None:
        self._tts_speaking = bool(msg.data)
        if self._mute_during_tts:
            self._monitor.set_muted(bool(msg.data))
        else:
            self._monitor.set_speaking(bool(msg.data))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WakewordNode()
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

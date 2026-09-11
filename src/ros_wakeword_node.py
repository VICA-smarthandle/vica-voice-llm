"""VICA 웨이크워드 ROS2 노드 (/vica/wakeword_node) — P1-b."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Empty, Float32, String
from vica_interfaces.msg import EmergencyEvent as EmergencyEventMsg

from .destination_loader import build_place_hint, load_destinations
from .dsp_state import (agc_desired_from_env, apply_agc_desired_level,
                        apply_echo_tuning, echo_tuning_from_env)
from .replies import WAKE_GREETING
from .stt_guard import strip_robot_echo
from .ros_convert import emergency_to_msg
from .schema import EmergencyEvent
from .tts_queue import RESPONSE, build_request
from .wakeword_monitor import WakewordMonitor


ROBOT_ECHO_TTL_SEC = 12.0


class WakewordNode(Node):
    def __init__(self) -> None:
        super().__init__("vica_wakeword_node")
        self._pub_emergency = self.create_publisher(EmergencyEventMsg, "/vica/emergency", 10)
        self._pub_text = self.create_publisher(String, "/vica/user_text", 10)
        self._robot_recent: list = []
        self.create_subscription(
            String, "/vica/tts_done", self._on_tts_done_text, 10)
        self._pub_wake = self.create_publisher(String, "/vica/wake", 10)
        self._pub_wake_doa = self.create_publisher(Float32, "/vica/wake_doa", 10)
        self._pub_listen_state = self.create_publisher(String, "/vica/listen_state", 10)
        self._tts_pub = self.create_publisher(String, "/vica/tts_request", 10)
        self._stop_pub = self.create_publisher(Empty, "/vica/tts_stop", 10)
        self._tts_speaking = False
        self.create_subscription(Bool, "/vica/tts_state", self._on_tts_state, 10)
        self.create_subscription(Bool, "/vica/listen_request", self._on_listen_request, 10)

        self._mute_during_tts = os.environ.get(
            "VICA_TTS_MUTE", "off").strip().lower() not in ("off", "0", "false")
        self._voice_barge_in = os.environ.get(
            "VICA_BARGE_IN_VOICE", "on").strip().lower() not in ("off", "0", "false")
        self._doa_gate = os.environ.get(
            "VICA_BARGE_DOA_GATE", "1").strip() not in ("0", "false", "off")
        doa_center = os.environ.get("VICA_USER_DOA_CENTER", "").strip()
        self._user_doa_center = float(doa_center) if doa_center else None
        self._user_doa_width = float(
            os.environ.get("VICA_USER_DOA_WIDTH", "45") or 45)

        self.declare_parameter(
            "destinations_yaml",
            str(Path.home() / "vica_data" / "destinations" / "vica_map_0630"
                / "destinations.yaml"),
        )
        hint_path = os.environ.get(
            "VICA_DESTINATIONS_YAML",
            str(self.get_parameter("destinations_yaml").value))
        try:
            listen_hint = build_place_hint(load_destinations(hint_path))
            self.get_logger().info(f"장소 귀띔 준비 ({hint_path}): '{listen_hint}'")
        except Exception as exc:
            listen_hint = None
            self.get_logger().warning(f"장소 귀띔 생략 (목적지 로드 실패): {exc}")

        self._monitor = WakewordMonitor(
            listen_hint=listen_hint,
            on_emergency=self._on_emergency,
            on_user_text=self._on_user_text,
            on_wake=self._on_wake,
            on_barge_in=self._on_barge_in,
            on_reject=self._on_reject,
            on_listen_empty=self._on_listen_empty,
            on_listen_state=self._on_listen_state,
            on_wake_doa=self._publish_wake_doa,
            voice_barge_in=self._voice_barge_in,
            user_doa_center=self._user_doa_center,
            doa_gate=self._doa_gate,
            user_doa_width=self._user_doa_width,
        )
        desired = agc_desired_from_env(
            os.environ.get("VICA_MIC_AGC_DESIRED", "0.010"))
        if desired is not None:
            if apply_agc_desired_level(desired):
                self.get_logger().info(f"AGC 목표 레벨 설정: {desired}")
            else:
                self.get_logger().warning(
                    f"AGC 목표 레벨 설정 실패({desired}) — 공장 기본으로 감시 계속")

        etail, nlatten = echo_tuning_from_env(
            os.environ.get("VICA_MIC_GAMMA_ETAIL", ""),
            os.environ.get("VICA_MIC_NLATTEN", ""))
        if etail is not None or nlatten is not None:
            wrote = apply_echo_tuning(etail, nlatten)
            if wrote:
                self.get_logger().info(
                    "에코 억제 설정: "
                    + " · ".join(f"{k}={v}" for k, v in wrote.items()))
            else:
                self.get_logger().warning(
                    "에코 억제 설정 실패 — 공장 기본(1.0/0)으로 감시 계속")

        self._thread = threading.Thread(target=self._monitor.run, daemon=True)
        self._thread.start()
        mode = "뮤트" if self._mute_during_tts else "감시 유지(AEC)"
        barge = "켜짐" if self._voice_barge_in else "꺼짐"
        gate = (f"켜짐 center={self._user_doa_center}±{self._user_doa_width}"
                if self._doa_gate else "꺼짐")
        self.get_logger().info(
            "VICA 웨이크워드 감시 시작 (발행: /vica/emergency, /vica/user_text | "
            f"TTS 중 {mode} | 음성 barge-in {barge} | DOA 관문 {gate})")

    def _on_emergency(self, event: EmergencyEvent) -> None:
        self._stop_pub.publish(Empty())
        self._pub_emergency.publish(emergency_to_msg(event))
        self.get_logger().warn(
            f"🚨 긴급 '{event.keyword}' 확정 -> /vica/emergency (인식: {event.source_text!r})")

    def _on_tts_done_text(self, msg: String) -> None:
        now = time.time()
        self._robot_recent = [
            (t, s) for t, s in self._robot_recent if now - t < ROBOT_ECHO_TTL_SEC]
        self._robot_recent.append((now, msg.data))

    def _on_user_text(self, text: str) -> None:
        now = time.time()
        recent = [s for t, s in self._robot_recent if now - t < ROBOT_ECHO_TTL_SEC]
        cleaned = strip_robot_echo(text, recent)
        if cleaned != text.strip():
            if not cleaned:
                self._on_listen_state(f"empty:echo {text[:40]!r}")
                return
            self.get_logger().info(f"에코 제거: {text!r} -> {cleaned!r}")
            text = cleaned
        msg = String()
        msg.data = text
        self._pub_text.publish(msg)
        self.get_logger().info(f"🗣️ 호출 발화 -> /vica/user_text: {text!r}")
        stats = self._monitor.last_listen_stats
        if stats:
            self.get_logger().info(
                f"수음 품질: rms {stats['rms']:.4f} · peak {stats['peak']:.2f}"
                f" · clip {stats['clip_ratio']:.1%}")
        timing = self._monitor.last_listen_timing
        if timing:
            self.get_logger().info(
                "계측: 대기 {wait:.2f}s · 발화 {speech:.2f}s · "
                "말끝판정 {tail:.2f}s · STT {stt:.2f}s".format(**timing))

    def _on_listen_empty(self) -> None:
        self.get_logger().info("청취 창 빈손 종료 (발화 없음/빈 전사)")

    def _on_listen_state(self, state: str) -> None:
        self._pub_listen_state.publish(String(data=state))
        if state.startswith("wake-rescue"):
            self.get_logger().info(f"🙋 창 안 호출 (소리로 구제): {state}")
            self._pub_wake.publish(String(data="wake"))
        if state.startswith("barge-miss"):
            self.get_logger().info(f"🔇 {state}")
        if state.startswith("answer-beats-rescue"):
            self.get_logger().info(f"🛡️ {state}")
        if ":" in state:
            self.get_logger().warn(f"청취 기각: {state}")

    def _on_wake(self) -> None:
        self._tts_pub.publish(String(data="control:stop"))
        self._greet()
        msg = String()
        msg.data = "wake"
        self._pub_wake.publish(msg)
        self.get_logger().info("🙋 비카야 호출 — 청취 창 열림")

    def _publish_wake_doa(self, doa: float) -> None:
        """호출이 온 방향. 못 읽으면 monitor 가 아예 부르지 않는다."""
        self._pub_wake_doa.publish(Float32(data=float(doa)))
        self.get_logger().info(f"🧭 호출 방향 {doa:.0f}°")

    def _on_barge_in(self) -> None:
        """질문 재생 중 사용자가 답을 시작했다 — 하던 말을 끊고 듣는다."""
        self._stop_pub.publish(Empty())
        self.get_logger().info("답변 barge-in — 질문 재생 중단, 청취 시작")

    def _on_reject(self, text: str) -> None:
        """긴급 관문은 발동했으나 STT 정확 매칭에서 기각된 사건."""
        self.get_logger().warn(f"긴급 관문 발동 → STT 기각 (전사: {text!r})")

    def _greet(self) -> None:
        """호출 응답 "네?". 미리 만든 음성이 있으면 즉시, 없으면 TTS 로."""
        self._tts_pub.publish(String(data=build_request(RESPONSE, WAKE_GREETING)))

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

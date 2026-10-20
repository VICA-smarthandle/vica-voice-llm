"""VICA 웨이크워드 ROS2 노드 (/vica/wakeword_node) — P1-b."""
from __future__ import annotations

import array
import os
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import (Bool, Empty, Float32, MultiArrayDimension, String,
                          UInt8MultiArray)
from vica_interfaces.msg import EmergencyEvent as EmergencyEventMsg
from vica_interfaces.msg import RobotState as RobotStateMsg

from .destination_loader import build_place_hint, load_destinations
from .dsp_state import (agc_desired_from_env, agc_max_gain_from_env, apply_agc_desired_level,
                        apply_agc_max_gain, apply_echo_tuning, echo_tuning_from_env)
from .realtime_intent import float32_to_pcm16
from .replies import WAKE_GREETING
from .stt_guard import strip_robot_echo
from .stt_qwen import QwenUnavailable, ear_stt_engines
from .ros_convert import emergency_to_msg
from .schema import EmergencyEvent
from .tts_queue import RESPONSE, build_request
from .wakeword_monitor import (
    LISTEN_START_SEC, WakewordMonitor, describe_emergency_check, describe_emergency_near,
    ear_stt_model_names, save_emergency_clip)


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
        self.create_subscription(String, "/vica/tts_now", self._on_tts_now, 10)
        self.create_subscription(Bool, "/vica/listen_request", self._on_listen_request, 10)

        self._wake_by_mission = os.environ.get(
            "VICA_WAKE_BY_MISSION", "on").strip().lower() not in ("off", "0", "false")
        self.create_subscription(String, "/vica/wake_reply", self._on_wake_reply, 10)
        self._wake_off = False
        self.create_subscription(RobotStateMsg, "/vica/robot_state", self._on_robot_state, 10)

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
        self._wake_doa_turn = os.environ.get(
            "VICA_WAKE_DOA_TURN", "off").strip().lower() in ("1", "on", "true")

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

        self._intent_input = os.environ.get("VICA_INTENT_INPUT", "text").strip().lower()
        clip_dir = os.environ.get("VICA_ESTOP_CLIP_DIR", "~/vica_data/estop_clips").strip()
        self._estop_clip_dir = None if clip_dir.lower() in ("", "off", "0", "false") else clip_dir
        self._estop_clip_warned = False
        self._pub_audio = self.create_publisher(UInt8MultiArray, "/vica/user_audio", 10)

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
            on_user_audio=self._on_user_audio if self._intent_input == "audio" else None,
            voice_barge_in=self._voice_barge_in,
            user_doa_center=self._user_doa_center,
            doa_gate=self._doa_gate,
            user_doa_width=self._user_doa_width,
            wake_gate=self._wake_by_mission,
            on_wake_score=self._on_wake_score,
            on_emergency_score=self._on_emergency_score,
        )
        self.get_logger().info(
            f"호출 문턱 {self._monitor.gate_a.threshold:.2f} · "
            f"말 시작 기다림 {LISTEN_START_SEC:.1f}s · "
            f"수음 최소 {self._monitor.listen_min_rms:.4f} · "
            f"짧은 답 최소 {self._monitor.short_answer_min_rms:.4f} · "
            + (f"접근 대답 최소 {self._monitor.approach_min_rms:.4f} · "
               if self._monitor.approach_min_rms is not None else "")
            + f"긴급 소리 저장 {self._estop_clip_dir or '꺼짐'}")
        single = self._monitor.gate_b.single
        listen_stt, verify_stt = ear_stt_model_names()
        try:
            listen_engine, verify_engine = ear_stt_engines()
        except QwenUnavailable as exc:
            listen_engine = verify_engine = f"설정 오류({exc})"
        qwen_name = Path(os.environ.get("VICA_QWEN_MODEL") or "").stem or "모델 경로 없음"

        def _stt_label(engine: str, whisper_name: str) -> str:
            if engine == "qwen":
                return f"qwen({qwen_name})"
            return whisper_name if engine == "whisper" else engine

        self.get_logger().info(
            f"긴급 관문 {self._monitor.gate_b.threshold:.2f} 2연속"
            + (f" 또는 한 조각 {single:.2f}" if single is not None else " (한 조각 규칙 꺼짐)")
            + f" · 긴급 확인 STT {_stt_label(verify_engine, verify_stt)}"
            + f" · 대화 STT {_stt_label(listen_engine, listen_stt)}")
        desired = agc_desired_from_env(
            os.environ.get("VICA_MIC_AGC_DESIRED", "0.010"))
        if desired is not None:
            if apply_agc_desired_level(desired):
                self.get_logger().info(f"AGC 목표 레벨 설정: {desired}")
            else:
                self.get_logger().warning(
                    f"AGC 목표 레벨 설정 실패({desired}) — 공장 기본으로 감시 계속")

        max_gain = agc_max_gain_from_env(os.environ.get("VICA_MIC_AGC_MAX_GAIN", ""))
        if max_gain is not None:
            if apply_agc_max_gain(max_gain):
                self.get_logger().info(f"AGC 최대 증폭 한도 설정: ×{max_gain:g}")
            else:
                self.get_logger().warning(
                    f"AGC 최대 증폭 한도 설정 실패(×{max_gain:g}) — 공장값(×31.6)으로 감시 계속")

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

        self._monitor_error: "BaseException | None" = None
        self._thread = threading.Thread(target=self._run_monitor, daemon=True)
        self._thread.start()
        self.create_timer(1.0, self._check_monitor)
        mode = "뮤트" if self._mute_during_tts else "감시 유지(AEC)"
        barge = "켜짐" if self._voice_barge_in else "꺼짐"
        gate = (f"켜짐 center={self._user_doa_center}±{self._user_doa_width}"
                if self._doa_gate else "꺼짐")
        self.get_logger().info(
            "VICA 웨이크워드 감시 시작 (발행: /vica/emergency, /vica/user_text | "
            f"TTS 중 {mode} | 음성 barge-in {barge} | DOA 관문 {gate} | "
            f"호출 방향 회전 {'켜짐' if self._wake_doa_turn else '꺼짐'} | "
            f"호출 응답 {'미션 판정' if self._wake_by_mission else '이 노드(옛 판)'})")

    def _on_emergency(self, event: EmergencyEvent) -> None:
        self._stop_pub.publish(Empty())
        self._pub_emergency.publish(emergency_to_msg(event))
        self.get_logger().warn(
            f"🚨 긴급 '{event.keyword}' 확정 -> /vica/emergency (인식: {event.source_text!r})"
            + describe_emergency_check(self._monitor.last_emergency_check))
        self._save_estop_clip(self._monitor.last_emergency_audio, self._monitor.last_emergency_check)

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
            self._pub_wake.publish(
                String(data="rescue" if self._wake_by_mission else "wake"))
        if state.startswith("barge-miss"):
            self.get_logger().info(f"🔇 {state}")
        if state.startswith("answer-beats-rescue"):
            self.get_logger().info(f"🛡️ {state}")
        if ":" in state:
            self.get_logger().warn(f"청취 기각: {state}")

    def _on_wake(self) -> None:
        if self._wake_by_mission:
            seq = self._monitor.wake_seq
            self._pub_wake.publish(String(data=f"wake:{seq}"))
            self.get_logger().info(f"🙋 비카야 호출 #{seq} — 청취 창 열림, 미션 판정 대기")
            return
        self._tts_pub.publish(String(data="control:stop"))
        self._greet()
        msg = String()
        msg.data = "wake"
        self._pub_wake.publish(msg)
        self.get_logger().info("🙋 비카야 호출 — 청취 창 열림")

    def _on_robot_state(self, msg: RobotStateMsg) -> None:
        """접근 질문('안내를 받으시겠어요?')의 답을 기다리는 동안 '비카야'를 호출로 듣지 않는다(2026-10-09"""
        self._monitor.set_wake_suppressed(msg.dialog_state == "awaiting_user")
        off = msg.dialog_state == "awaiting_user"
        if off != self._wake_off:
            self._wake_off = off
            self.get_logger().info("🔕 접근 질문 중 — '비카야' 끔" if off else "🔔 '비카야' 다시 켬")

    def _on_wake_reply(self, msg: String) -> None:
        """미션의 "비카야" 판정("listen:N"/"ignore:N") — 쥔 호출 창의 말을 넘길지 버릴지."""
        verdict, _, seq_text = (msg.data or "").strip().partition(":")
        if not self._wake_by_mission or verdict not in ("listen", "ignore"):
            return
        seq = int(seq_text) if seq_text.isdigit() else None
        self._monitor.set_wake_verdict(verdict, seq)
        if verdict == "ignore":
            self.get_logger().info(f"🙊 미션 판정 #{seq_text or '-'}: 대답 안 함 — 호출 창의 말을 버린다")

    def _publish_wake_doa(self, doa: float) -> None:
        """호출이 온 방향. 못 읽으면 monitor 가 아예 부르지 않는다."""
        if not self._wake_doa_turn:
            self.get_logger().info(f"🧭 호출 방향 {doa:.0f}° (방향 회전 꺼짐 — 미션에 안 보냄)")
            return
        self._pub_wake_doa.publish(Float32(data=float(doa)))
        self.get_logger().info(f"🧭 호출 방향 {doa:.0f}°")

    def _on_user_audio(self, audio) -> None:
        pcm = float32_to_pcm16(audio)
        msg = UInt8MultiArray()
        msg.layout.dim.append(MultiArrayDimension(label="pcm16_mono_16000", size=len(pcm) // 2, stride=len(pcm)))
        msg.data = array.array('B', pcm)
        self._pub_audio.publish(msg)
        self.get_logger().info(f"🎧 발화 소리 -> /vica/user_audio ({len(pcm)/2/16000:.2f}s)")

    def _on_barge_in(self) -> None:
        """질문 재생 중 사용자가 답을 시작했다 — 하던 말을 끊고 듣는다."""
        self._stop_pub.publish(Empty())
        self.get_logger().info("답변 barge-in — 질문 재생 중단, 청취 시작")

    def _on_reject(self, text: str) -> None:
        """긴급 관문은 발동했으나 STT 정확 매칭에서 기각된 사건."""
        self.get_logger().warn(f"긴급 관문 발동 → STT 기각 (전사: {text!r})"
                               + describe_emergency_check(self._monitor.last_emergency_check))
        self._save_estop_clip(self._monitor.last_emergency_audio, self._monitor.last_emergency_check)

    def _on_emergency_score(self, info: dict) -> None:
        """긴급 모델이 반응했지만 발동하지 않은 소리 — 점수 미달·한 조각만·쉬는 시간(2026-10-12)."""
        self.get_logger().info(describe_emergency_near(info, self._monitor.gate_b.threshold))
        self._save_estop_clip(info.get("audio"), info)

    def _save_estop_clip(self, audio, info) -> None:
        if self._estop_clip_dir is None or audio is None or not info or len(audio) == 0:
            return
        try:
            save_emergency_clip(self._estop_clip_dir, audio, info)
        except Exception as exc:
            if not self._estop_clip_warned:
                self._estop_clip_warned = True
                self.get_logger().warn(f"긴급 소리 저장 실패(이후 경고 생략): {exc}")

    def _greet(self) -> None:
        """호출 응답 "네?". 미리 만든 음성이 있으면 즉시, 없으면 TTS 로."""
        self._tts_pub.publish(String(data=build_request(RESPONSE, WAKE_GREETING)))

    def _on_listen_request(self, msg: Bool) -> None:
        if msg.data:
            self._monitor.arm_followup()
            self.get_logger().info("질문 예약 — TTS 종료 후 재청취 창을 연다")
        else:
            self._monitor.disarm_followup()

    def _on_tts_now(self, msg: String) -> None:
        self._monitor.note_robot_speech(msg.data)

    def _on_wake_score(self, kind: str, peak: float) -> None:
        label = {"fire": "확정", "near": "놓침", "self": "자기 말 무시"}.get(kind, kind)
        self.get_logger().info(
            f"🎯 호출 점수 {label}: {peak:.2f} (문턱 {self._monitor.gate_a.threshold:.2f})")

    def _on_tts_state(self, msg: Bool) -> None:
        self._tts_speaking = bool(msg.data)
        if self._mute_during_tts:
            self._monitor.set_muted(bool(msg.data))
        else:
            self._monitor.set_speaking(bool(msg.data))

    def _run_monitor(self) -> None:
        try:
            self._monitor.run()
        except BaseException as exc:  # noqa: BLE001 — 원인을 남긴다. 감시 타이머가 노드를 끝낸다
            self._monitor_error = exc
            raise

    def _check_monitor(self) -> None:
        if self._thread.is_alive():
            return
        self.get_logger().fatal(f"귀 감시가 멈췄습니다 — 노드를 끝냅니다: {self._monitor_error!r}")
        raise SystemExit(1)


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

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
import time
from pathlib import Path

from dotenv import load_dotenv

# .env 를 다른 어떤 모듈보다 먼저 로드한다 (2026-09-01). 예전엔 stt.py 의
# load_dotenv 에 기대고 있었는데, 그 모듈은 첫 인식 때에야 임포트돼서
# 기동 초반에 읽는 설정(AGC 목표·DOA 관문·청취 창 시간)이 전부 .env 를
# 못 보고 기본값으로 돌았다 — "0.007 로 내렸는데 로그는 0.01" 의 정체.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Empty, String
from vica_interfaces.msg import EmergencyEvent as EmergencyEventMsg

from .destination_loader import build_place_hint, load_destinations
from .dsp_state import agc_desired_from_env, apply_agc_desired_level
from .replies import WAKE_GREETING
from .stt_guard import strip_robot_echo
from .ros_convert import emergency_to_msg
from .schema import EmergencyEvent
from .tts_queue import RESPONSE, build_request
from .wakeword_monitor import WakewordMonitor

# 첫 호출 인사("네?")를 미리 합성해 둔 파일. 있으면 TTS 큐를 거치지 않고 즉시 난다.
# 호출 응답은 빠를수록 좋다 — 사용자가 "들었나?" 하고 기다리는 순간이다.
# 만드는 법: scripts/make_cue_wavs.py (TTS 가 있는 기기에서 한 번 실행)

# 에코 대조용으로 로봇 발화를 기억하는 시간. 발화 종료(tts_done) 직후의
# 재청취 창에서 잡히는 에코를 덮으면 된다 — 길게 두면 사용자가 로봇 말을
# 그대로 따라 하는 정당한 발화까지 지울 위험만 커진다.
ROBOT_ECHO_TTL_SEC = 12.0


class WakewordNode(Node):
    def __init__(self) -> None:
        super().__init__("vica_wakeword_node")
        self._pub_emergency = self.create_publisher(EmergencyEventMsg, "/vica/emergency", 10)
        self._pub_text = self.create_publisher(String, "/vica/user_text", 10)
        # 로봇이 방금 한 말(에코 대조용). AEC 잔여로 자기 발화가 전사에
        # 섞인다 — tts_done(완주·중단 불문 발행)의 문장을 잠시 기억해
        # 전사에서 걷어낸다 (strip_robot_echo, 2026-09-01).
        self._robot_recent: list = []
        self.create_subscription(
            String, "/vica/tts_done", self._on_tts_done_text, 10)
        self._pub_wake = self.create_publisher(String, "/vica/wake", 10)  # 계측·UI 앵커
        # 청취 상태 (open/speech/closed/empty) — 미션이 무응답 시계를 귀가
        # 바쁜 동안 멈추는 데 쓴다 (2026-08-30).
        self._pub_listen_state = self.create_publisher(String, "/vica/listen_state", 10)
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

        # TTS 재생 중 마이크 처리. 기본은 감시 유지(AEC) — 로봇의 표준 배선이
        # "reSpeaker USB 연결 + 스피커는 reSpeaker 출력에 물림"으로 확정되어
        # (2026-08-25 사용자 결정) 칩이 자기 목소리를 지워 준다. 로봇이 말하는
        # 도중의 "멈춰"가 들린다. 스피커가 reSpeaker 를 거치지 않는 예외 환경
        # 에서만 VICA_TTS_MUTE=on 으로 옛 뮤트 방식으로 되돌린다 (자기 소리
        # 오탐 방지). 판정 기준: robot-mount-checklist.md 의 자가 웨이크 0회.
        self._mute_during_tts = os.environ.get(
            "VICA_TTS_MUTE", "off").strip().lower() not in ("off", "0", "false")
        # 질문 재생 중 "아무 말" 끼어들기. 칩 발화 판정(자기 메아리 면역,
        # vad_probe 실측 0%)과 사용자 방향("비카야" 잠금 또는 장착 보정
        # 부채꼴)의 이중 증거가 있을 때만 발동하고, 방향을 모르면 스스로
        # 잠들므로 기본 켬이다. 문제가 보이면 VICA_BARGE_IN_VOICE=off.
        self._voice_barge_in = os.environ.get(
            "VICA_BARGE_IN_VOICE", "on").strip().lower() not in ("off", "0", "false")
        # 사용자 방향 부채꼴 (barge-in 전용 — 긴급어에는 방향 조건 없음).
        # 실기 장착 시 사용자(핸들) 방향을 실측해 넣는다 (tools/doa_probe).
        # 방향 관문 스위치 (기본 켬). 0 이면 방향 불문 — 칩 VAD 만 본다.
        self._doa_gate = os.environ.get(
            "VICA_BARGE_DOA_GATE", "1").strip() not in ("0", "false", "off")
        doa_center = os.environ.get("VICA_USER_DOA_CENTER", "").strip()
        self._user_doa_center = float(doa_center) if doa_center else None
        self._user_doa_width = float(
            os.environ.get("VICA_USER_DOA_WIDTH", "45") or 45)

        # 장소 이름 귀띔 — 자유 명령 창의 목적지 오전사('휴게실'→'조계실') 대책.
        # 목적지를 못 읽어도 감시는 시작해야 하므로 실패는 경고로만 남긴다.
        try:
            listen_hint = build_place_hint(load_destinations(os.environ.get(
                "VICA_DESTINATIONS_YAML",
                str(Path.home() / "vica_data" / "destinations" / "vica_map_0630"
                    / "destinations.yaml"))))
            self.get_logger().info(f"장소 귀띔 준비: '{listen_hint}'")
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
            voice_barge_in=self._voice_barge_in,
            user_doa_center=self._user_doa_center,
            doa_gate=self._doa_gate,
            user_doa_width=self._user_doa_width,
        )
        # AGC 목표 레벨 굳히기 — 칩은 전원 재투입마다 초기값(0.005)으로
        # 돌아간다. 반드시 마이크 스트림을 열기 전에 (스트림과 겹치면 제어
        # 전송 거부). D7 동결의 승인된 유일한 예외 (dsp_state 모듈 주석).
        desired = agc_desired_from_env(
            os.environ.get("VICA_MIC_AGC_DESIRED", "0.010"))
        if desired is not None:
            if apply_agc_desired_level(desired):
                self.get_logger().info(f"AGC 목표 레벨 설정: {desired}")
            else:
                self.get_logger().warning(
                    f"AGC 목표 레벨 설정 실패({desired}) — 공장 기본으로 감시 계속")

        # 마이크 감시 루프는 blocking 이라 별도 스레드 (ros_emergency_node 와 동일 패턴)
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
        # 긴급이 확정되면 로봇부터 입을 다문다 — 정지 안내(긴급 발화)는
        # 큐에 남아 이어서 나간다 (tts_stop 은 긴급 발화를 버리지 않는다).
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
        # 자기 목소리 에코를 먼저 걷어낸다 — 로봇의 질문이 사용자 답으로
        # 둔갑하면 LLM 이 자기 말에 자기가 대답하는 연쇄가 된다.
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
        # 수음 품질을 항상 남긴다 — "잘 못 알아듣는다"의 원인 조사 재료
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
        # "비카야" 창이 빈손으로 닫힘. 멘트는 하지 않는다 — 안내를 넣었다가
        # 사용자가 뺐다(2026-08-28 "쓸데없이 멘트 늘어나는 게 제일 싫다").
        # 로그는 남긴다: 이 흔적이 없어서 "감지가 안 되는" 조사가 어려웠다.
        self.get_logger().info("청취 창 빈손 종료 (발화 없음/빈 전사)")

    def _on_listen_state(self, state: str) -> None:
        self._pub_listen_state.publish(String(data=state))
        if state.startswith("wake-rescue"):
            # 창 안에서 소리로 건진 호출도 **진짜 호출과 똑같이** 미션에
            # 알린다. 종전에는 전사만 흘러가(user_text="비카야") 음성은
            # "네?"라고 답했는데 미션은 확인 상태를 그대로 들고 있었고,
            # 그 뒤 같은 목적지를 다시 말하자 "재제안=답" 규칙이 그것을
            # 승낙으로 읽어 **확인 없이 출발**했다(2026-09-02 실기 9회차).
            self.get_logger().info(f"🙋 창 안 호출 (소리로 구제): {state}")
            self._pub_wake.publish(String(data="wake"))
        if ":" in state:
            # 기각 사유(유령 문턱·환각·빈 전사) — 이전엔 followup 기각이
            # 무로그라 "대기해가 왜 죽었나"를 사후 진단할 수 없었다(2026-08-31).
            self.get_logger().warn(f"청취 기각: {state}")

    def _on_wake(self) -> None:
        # 호출 = 새 대화 (2026-09-01 사용자 결정). 하던 말을 끊고(barge-in)
        # 큐에 밀린 비긴급 발화도 함께 비운다 — 예전엔 "말하는 중일 때만"
        # 이라 문장 사이 침묵에 부르면 "네?" 뒤로 낡은 말이 이어졌다.
        # 청소는 tts_request 의 제어 메시지로 보낸다: 별도 토픽(tts_stop)은
        # 뒤이은 "네?"와 도착 순서가 뒤집혀 청소가 "네?"를 지웠다(실기
        # 3회 + 재현 1회). 같은 발신자·같은 토픽은 순서가 보장된다.
        self._tts_pub.publish(String(data="control:stop"))
        self._greet()
        msg = String()
        msg.data = "wake"
        self._pub_wake.publish(msg)
        self.get_logger().info("🙋 비카야 호출 — 청취 창 열림")

    def _on_barge_in(self) -> None:
        """질문 재생 중 사용자가 답을 시작했다 — 하던 말을 끊고 듣는다."""
        self._stop_pub.publish(Empty())
        self.get_logger().info("답변 barge-in — 질문 재생 중단, 청취 시작")

    def _on_reject(self, text: str) -> None:
        """긴급 관문은 발동했으나 STT 정확 매칭에서 기각된 사건.

        반드시 남긴다 — "멈춰"가 씹혔을 때 관문을 못 넘은 것인지, 넘고
        기각된 것인지 이 로그 없이는 사후에 가릴 수 없다 (2026-08-24 실기).
        """
        self.get_logger().warn(f"긴급 관문 발동 → STT 기각 (전사: {text!r})")

    def _greet(self) -> None:
        """호출 응답 "네?". 미리 만든 음성이 있으면 즉시, 없으면 TTS 로."""
        # "네?"는 무조건 TTS 큐로 보낸다 (2026-08-31 실기 — 직접 재생은
        # 구조적으로 불가였다: TTS 노드가 reSpeaker 출력을 상시 점유
        # (persistent stream, barge-in 끊기용)해서 다른 프로세스의 직접
        # 재생은 항상 busy 로 죽고, 그 실패가 삼켜져 "비카야에 대답 안 함"
        # 이 됐다. 장치 주인은 하나(단일 출구) — "네?"는 캐시 녹음이라
        # 큐에 닿는 즉시 0초 재생된다. 로봇이 말하는 중이면 위에서 이미
        # stop 을 보냈으므로(호출 barge-in) 곧바로 이어 나온다.
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

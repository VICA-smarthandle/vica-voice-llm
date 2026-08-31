"""VICA 청각 안내 노드 — 회전·도착을 소리와 말로 알린다.

구독: /vica/turn_guide   (vica_interfaces/TurnGuide) — 회전 감지
      /vica_goal_event   (std_msgs/String)          — 안내 시작·끝
      /navigate_to_pose/_action/feedback            — 잔여거리 (도착 근접 억제)
발행: /vica/tts_request  (std_msgs/String)          — 회전 안내 문구

신중 모드(2026-08-26): 신호를 받아도 1.5초 회전이 지속될 때만, 그리고 잔여
거리 5 m 초과일 때만 말한다 — 잔 보정 흔들림과 도착 정렬 회전을 걸러낸다.

회전 안내는 **음 + 말** 이다. 음이 주의를 끌고 말이 뜻을 전한다. 음만으로는 처음
쓰는 사용자가 무슨 뜻인지 모르고, 말만으로는 앞부분을 놓친다.

문구를 축약하지 않는 이유: 방향 안내는 사용자의 안전과 직결된다. 매번 같은
문장이어야 매번 같은 판단을 할 수 있다 (2026-08-05 사용자 결정).

우선순위를 response 로 두는 이유: narration 은 큐 정원 초과 시 가장 먼저 버려진다
(tts_queue._trim). 방향 안내가 사라지면 사용자는 어디로 가는지 모른 채 끌려간다.

안전 원칙: 알리기만 한다. /cmd_vel*, Nav2 goal, CAN 없음.

실행:
    source /opt/ros/humble/setup.bash && source ../vica_ros2_ws/install/setup.bash
    .venv/bin/python -m src.ros_audio_cue_node
"""
from __future__ import annotations

import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose
from vica_interfaces.msg import TurnGuide

from . import audio_cue
from .cue_logic import (
    DIRECTION_LEFT,
    GUIDANCE_ARRIVED_EVENT,
    GUIDANCE_END_EVENTS,
    TurnAnnouncer,
    parse_goal_event,
)
from .replies import TURN_LEFT
from .tts_queue import RESPONSE, build_request


class AudioCueNode(Node):
    def __init__(self) -> None:
        super().__init__("vica_audio_cue")
        self._announcer = TurnAnnouncer()
        self._tts_pub = self.create_publisher(String, "/vica/tts_request", 10)
        self.create_subscription(TurnGuide, "/vica/turn_guide", self._on_turn, 10)
        self.create_subscription(String, "/vica_goal_event", self._on_goal_event, 10)
        # 잔여거리 — Nav2 가 주행 내내 발행하는 피드백을 구독만 한다 (ros2 무변경).
        self.create_subscription(
            NavigateToPose.Impl.FeedbackMessage,
            "/navigate_to_pose/_action/feedback",
            self._on_nav_feedback, 10)
        # 발화 판정은 타이머가 한다 — "0.8초 지속" 확인은 신호가 아니라 시간이 정한다.
        self.create_timer(0.1, self._poll_turn)
        self.get_logger().info(
            "VICA 청각 안내 시작 (구독: /vica/turn_guide, /vica_goal_event)")

    def _on_turn(self, msg: TurnGuide) -> None:
        self._announcer.on_turn(
            direction=msg.direction,
            phase=msg.phase,
            sequence_id=msg.sequence_id,
            now=time.time(),
            source_stale=msg.source_stale,
        )

    def _on_nav_feedback(self, msg) -> None:
        self._announcer.set_distance(
            float(msg.feedback.distance_remaining), time.time())

    def _poll_turn(self) -> None:
        text = self._announcer.poll(time.time())
        if text is None:
            return
        # 효과음만 낸다 — "좌회전 할게요" 멘트는 방송하지 않는다 (2026-08-31
        # 사용자 결정: 회전 안내는 효과음 + 스마트핸들 서보·LED 로 충분하고,
        # 잦은 회전마다 말이 나오면 소음이다). 멘트 복원은 git 이력 참고.
        audio_cue.play(
            audio_cue.turn_left() if text == TURN_LEFT else audio_cue.turn_right()
        )
        self.get_logger().info(f"🔀 회전 안내: {text}")

    def _on_goal_event(self, msg: String) -> None:
        event = parse_goal_event(msg.data)
        if event is None:
            # payload 형식이 어긋나면 도착음이 조용히 사라진다. 단서를 남긴다
            # (핸들 노드가 2026-07-29 실기에서 겪은 함정과 같다).
            self.get_logger().warn(
                "/vica_goal_event 파싱 실패 — 도착음·회전 리셋이 동작하지 않습니다. "
                f"JSON 에 event 키가 필요합니다. payload={(msg.data or '')[:120]!r}",
                throttle_duration_sec=5.0,
            )
            return
        if event == GUIDANCE_ARRIVED_EVENT:
            # 도착 안내 멘트는 Mission Manager 가 말한다. 여기서는 그 앞에 음만
            # 붙여 "이제 안내가 나온다" 를 알린다.
            audio_cue.play(audio_cue.arrived())
            self.get_logger().info("🏁 도착음")
        if event in GUIDANCE_END_EVENTS:
            # 다음 안내의 첫 회전을 놓치지 않도록 중복 억제를 푼다.
            self._announcer.reset()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AudioCueNode()
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

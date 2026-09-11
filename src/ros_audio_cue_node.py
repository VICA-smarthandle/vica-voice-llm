"""VICA 청각 안내 노드 — 도착을 소리로 알린다.

구독: /vica_goal_event   (std_msgs/String)          — 안내 시작·끝

회전 안내("좌회전 할게요"·"우회전 할게요"와 그 효과음)는 2026-09-11 에 뺐다
(사용자 결정): 회전은 스마트핸들의 서보·LED 만으로 안내한다. 실기에서 이
노드의 회전 효과음은 애초에 들린 적이 없었다 — reSpeaker 재생 장치를 TTS
노드가 상시 점유(audio_out._persistent_stream)하고 있어 이 노드의 재생 요청은
장치 사용 중 오류로 조용히 실패한다(stderr 에만 남는다). **도착음도 같은
이유로 지금은 들리지 않는다** — 살리려면 TTS 노드에 재생을 부탁하는 길로
바꿔야 한다(단일 출구 원칙). 회전 안내 복원은 git 이력 참고.

안전 원칙: 알리기만 한다. /cmd_vel*, Nav2 goal, CAN 없음.

실행:
    source /opt/ros/humble/setup.bash && source ../vica_ros2_ws/install/setup.bash
    .venv/bin/python -m src.ros_audio_cue_node
"""
from __future__ import annotations

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from . import audio_cue
from .cue_logic import GUIDANCE_ARRIVED_EVENT, parse_goal_event


class AudioCueNode(Node):
    def __init__(self) -> None:
        super().__init__("vica_audio_cue")
        self.create_subscription(String, "/vica_goal_event", self._on_goal_event, 10)
        self.get_logger().info("VICA 청각 안내 시작 (구독: /vica_goal_event)")

    def _on_goal_event(self, msg: String) -> None:
        event = parse_goal_event(msg.data)
        if event is None:
            # payload 형식이 어긋나면 도착음이 조용히 사라진다. 단서를 남긴다
            # (핸들 노드가 2026-07-29 실기에서 겪은 함정과 같다).
            self.get_logger().warn(
                "/vica_goal_event 파싱 실패 — 도착음이 동작하지 않습니다. "
                f"JSON 에 event 키가 필요합니다. payload={(msg.data or '')[:120]!r}",
                throttle_duration_sec=5.0,
            )
            return
        if event == GUIDANCE_ARRIVED_EVENT:
            # 도착 안내 멘트는 Mission Manager 가 말한다. 여기서는 그 앞에 음만
            # 붙여 "이제 안내가 나온다" 를 알린다.
            audio_cue.play(audio_cue.arrived())
            self.get_logger().info("🏁 도착음")


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

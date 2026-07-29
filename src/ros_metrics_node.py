"""[SIM ONLY] 계측 ROS2 노드 — 모든 서비스 이벤트와 시스템 사용량을 기록한다.

구독: /vica/wake · /vica/user_text · /vica/intent · /vica/emergency
      /vica/tts_active · /vica/sim/event
기록: logs/sim/<세션>.jsonl  (이벤트 원본 + 1초 간격 시스템 사용량)
      세션 이름: VICA_SIM_SESSION 환경변수, 없으면 시각

요약 보고서는 종료 후: .venv/bin/python tools/metrics_report.py logs/sim/<세션>.jsonl

실행:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    .venv/bin/python -m src.ros_metrics_node
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String
from vica_interfaces.msg import EmergencyEvent as EmergencyEventMsg
from vica_interfaces.msg import VicaIntent as VicaIntentMsg

from .metrics import sample_system

LOG_DIR = Path("logs/sim")


class MetricsNode(Node):
    def __init__(self) -> None:
        super().__init__("vica_metrics")
        session = os.environ.get(
            "VICA_SIM_SESSION", datetime.now().strftime("sim_%Y%m%d_%H%M"))
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._path = LOG_DIR / f"{session}.jsonl"
        self._f = self._path.open("a", encoding="utf-8")

        self.create_subscription(String, "/vica/wake",
                                 lambda m: self._event("wake", m.data), 10)
        self.create_subscription(String, "/vica/user_text",
                                 lambda m: self._event("user_text", m.data), 10)
        self.create_subscription(VicaIntentMsg, "/vica/intent",
                                 lambda m: self._event("intent", m.intent or "(reply)"), 10)
        self.create_subscription(EmergencyEventMsg, "/vica/emergency",
                                 lambda m: self._event("emergency", m.keyword), 10)
        self.create_subscription(Bool, "/vica/tts_active",
                                 lambda m: self._event("tts_start" if m.data else "tts_end", ""), 10)
        self.create_subscription(String, "/vica/sim/event",
                                 lambda m: self._event("sim_event", m.data), 10)
        self.create_timer(1.0, self._sample)
        self.get_logger().info(f"계측 시작 → {self._path}")

    def _write(self, rec: dict) -> None:
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._f.flush()

    def _event(self, kind: str, detail: str) -> None:
        self._write({"t": time.time(), "type": "event", "kind": kind, "detail": detail})

    def _sample(self) -> None:
        s = sample_system()
        if s:
            self._write({"t": time.time(), "type": "sys", **s})


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MetricsNode()
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

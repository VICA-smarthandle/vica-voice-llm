"""[SIM ONLY] 로봇 없는 전체 서비스 시뮬레이션 launch.

실기 launch(vica_voice.launch.py)와의 차이:
  - 스텁 2개 대신 가상 로봇(ros_robot_sim) — 주행 시간·도착 안내·E-stop 래치 모사
  - 계측 노드(ros_metrics_node) 추가 — 단계별 시간·시스템 사용량을 logs/sim/ 에 기록

실행:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    VICA_SIM_SESSION=평가1 ros2 launch launch/vica_sim.launch.py
종료 후 보고서:
    .venv/bin/python tools/metrics_report.py logs/sim/평가1.jsonl
"""
import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")


def _python_node(module: str, name: str) -> ExecuteProcess:
    return ExecuteProcess(
        cmd=[VENV_PYTHON, "-m", module],
        cwd=PROJECT_ROOT,
        name=name,
        output="screen",
    )


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            _python_node("src.ros_node", "vica_llm"),
            _python_node("src.ros_tts_node", "vica_tts"),
            _python_node("src.ros_wakeword_node", "vica_wakeword"),
            _python_node("src.ros_robot_sim", "vica_robot_sim"),
            _python_node("src.ros_metrics_node", "vica_metrics"),
        ]
    )

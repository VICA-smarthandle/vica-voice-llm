"""[SIM ONLY] 로봇 없는 전체 서비스 시뮬레이션 launch.

실기 launch(vica_voice.launch.py)와의 차이:
  - 스텁 2개 대신 가상 로봇(ros_robot_sim) — 주행 시간·도착 안내·E-stop 래치 모사
  - 계측 노드(ros_metrics_node) 추가 — 단계별 시간·시스템 사용량을 logs/sim/ 에 기록
  - 목적지 파일이 저장소의 config/destinations.yaml 로 고정된다 (아래 참고)

목적지 파일을 명시하는 이유:
    ros_node 의 destinations_yaml 기본값은 실기 배포 경로(~/vica_data/...)이고,
    ros_robot_sim 은 load_destinations() 기본값(config/destinations.yaml)을 읽는다.
    그대로 두면 LLM 이 고른 목적지 id 를 가상 로봇이 모르는 상태가 되어, 확인까지
    해 놓고 이동이 시작되지 않는다. 시뮬레이션에서는 양쪽을 같은 파일로 맞춘다.

실행:
    source /opt/ros/humble/setup.bash && source ../vica_ros2_ws/install/setup.bash
    VICA_SIM_SESSION=평가1 ros2 launch launch/vica_sim.launch.py
종료 후 보고서:
    .venv/bin/python tools/metrics_report.py logs/sim/평가1.jsonl
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
SIM_DESTINATIONS = os.path.join(PROJECT_ROOT, "config", "destinations.yaml")


def _python_node(module: str, name: str, ros_args=None) -> ExecuteProcess:
    return ExecuteProcess(
        cmd=[VENV_PYTHON, "-m", module, *(ros_args or [])],
        cwd=PROJECT_ROOT,
        name=name,
        output="screen",
    )


def generate_launch_description() -> LaunchDescription:
    destinations_yaml = LaunchConfiguration("destinations_yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "destinations_yaml", default_value=SIM_DESTINATIONS),
            _python_node(
                "src.ros_node",
                "vica_llm",
                ["--ros-args", "-p", ["destinations_yaml:=", destinations_yaml]],
            ),
            _python_node("src.ros_tts_node", "vica_tts"),
            _python_node("src.ros_wakeword_node", "vica_wakeword"),
            _python_node("src.ros_robot_sim", "vica_robot_sim"),
            _python_node("src.ros_metrics_node", "vica_metrics"),
        ]
    )

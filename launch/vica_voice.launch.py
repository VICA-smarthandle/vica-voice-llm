"""VICA 음성 파이프라인 ROS2 노드 일괄 실행 (launch).

서비스형 노드인 LLM intent 노드 + TTS 노드를 함께 띄운다.

STT 노드는 push-to-talk(엔터 입력) 대화형이라 launch 에 넣지 않는다.
마이크로 말하려면 별도 터미널에서 실행한다:
    source /opt/ros/humble/setup.bash
    .venv/bin/python -m src.ros_stt_node

실행:
    source /opt/ros/humble/setup.bash
    ros2 launch launch/vica_voice.launch.py
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)

# 이 파일(launch/)의 부모가 프로젝트 루트.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")


def _python_node(module: str, name: str, ros_args=None) -> ExecuteProcess:
    """.venv 파이썬으로 모듈을 실행하는 노드 프로세스."""
    return ExecuteProcess(
        cmd=[VENV_PYTHON, "-m", module, *(ros_args or [])],
        cwd=PROJECT_ROOT,
        name=name,
        output="screen",
    )


def generate_launch_description() -> LaunchDescription:
    map_id = LaunchConfiguration("map_id")
    storage_root = LaunchConfiguration("destination_storage_root")
    destinations_yaml = LaunchConfiguration("destinations_yaml")
    default_root = PathJoinSubstitution(
        [EnvironmentVariable("HOME"), "vica_data", "destinations"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("map_id", default_value="vica_map_0630"),
            DeclareLaunchArgument(
                "destination_storage_root",
                default_value=default_root,
            ),
            DeclareLaunchArgument(
                "destinations_yaml",
                default_value=PathJoinSubstitution(
                    [storage_root, map_id, "destinations.yaml"]
                ),
            ),
            _python_node(
                "src.ros_node",
                "vica_llm",
                [
                    "--ros-args",
                    "-p",
                    ["destinations_yaml:=", destinations_yaml],
                ],
            ),
            _python_node("src.ros_tts_node", "vica_tts"),
            # 상시 긴급어 감지 (LLM 우회 안전 경로, Phase 4)
            _python_node("src.ros_emergency_node", "vica_emergency"),
        ]
    )

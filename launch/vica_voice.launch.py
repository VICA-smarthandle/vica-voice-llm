"""VICA 음성 파이프라인 ROS2 노드 일괄 실행 (launch).

서비스형 노드인 LLM intent 노드 + TTS 노드를 함께 띄운다.

STT 노드는 push-to-talk(엔터 입력) 대화형이라 launch 에 넣지 않는다.
마이크로 말하려면 별도 터미널에서 실행한다:
    source /opt/ros/jazzy/setup.bash
    .venv/bin/python -m src.ros_stt_node

실행:
    source /opt/ros/jazzy/setup.bash
    ros2 launch launch/vica_voice.launch.py
"""
import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess

# 이 파일(launch/)의 부모가 프로젝트 루트.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")


def _python_node(module: str, name: str) -> ExecuteProcess:
    """.venv 파이썬으로 모듈을 실행하는 노드 프로세스."""
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
            # 개발용 스텁 (로봇 통합 시 교체/제거). 혼자 전체 흐름을 데모하기 위한 자리표시자.
            _python_node("src.ros_robot_state_stub", "vica_robot_state_stub"),
            _python_node("src.ros_state_machine_stub", "vica_state_machine_stub"),
        ]
    )

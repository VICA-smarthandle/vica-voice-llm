"""VICA 음성 파이프라인 ROS2 노드 일괄 실행 (launch).

LLM intent 노드 + TTS 노드 + 웨이크워드 노드(마이크 앞단)를 함께 띄운다.

마이크 입력은 웨이크워드 노드가 담당한다 — "비카야" 호출 후 말하면
/vica/user_text 로, 긴급어("멈춰" 등)는 whisper 검증을 거쳐 /vica/emergency 로
발행된다 (P1-b, 근거: vica-wakeword/docs/integration-design.md).

개발용 push-to-talk 이 필요하면 웨이크워드 노드 대신 별도 터미널에서:
    .venv/bin/python -m src.ros_stt_node      # (마이크를 두 노드가 동시에 못 쓴다)

실행:
    source /opt/ros/humble/setup.bash
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
            # 웨이크워드 앞단: 호출(비카야) + 긴급어(whisper 검증) — LLM 우회 안전 경로.
            # 기존 ros_emergency_node(whisper 상시)를 대체한다. 롤백 = 아랫줄을
            # ros_emergency_node 로 되돌리고 push-to-talk STT 를 별도 실행.
            _python_node("src.ros_wakeword_node", "vica_wakeword"),
            # 개발용 스텁 (로봇 통합 시 교체/제거). 혼자 전체 흐름을 데모하기 위한 자리표시자.
            _python_node("src.ros_robot_state_stub", "vica_robot_state_stub"),
            _python_node("src.ros_state_machine_stub", "vica_state_machine_stub"),
        ]
    )

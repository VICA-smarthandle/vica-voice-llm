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
            # TTS 도 같은 목적지 경로를 받는다(2026-10-07) — 켤 때 확인 질문·도착 멘트·
            # 입구 방향(M1)을 이 지도의 목적지로 미리 합성한다. 안 넘기면 옛 기본 지도
            # (vica_map_0630)를 데워 첫 도착 말이 0.9초쯤 늦는다.
            _python_node(
                "src.ros_tts_node",
                "vica_tts",
                [
                    "--ros-args",
                    "-p",
                    ["destinations_yaml:=", destinations_yaml],
                ],
            ),
            # 웨이크워드 앞단: 호출(비카야) + 긴급어(whisper 검증) — LLM 우회 안전 경로.
            # 기존 ros_emergency_node(whisper 상시)를 대체한다. 롤백 = 아랫줄을
            # ros_emergency_node 로 되돌리고 push-to-talk STT 를 별도 실행.
            # 목적지 경로는 LLM 노드와 **같은 값**을 넘긴다. 이 노드는 그것으로
            # STT 장소 귀띔을 만든다 — 안 넘기면 옛 지도를 외운 채 듣는다
            # (2026-09-03 수리).
            _python_node(
                "src.ros_wakeword_node",
                "vica_wakeword",
                [
                    "--ros-args",
                    "-p",
                    ["destinations_yaml:=", destinations_yaml],
                ],
            ),
            # 로컬 LLM 폴백용 Ollama 서버 (2026-09-19 설계 §4.4). 다른 노드처럼 같이
            # 뜨고 같이 꺼진다. 포트 11434 를 이미 누가 쓰면 바인드 실패로 끝나고
            # 그 서버를 쓴다 — launch 는 이 종료를 치명으로 보지 않는다.
            # 모델은 미리 올리지 않는다(필요할 때 적재). 올라온 뒤엔 내리지 않는다.
            ExecuteProcess(
                cmd=["ollama", "serve"],
                name="ollama_serve",
                output="screen",
                additional_env={
                    "OLLAMA_KEEP_ALIVE": "-1",
                    "OLLAMA_NUM_PARALLEL": "1",
                    "OLLAMA_MAX_LOADED_MODELS": "1",
                    # 직접 등록한 GGUF(폴백 midm2-mini)는 이게 없으면 Ollama 가
                    # GGUF 내장 Jinja 틀을 골라 system(목적지 목록)을 빠뜨린다
                    # (2026-09-28 실측: 이름 지시 무시·EXAONE 4.0 20.8 %). Go 틀
                    # 경로로 강제해 Modelfile TEMPLATE·JSON 스키마를 살린다.
                    # 공식 태그(gemma4·exaone3.5)는 영향 없음.
                    "OLLAMA_GO_TEMPLATE": "1",
                },
            ),
        ]
    )

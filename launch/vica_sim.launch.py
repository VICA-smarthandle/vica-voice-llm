"""[SIM ONLY] 로봇 없는 전체 서비스 시뮬레이션 launch."""
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

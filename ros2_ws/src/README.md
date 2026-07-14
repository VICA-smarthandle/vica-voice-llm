# vica_interfaces 이관 안내

`vica_interfaces` 패키지(VicaIntent / EmergencyEvent / RobotState msg)는
통합 진행순서 ⓪에 따라 **로봇 워크스페이스(vica_ros2_ws)의 `src/vica_interfaces`로
이관**되었습니다. 단일 소스는 이제 로봇 워크스페이스입니다 — 여기에 복사본을
다시 만들지 마세요.

이 저장소의 ROS2 노드(`src/ros_*.py`)를 실행하기 전에 로봇 워크스페이스를
빌드하고 source 하면 msg를 그대로 사용할 수 있습니다:

```bash
cd ~/vica_ros2_ws   # 또는 로봇 ws 경로
colcon build --packages-select vica_interfaces
source install/setup.bash
```

msg 정의를 변경할 때는 로봇 워크스페이스에서 수정한 뒤 양쪽(로봇·음성)을
재빌드/재시험해야 합니다. 계약 문서는 `docs/ros2-interface.md` 참조.

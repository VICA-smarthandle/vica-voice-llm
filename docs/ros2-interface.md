# VICA 음성·LLM ROS 2 계약

검토 기준일: 2026-07-26

## 1. 현재 그래프

```text
ros_stt_node
├─ /vica/user_text ───────────────→ ros_node
└─ /vica/tts_request ─────────────→ ros_tts_node

ros_node
├─ /vica/intent ──────────────────→ Mission Manager
├─ /vica/tts_request ─────────────→ ros_tts_node
└─ /vica/robot_state ←───────────── Mission Manager

ros_emergency_node
├─ /vica/emergency ───────────────→ Mission Manager + E-stop bridge
└─ /vica/tts_state ←─────────────── ros_tts_node

Mission Manager
└─ /vica/tts_request ─────────────→ ros_tts_node
```

## 2. Topic

| 이름 | 타입 | producer | consumer | 의미 |
| --- | --- | --- | --- | --- |
| `/vica/user_text` | `std_msgs/msg/String` | STT | LLM node | 인식 문장 |
| `/vica/intent` | `vica_interfaces/msg/VicaIntent` | LLM node | Mission Manager | 이동 명령이 아닌 의도 후보 |
| `/vica/robot_state` | `vica_interfaces/msg/RobotState` | Mission Manager | LLM node | 층·건물·이동 상태 |
| `/vica/emergency` | `vica_interfaces/msg/EmergencyEvent` | 긴급어 감시 | Mission Manager, E-stop bridge | LLM 우회 긴급 이벤트 |
| `/vica/tts_request` | `std_msgs/msg/String` | STT, LLM, Mission Manager | TTS | `priority:text` 재생 요청 |
| `/vica/tts_state` | `std_msgs/msg/Bool` | TTS | 긴급어 감시 | 로봇 음성 재생 중 여부 |

기본 QoS는 depth 10 reliable이다.

## 3. 공용 메시지

정본은 `vica_ros2_ws/src/vica_interfaces/`다. 음성 저장소에는 메시지 사본을 두지 않으며,
`vica_ros2_ws`를 빌드하고 source한 환경에서 import한다.

### `VicaIntent`

```text
string intent
string destination_candidate
string matched_destination_id
float32 confidence
bool need_confirm
string reply
string safety_flag
```

Mission Manager는 최소한 다음 조건과 자체 gate를 함께 검사한다.

```text
intent == navigate
matched_destination_id != ""
need_confirm == false
safety_flag == normal
```

### `RobotState`

```text
int32 current_floor
string current_building
bool is_moving
```

### `EmergencyEvent`

```text
string keyword
string source_text
float64 detected_at
```

하드 긴급어는 `멈춰`, `정지`, `스탑`, `스톱`, `안돼`, `위험해`다. 음성 node는
감지 이벤트만 발행하고, E-stop bridge가 `/voice_emergency_stop` 펄스로 변환한다.

## 4. 안전 경계

- 음성·LLM은 `/cmd_vel*`, Nav2 action과 CAN을 발행하지 않는다.
- `/vica/intent`는 Mission gate 입력이며 Goal 자체가 아니다.
- `/vica/emergency`의 실제 정지 권한은 중앙 E-stop 래치와 Safety Supervisor에 있다.
- 음성·STT에는 E-stop reset 권한이 없다.
- TTS 재생 중 감시 억제는 자가 오탐 방지 기능이며 물리 E-stop을 대체하지 않는다.

## 5. 실행 전제

```bash
source /opt/ros/humble/setup.bash
cd ../vica_ros2_ws
colcon build --packages-select vica_interfaces
source install/setup.bash
cd ../vica-voice-llm
ros2 launch launch/vica_voice.launch.py
```

push-to-talk STT는 별도 터미널에서 `.venv/bin/python -m src.ros_stt_node`로 실행한다.
개발 stub은 없으므로 Mission Manager와 `vica_safety`를 별도로 기동해야 한다.
실제 음성·Mission·E-stop 종단은 `[미검증]`이다.

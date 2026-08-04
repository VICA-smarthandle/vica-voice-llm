# VICA 음성·LLM ROS 2 계약

검토 기준일: 2026-07-26

## 1. 현재 그래프

```text
ros_wakeword_node                     ← 마이크 앞단 (상시). launch 의 기본 진입 경로
├─ /vica/user_text ───────────────→ ros_node
├─ /vica/emergency ───────────────→ Mission Manager + E-stop bridge
├─ /vica/wake ────────────────────→ (음성 저장소 내부 — 계측·UI 앵커)
└─ /vica/tts_state ←─────────────── ros_tts_node

ros_node
├─ /vica/intent ──────────────────→ Mission Manager
├─ /vica/tts_request ─────────────→ ros_tts_node
└─ /vica/robot_state ←───────────── Mission Manager

Mission Manager
└─ /vica/tts_request ─────────────→ ros_tts_node
```

launch 에 들어가지 않는 대체 경로 2개 (계약은 위와 동일):

```text
ros_stt_node        개발용 push-to-talk. 마이크를 웨이크워드와 동시에 못 쓴다
└─ /vica/user_text · /vica/tts_request

ros_emergency_node  whisper 상시 감시. 웨이크워드 롤백용으로 남겨 둔다
└─ /vica/emergency ·  /vica/tts_state ←
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

"긴급어 감시"의 현재 구현은 `ros_wakeword_node`다(`ros_emergency_node`는 롤백용).
`keyword`는 whisper 전사에서 정확 매칭으로 뽑으므로 값 범위는 종전과 같다 —
브리지·래치 체인은 변경되지 않는다.

`/vica/tts_state`는 문장 단위로 켜지고 꺼진다. 한 발화가 여러 번 true/false를
낼 수 있으므로, 소비자는 첫 true를 재생 시작으로, 마지막 false를 종료로 본다.

음성 저장소 내부 토픽(팀 계약 아님, 임의 변경 가능):

| 이름 | 타입 | 용도 |
| --- | --- | --- |
| `/vica/wake` | `std_msgs/msg/String` | 호출 감지 앵커 — 계측의 "체감 응답" 기준점 |
| `/vica/sim/event` | `std_msgs/msg/String` | **[SIM ONLY]** 가상 로봇 상태 변화 |
| `/vica/sim/reset` | `std_msgs/msg/Empty` | **[SIM ONLY]** 래치 해제 (실기의 관리자 앱 reset 자리) |

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

`intent` 허용값의 정본은 `vica_ros2_ws/src/vica_interfaces/msg/VicaIntent.msg`다.
2026-07-27에 진행 중인 안내를 조작하는 `cancel`/`pause`/`resume`가 추가됐다
(세 값에는 `matched_destination_id`가 필요 없다).

**[GAP]** 음성 저장소의 파서는 아직 `navigate`/`question`/`clarify`/`unknown`
네 값만 만든다 (`src/schema.py`의 `VicaIntentType`). 세 값이 실제로 필요한 시점은
로봇 팀과 확인이 필요하다.

### `RobotState`

```text
int32 current_floor
string current_building
bool is_moving
bool is_paused     # 2026-07-27 정본 추가. 목적지를 기억한 채 멈춘 상태
```

`is_paused`는 `is_moving=false`만으로는 구분되지 않는 "일시정지"를 나타낸다.
**[GAP]** 음성 저장소는 아직 이 필드를 만들지도 읽지도 않는다 (`ros_robot_sim`
포함). `cancel`/`pause`/`resume` intent 지원과 함께 다뤄야 한다.

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

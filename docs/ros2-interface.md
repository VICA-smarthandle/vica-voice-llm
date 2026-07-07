# VICA 음성/LLM 파트 — ROS2 인터페이스 명세 (로봇 팀 전달용)

작성: 2026-07-07. 음성/LLM 파트가 발행·구독하는 모든 토픽과 메시지 정의,
그리고 로봇 팀이 구현/교체해야 할 부분을 정리한다.

## 전체 그래프

```text
[마이크] ─ ros_stt_node ──/vica/user_text──▶ ros_node (LLM) ──/vica/intent──▶ ros_tts_node ─▶ [스피커]
                                                 ▲                              │
                              /vica/robot_state ─┘                              ▼
                              (로봇 팀 발행)                        ★ state machine (로봇 팀)

[마이크] ─ ros_emergency_node ──/vica/emergency──▶ ★ safety supervisor / state machine (로봇 팀)
          (상시 감시, LLM 우회)
```

★ = 현재 개발용 스텁(`ros_state_machine_stub`, `ros_robot_state_stub`)이 자리를
차지하고 있으며, **로봇 팀의 실제 노드로 교체 대상**이다.

## 토픽 목록

| 토픽 | 타입 | 발행 | 구독 | 설명 |
|---|---|---|---|---|
| `/vica/user_text` | `std_msgs/String` | ros_stt_node | ros_node | STT 인식 결과 (한국어 문장) |
| `/vica/intent` | `vica_interfaces/VicaIntent` | ros_node | ros_tts_node, **state machine** | LLM 의도 해석 결과 (아래 상세) |
| `/vica/robot_state` | `vica_interfaces/RobotState` | **로봇 팀** | ros_node | 로봇 현재 상태 (질문 답변에 활용) |
| `/vica/emergency` | `vica_interfaces/EmergencyEvent` | ros_emergency_node | **safety supervisor** | 긴급어 감지 이벤트 (LLM 우회) |

QoS 는 모두 기본 프로파일 depth 10 이다.

## 메시지 정의 (`vica_interfaces`)

### VicaIntent — LLM 의 '제안' (이동 명령 아님)

```text
string intent                  # navigate / question / clarify / unknown
string destination_candidate   # LLM 이 고른 목적지 표현 (없으면 "")
string matched_destination_id  # 코드가 확정한 목적지 id (없으면 "")
float32 confidence             # 0.0 ~ 1.0
bool need_confirm              # true 면 사용자 확인이 아직 안 끝났다
string reply                   # 사용자에게 들려줄 한국어 답변 (TTS 가 재생)
string safety_flag             # normal / emergency
```

state machine 이 이동을 시작해도 되는 조건 (모두 만족해야 함):

```text
intent == "navigate"
matched_destination_id != ""     # 목적지가 DB 에서 확정됨
need_confirm == false            # 사용자가 확인을 마침 ("응 맞아" 등)
safety_flag == "normal"
```

이 조건을 만족해도 **최종 이동 판단은 state machine 몫**이다
(현재 이동 가능 상태, 접근 가능 여부, safety supervisor 확인 등).

### RobotState — 로봇 팀이 발행

```text
int32 current_floor    # 층. 알 수 없으면 -1
string current_building
bool is_moving
```

"지금 몇 층이야?" 같은 질문 답변에 쓰인다. 주기 발행(예: 1Hz) 또는 변경 시 발행.

### EmergencyEvent — 긴급 정지 요청 (최우선 처리)

```text
string keyword        # 매칭된 긴급어 (예: "멈춰")
string source_text    # STT 가 인식한 원본 텍스트
float64 detected_at   # 감지 시각 (unix time)
```

- 마이크 상시 감시로 감지되며, **LLM 을 전혀 거치지 않는다** (감지 지연 약 1초).
- 긴급어 목록: 멈춰, 정지, 스탑, 스톱, 안돼, 위험해, 잠깐, 천천히, 느리게
- 이 이벤트를 받으면 safety supervisor / state machine 이 즉시 정지를 판단·실행한다.

## 안전 계약 (음성/LLM 파트가 보장하는 것)

- `/cmd_vel`, `/cmd_vel_safe` 를 발행하지 않는다.
- Nav2 goal 을 직접 보내지 않는다.
- 모터/속도/회전/정지 명령을 실행하지 않는다.
- `VicaIntent` 는 제안일 뿐이며, 실제 이동·정지의 결정과 실행은
  로봇 팀의 state machine / safety supervisor 가 한다.

## 로봇 팀이 할 일

1. **state machine 노드**: `/vica/intent` 와 `/vica/emergency` 구독,
   위 조건 검사 후 Nav2 goal 생성 여부 결정. (`src/ros_state_machine_stub.py` 참고 후 교체)
2. **robot_state 발행**: `/vica/robot_state` 를 실제 값으로 발행.
   (`src/ros_robot_state_stub.py` 교체)
3. launch 파일(`launch/vica_voice.launch.py`)에서 스텁 2개를 제거하고 실제 노드 연결.

## 빌드/실행

```bash
# 메시지 패키지 빌드 (최초 1회)
source /opt/ros/humble/setup.bash
cd ros2_ws && colcon build --packages-select vica_interfaces && cd ..

# 음성 파트 실행
source ros2_ws/install/setup.bash
ros2 launch launch/vica_voice.launch.py        # LLM + TTS + 긴급 감시 + (스텁)
# 별도 터미널: .venv/bin/python -m src.ros_stt_node   (push-to-talk 마이크)

# 동작 확인
ros2 topic echo /vica/intent
ros2 topic echo /vica/emergency   # "멈춰!" 외치면 수신됨
```

주의: `ros2 topic echo` 는 발행자가 아직 없으면 즉시 종료한다. 노드를 먼저 띄울 것.

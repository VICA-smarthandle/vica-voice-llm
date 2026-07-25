# VICA 음성·LLM 현재 동작

검토 기준일: 2026-07-26

## 1. 역할과 권한

음성 저장소는 사용자 발화를 텍스트와 `VicaIntent`로 변환하고, 긴급어를 LLM보다 먼저
감지하며, 여러 구성요소의 안내 문구를 TTS로 재생한다.

```text
일반 발화
마이크 → STT → /vica/user_text → LLM·코드 검증 → /vica/intent
                                      └──────────→ /vica/tts_request → TTS

긴급 발화
마이크 → 긴급어 감시 → /vica/emergency → Mission E-stop bridge
                                      → emergency_stop_node 중앙 래치
```

이 저장소는 Nav2 Goal, `/cmd_vel*` 또는 CAN을 직접 발행하지 않는다. 목적지 후보의 최종
수락과 Goal 생성은 `vica_ros2_ws/src/vica_mission_manager/`, 주행 명령의 최종 승인은
`vica_ros2_ws/src/vica_safety/`가 담당한다.

## 2. 현재 실행 구성

`launch/vica_voice.launch.py`는 다음 세 프로세스를 실행한다.

- `src.ros_node`: LLM 의도 해석과 `VicaIntent` 발행
- `src.ros_tts_node`: `/vica/tts_request` 우선순위 재생
- `src.ros_emergency_node`: 상시 긴급어 감시

push-to-talk STT는 터미널 입력이 필요하므로 별도로 실행한다.

```bash
source /opt/ros/humble/setup.bash
source ../vica_ros2_ws/install/setup.bash
ros2 launch launch/vica_voice.launch.py
```

```bash
source /opt/ros/humble/setup.bash
source ../vica_ros2_ws/install/setup.bash
.venv/bin/python -m src.ros_stt_node
```

개발용 RobotState·state-machine stub, 테스트용 FastAPI/SQLite backend와 저장소 내부
`vica_interfaces` 사본은 제거됐다.

## 3. 일반 발화 처리

1. `ros_stt_node`가 음성을 `/vica/user_text`로 발행한다.
2. `ros_node`가 하드 긴급어를 먼저 검사한다.
3. 일반 발화면 LLM이 목적지 표현과 의도를 구조화한다.
4. `destination_matcher.py`가 등록된 목적지에서 실제 ID를 검증한다.
5. `VicaIntent`를 `/vica/intent`로 발행한다.
6. 확인·질문 응답은 `/vica/tts_request`로 보내고, navigate 확정 결과 문구는 Mission
   Manager가 gate 판정 뒤 발행한다.

LLM은 `destination_candidate`만 제안한다. `matched_destination_id`는 코드가 등록
목적지와 대조해 채운다.

## 4. 목적지 데이터

ROS 실행의 기본 정본은 다음 지도별 파일이다.

```text
~/vica_data/destinations/<map_id>/destinations.yaml
```

`ros_node`는 파일 변경 시각을 확인하고 다음 발화 전에 다시 읽으며,
`authorization == public`인 목적지만 LLM에 제공한다. 파일이 없으면 빈 catalog로
시작한다. CLI 모드는 로컬 시험용 `config/destinations.yaml`을 사용한다.

목적지 저장·삭제는 `vica_destination_manager`, Goal 요청 검증은 Mission Manager가
담당한다. 음성 저장소는 목적지를 수정하지 않는다.

## 5. 긴급어와 TTS

하드 긴급어는 다음 6개다.

```text
멈춰, 정지, 스탑, 스톱, 안돼, 위험해
```

`잠깐`, `천천히`, `느리게`는 E-stop 키워드가 아니며 일반 발화로 처리한다. 감속
intent는 아직 별도 구현 대상이다.

TTS는 `/vica/tts_request`의 `"<priority>:<text>"` 형식을 파싱해 큐로 재생하고,
`/vica/tts_state`에 재생 중 여부를 발행한다. 긴급어 감시는 로봇 음성의 자가 트리거를
막기 위해 재생 중 잠시 멈추며, TTS 종료 신호가 없어도 제한 시간 뒤 자동 재개한다.

## 6. 핵심 파일

| 경로 | 역할 |
| --- | --- |
| `launch/vica_voice.launch.py` | LLM·TTS·긴급어 감시 실행 |
| `src/ros_stt_node.py` | push-to-talk STT |
| `src/ros_node.py` | LLM, 목적지 검증, intent·TTS 요청 |
| `src/ros_tts_node.py` | TTS 큐 재생과 상태 발행 |
| `src/ros_emergency_node.py` | 상시 긴급어 감시 |
| `src/emergency_filter.py` | LLM 이전 하드 긴급어 판정 |
| `src/destination_loader.py` | 지도별 YAML 로드 |
| `src/destination_matcher.py` | 목적지 후보를 등록 ID로 검증 |
| `src/tts_queue.py` | TTS 우선순위·중복·선점 정책 |
| `src/history.py` | 제한 시간 기반 멀티턴 문맥 |

토픽과 메시지 상세 계약은 `docs/ros2-interface.md`, 장치 시험 순서는
`docs/voice-field-test.md`를 따른다. 실제 마이크·스피커·Mission·Safety 종단은
`[미검증]`이다.

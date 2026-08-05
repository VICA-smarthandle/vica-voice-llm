# VICA 음성·LLM 데이터 설계

검토 기준일: 2026-07-26

## 1. 설계 원칙

```text
이해와 대화: LLM
목적지 확정: 코드
Goal 승인: Mission Manager
긴급정지: LLM 우회 → 중앙 E-stop 래치
주행 최종 승인: Safety Supervisor
```

LLM 출력은 이동 명령이 아니다. LLM은 `/cmd_vel*`, Nav2 action과 CAN을 직접 사용하지
않으며 E-stop reset 권한도 갖지 않는다.

## 2. `VicaIntent` 계약

| 필드 | 주체 | 의미 |
| --- | --- | --- |
| `intent` | LLM + 코드 | `navigate`, `question`, `clarify`, `unknown` |
| `destination_candidate` | LLM | 사용자가 표현한 목적지 후보 |
| `matched_destination_id` | 코드 | 등록 catalog와 매칭된 ID |
| `confidence` | LLM | 0.0~1.0 해석 확신도 |
| `need_confirm` | 코드 + LLM | 사용자 확인이 더 필요한지 |
| `reply` | LLM 또는 코드 | 안내 문구 |
| `safety_flag` | 긴급어 필터 | `normal`, `emergency` |

`destination_candidate`가 있어도 등록 ID와 매칭되지 않으면 이동 요청으로 확정하지 않는다.
navigate 확정 요청도 Mission Manager의 지도·UUID·접근권한·pose·E-stop·Nav2·IDLE
gate를 통과해야 한다.

## 3. 목적지 계약

ROS 실행의 정본은 다음 파일이다.

```text
~/vica_data/destinations/<map_id>/destinations.yaml
```

음성 node는 `authorization == public` 목적지만 사용한다. 주요 필드는 다음과 같다.

```text
id, name, aliases, category1, category2
building, floor, owner, authorization
is_approachable, unavailable_reason
pose(frame_id, x, y, yaw)
confirm_prompt, arrival_message
```

목적지 저장·삭제는 `vica_destination_manager`가 담당한다. 음성 node는 파일 변경 시
다음 발화 전에 catalog를 다시 읽고, 로드 실패 시 기존 목록을 유지한다.

## 4. 대화와 실패 처리

- 최근 발화 history로 “응 맞아” 같은 확인 응답을 해석한다.
- 대화 timeout 뒤에는 이전 사용자의 문맥을 버린다.
- LLM 호출 실패는 `unknown`과 재시도 문구로 처리한다.
- navigate 확정의 최종 안내 문구는 Mission gate 결과를 아는 Mission Manager가
  `/vica/tts_request`로 발행한다.

## 5. 긴급어와 TTS

하드 긴급어 6개는 LLM 호출 전에 규칙으로 감지한다.

```text
멈춰, 정지, 스탑, 스톱, 안돼, 위험해
```

`잠깐`, `천천히`, `느리게`는 E-stop이 아니다. 감속 intent는 `[TARGET]`이다.

TTS 입력은 `/vica/tts_request` 하나로 통합한다. `tts_queue.py`가 우선순위, 중복 억제와
긴급 발화 선점을 처리하고, `ros_tts_node.py`가 `/vica/tts_state`를 발행해 긴급어
감시의 자가 트리거를 줄인다.

## 6. 코드 구조

```text
launch/vica_voice.launch.py     실기       (LLM·TTS·웨이크워드·청각 안내)
launch/vica_sim.launch.py       [SIM ONLY] (+ 가상 로봇·계측)
src/
├── ros_wakeword_node.py        마이크 앞단 (호출 + 긴급, launch 기본)
│   ├── wakeword_monitor.py     상시 감시 엔진
│   └── wakeword_gate.py        관문·정확 매칭 (순수 로직)
├── ros_audio_cue_node.py       회전·도착 청각 안내
│   ├── cue_logic.py            판정 (순수 로직)
│   └── audio_cue.py            안내음 생성·재생
├── ros_node.py                 LLM 해석 → /vica/intent
├── ros_tts_node.py             TTS 큐 재생 + /vica/tts_state
├── ros_stt_node.py             push-to-talk (개발용, launch 에 없음)
├── ros_emergency_node.py       whisper 상시 감시 (롤백, launch 에 없음)
├── destination_loader.py
├── destination_matcher.py
├── emergency_filter.py
├── langchain_intent_parser.py
├── tts_queue.py
├── tts_text.py                 문장 분할 (감시 사각지대 축소)
├── replies.py                  고정 문구 (자가 트리거 검사 대상)
├── history.py
├── schema.py
└── ros_convert.py
```

공용 ROS 메시지 정본은 `vica_ros2_ws/src/vica_interfaces/`다. 제거된 개발 stub,
FastAPI/SQLite backend와 음성 저장소 내부 메시지 사본은 현재 구조에 포함하지 않는다.

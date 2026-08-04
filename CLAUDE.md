# VICA LLM Voice Pipeline Roadmap

항상 한국어로 답변한다. 사용자는 LLM/API/STT 개발 입문자이므로 짧고 명확하게 설명하고, 작은 단계로 안내한다.

> **⚠️ 새 세션은 `docs/jetson-handoff.md` 를 먼저 읽는다** (2026-07-29 갱신).
> 웨이크워드 마이크 앞단(P1-a·b)과 가상 로봇 시뮬레이션·계측이 이미 통합돼
> 있고, 확정 결정(전 구간 STT 검증 등)·성능 수치·Jetson 설치 함정·다음 작업
> 우선순위가 그 문서에 있다. 아래 로드맵의 Phase 1~4 서술은 그보다 오래됐다
> (예: Phase 4 의 "openWakeWord 는 별도 실험에서 검증" → 검증 완료·통합됨).

## Project Goal

VICA는 시각장애인 사용자를 위한 자율 안내 로봇이다. 이 저장소는 VICA의 음성/LLM 파트를 개발한다.

최종 목표는 사용자 음성을 안전하게 이해해서 로봇이 이해할 수 있는 구조화 JSON으로 변환하는 것이다.

```text
Voice input
-> STT
-> emergency filter
-> LangChain intent agent
-> destination/tool lookup
-> VicaIntent JSON
-> ROS2 intent service
-> VICA state machine
```

LLM은 로봇을 직접 제어하지 않는다. LLM 출력은 state machine에 전달되는 제안이다.

## Core Safety Rules

- LLM은 `/cmd_vel`을 publish하지 않는다.
- LLM은 `/cmd_vel_safe`를 publish하지 않는다.
- LLM은 Nav2 goal을 직접 보내지 않는다.
- LLM은 모터, 속도, 회전, 정지 명령을 직접 실행하지 않는다.
- 긴급 명령은 LLM 전에 rule-based logic으로 먼저 처리한다.
- 긴급 정지는 Safety Layer / Safety Supervisor / State Machine이 최종 처리한다.
- LangChain tool에는 로봇 이동 tool을 만들지 않는다.

금지 tool 이름 예:

```text
move_robot
publish_cmd_vel
send_nav2_goal
stop_motor
```

## Layer Responsibility

```text
LangChain / LLM
- 사용자 발화 해석
- 목적지 후보 찾기
- 질문 응답
- VicaIntent JSON 생성

Destination Tool
- 목적지 DB/YAML/API 조회
- 로봇 이동 명령 없음

Emergency Filter / Emergency Monitor
- "멈춰", "정지", "스탑", "안돼" 등 긴급어 감지
- LLM 호출 없이 emergency event 생성

State Machine
- 최종 이동 여부 결정
- 사용자 확인 필요 여부 판단
- 목적지 접근 가능 여부 판단

Safety Supervisor
- 긴급 정지
- 위험 상황 override

Navigation / Motion
- Nav2 goal 처리
- 실제 로봇 이동
```

## Current Roadmap

### Phase 1: CLI Prototype

목표: PC에서 로봇 없이 음성 파이프라인을 검증한다.

```text
microphone
-> faster-whisper STT
-> emergency_filter
-> LangChain intent parser
-> VicaIntent JSON
-> TTS
-> terminal output
```

우선순위:

1. `src/main.py`를 LangChain 기반 intent parser와 연결한다.
2. emergency 명령은 LangChain 전에 차단한다.
3. `langchain/test_LLM_ToolEmul.py`는 실험 파일로 유지한다.
4. 실제 pipeline 코드는 `src/` 아래에 둔다.

### Phase 2: Destination Schema Cleanup

현재 목적지 원본은 `config/destinations.yaml`이다.

사용할 필드:

```text
id
name
aliases
category1
category2
building
floor
room
owner
authorization
is_approachable
unavailable_reason
pose
confirm_prompt
arrival_message
```

`safety_level`은 제거했다. 접근 제한은 `is_approachable: false`와 `unavailable_reason`으로 처리한다.

권장 정리:

- `DestinationData`와 `DestinationPose`를 `src/destination_schema.py`로 분리한다.
- `confirm_prompt`, `arrival_message`가 비어 있으면 `name`으로 자동 생성한다.
- LangChain 실험 파일은 이 schema를 import해서 사용한다.

### Phase 3: LangChain Intent Agent

VICA 음성 파이프라인에는 LangChain을 사용한다.

LangChain이 담당할 일:

```text
목적지 요청 이해
일반 질문 응답
반복 요청 처리
목적지 검색 tool 호출
최종 VicaIntent JSON 생성
```

추천 파일:

```text
src/langchain_intent_parser.py
src/destination_tool.py
src/destination_schema.py
```

LangChain agent는 `VicaIntent` 스키마를 반환해야 한다.

```json
{
  "intent": "navigate",
  "destination_candidate": "윤지영 교수님 사무실",
  "confidence": 0.85,
  "need_confirm": true,
  "reply": "윤지영 교수님 사무실로 안내해드릴까요?",
  "safety_flag": "normal"
}
```

### Phase 4: Always-On Emergency Path

긴급 정지는 LLM이 담당하지 않는다.

최종 구조:

```text
Always-on Emergency Monitor
-> openWakeWord / emergency keyword spotting
-> emergency_filter
-> EmergencyEvent
-> Safety Supervisor / State Machine
```

긴급어 후보:

```text
멈춰
정지
스탑
스톱
안돼
위험해
잠깐
천천히
느리게
```

초기 구현은 `emergency_filter.py`의 rule-based keyword detection으로 충분하다. openWakeWord는 별도 실험 파일에서 한국어 긴급어 모델 성능을 먼저 검증한다.

### Phase 5: FastAPI + SQLite Destination Backend (제거됨)

> 이 테스트용 백엔드는 제거되었다. 목적지는 `config/destinations.yaml` 단일 소스로
> 읽고(`src/destination_loader.py`), 목적지·pose 편집은 관리자 앱(VICA_Supervisor)이
> 담당한다. 아래 설계는 기록용으로만 남긴다.

추천 구조:

```text
Admin/Test App
-> FastAPI
-> SQLite
-> destinations API
-> LangChain destination tool
```

초기 API:

```text
GET    /destinations
GET    /destinations/search?query=407호
POST   /destinations
PATCH  /destinations/{id}/pose
```

LangChain tool은 DB를 직접 수정하지 않는다. 목적지 조회만 한다.

관리자 앱이나 calibration tool이 목적지와 pose를 수정한다.

### Phase 6: ROS2 Integration

ROS2 연결은 필요하지만, LangChain이 ROS2 motion API를 직접 호출하면 안 된다.

좋은 구조:

```text
/vica/llm_intent_node
- input: user_text, robot_state
- output: VicaIntent
- no /cmd_vel
- no Nav2 direct call
```

State machine이 할 일:

```text
목적지 존재 확인
현재 이동 가능 상태 확인
사용자 확인 여부 판단
safety supervisor 확인
Nav2 goal 생성 여부 결정
```

## Immediate Next Tasks

현재 우선순위는 다음과 같다.

1. `DestinationData`를 `src/destination_schema.py`로 분리한다.
2. `confirm_prompt`, `arrival_message` 자동 생성 로직을 추가한다.
3. `langchain/test_LLM_ToolEmul.py`의 schema 중복을 제거한다.
4. `src/destination_tool.py`를 만든다.
5. `src/langchain_intent_parser.py`를 만든다.
6. `src/main.py`에서 GPT parser 대신 LangChain parser를 사용하도록 조정한다.
7. emergency filter는 LangChain 호출 전에 유지한다.
8. 목적지는 `config/destinations.yaml` 단일 소스로 유지한다 (테스트용 API 백엔드는 제거됨).

## Testing Rules

- 실제 마이크, 스피커, GPT API, Ollama Cloud, ROS2에 의존하는 테스트는 자동화하지 않는다.
- schema, emergency filter, destination matching, parser failure path는 작은 unit test로 검증한다.
- LangChain/Ollama 실행 파일은 실험 파일로 두고, 핵심 로직은 import 가능한 `src/` 모듈로 분리한다.

## Current Important Files

```text
AGENTS.md
config/destinations.yaml
src/schema.py
src/emergency_filter.py
src/destination_matcher.py
src/main.py
langchain/test_LLM_ToolEmul.py
docs/superpowers/specs/2026-06-25-voice-llm-supertonic-pipeline-design.md
docs/superpowers/plans/2026-06-25-voice-llm-supertonic-pipeline.md
```

## Portfolio Message

VICA의 LLM 모듈은 한국어 사용자 음성을 검증된 구조화 intent JSON으로 변환한다. LLM은 로봇을 직접 제어하지 않으며, 긴급 정지는 LLM을 우회하는 안전 경로에서 처리된다. 최종 이동 판단은 ROS2 state machine과 safety supervisor가 수행한다.

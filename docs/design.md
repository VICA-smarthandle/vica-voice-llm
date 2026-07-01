# VICA 음성/LLM 파이프라인 설계 (데이터 계약)

이 문서는 VICA 음성 파이프라인의 **확정된 데이터 모양(계약)** 을 기록한다.
코드보다 먼저 "들어오는 데이터 모양"과 "나가는 데이터 모양"을 못 박는 것이 목적이다.

상위 로드맵은 `CLAUDE.md`를 참고한다. 이 문서는 그중 **스키마 확정** 부분이다.

## 파이프라인 개요

```text
Voice input
-> STT (faster-whisper)
-> emergency filter        # 긴급어는 여기서 LLM 없이 차단
-> LLM intent 파싱          # 발화 -> VicaIntent (LLM은 후보만 뽑음)
-> destination 매칭         # 파이썬 코드가 실제 목적지 확정
-> VicaIntent JSON
-> ROS2 intent service
```

LLM은 로봇을 직접 제어하지 않는다. LLM 출력은 state machine에 전달되는 **제안**이다.

---

## 1. 출력 계약: VicaIntent

LLM 파이프라인의 최종 출력 JSON.

```json
{
  "intent": "navigate",
  "destination_candidate": "윤지영 교수님 사무실",
  "matched_destination_id": "engineering_4f_room_407_prof_yoon_jiyoung_office",
  "confidence": 0.85,
  "need_confirm": true,
  "reply": "윤지영 교수님 사무실로 안내해드릴까요?",
  "safety_flag": "normal"
}
```

### 필드 설명

| 필드 | 타입 | 채우는 주체 | 설명 |
|---|---|---|---|
| `intent` | enum | LLM | `navigate` / `question` / `unknown` |
| `destination_candidate` | string\|null | **LLM** | 사용자가 말한 목적지 표현(원문에 가까움) |
| `matched_destination_id` | string\|null | **파이썬 코드** | 실제 목적지 DB와 매칭된 결과 id |
| `confidence` | float (0~1) | LLM | 해석 확신도 |
| `need_confirm` | bool | 코드(+LLM) | 이동 전 사용자 확인이 필요한지 |
| `reply` | string | LLM | 사용자에게 들려줄 음성 답변 |
| `safety_flag` | enum | 필터/코드 | `normal` / `emergency` |

### 핵심 설계 원칙

> `destination_candidate`는 **LLM이** 채우고,
> `matched_destination_id`는 **파이썬 코드가** 채운다.

LLM은 "사용자가 말한 목적지 표현"만 뽑고, **실제 어느 목적지인지 확정(매칭)은 코드가** 한다.
이렇게 역할을 나누면:

- LLM이 없는 목적지를 지어내도 코드 매칭에서 걸러진다.
- 매칭 로직을 단위 테스트로 검증할 수 있다.
- 안전성이 올라가고 LLM 호출 비용/지연이 준다.

### intent 종류

| intent | 의미 | 예시 발화 |
|---|---|---|
| `navigate` | 목적지 안내 요청 (**간접 표현 포함**) | "407호 데려다줘", "배 아파"(→화장실) |
| `question` | 일반 질문 / 정보 요청 (이동 아님) | "지금 몇 층이야?", "엘리베이터 어디야?" |
| `clarify` | 모호해서 되물어야 함 | "거기 데려다줘" → "어디로 갈까요?" |
| `unknown` | 안내 범위 밖 / 못 알아들음 | "오늘 날씨 어때?" |

`emergency`는 intent에 **넣지 않는다.** 긴급어("멈춰", "정지" 등)는 LLM 호출 **이전**에
emergency filter가 잡아서 `safety_flag: "emergency"` 로만 표시한다.
→ `CLAUDE.md`의 안전 원칙(긴급 정지는 LLM 우회)과 일치한다.

---

## 1.5 시나리오 대응 방식

대응 목표 시나리오 (2026-06-30 확정): **간접 의도 추론 · 일반 질문 · 멀티턴 대화**.

> 핵심 원칙: **이해와 대화는 LLM, 결정과 안전은 코드.**
> "여러 시나리오 대응"은 Agent 자유도가 아니라 (프롬프트 + 구조화 출력 + 대화 메모리)로 구현한다.

### 간접 의도 추론 ("배 아파" → 화장실)

- 프롬프트에 **사용 가능한 목적지 목록**(name + 주요 alias + category)을 넣어준다.
- LLM은 그 목록 **안에서** `destination_candidate`를 고른다 → 없는 목적지를 지어내지 못한다.
- 코드(`destination_matcher`)가 한 번 더 검증해 `matched_destination_id`를 확정한다.
- 후보가 없거나 모호하면 `intent: "clarify"`로 되묻는다.

### 멀티턴 대화 ("응 맞아" / "아니 그거 말고")

- 파서에 **대화 히스토리(메시지 목록)** 를 함께 넘긴다.
- LLM은 직전 로봇 발화를 보고 "응"/"아니"가 무엇에 대한 답인지 해석한다.
  - "응 맞아" → 직전 후보를 `navigate` + `need_confirm: false`로 확정
  - "아니 그거 말고" → 직전 후보 취소 후 다시 묻기(`clarify`)
- 구현: LLM이 `is_confirmation` 플래그(직전 제안 수락 여부)를 함께 판단하고,
  코드가 그때 `need_confirm: false`로 안내를 시작한다. (2026-06-30 구현·검증)
- history 는 CLI(main.py)와 ROS2 노드(ros_node.py) **둘 다** 유지한다.
  (2026-07-01: ros_node 에 history 가 빠져 있어 ROS2 에서 멀티턴이 안 되던 것을 수정)
- gemma 가 is_confirmation/confidence 를 null 로 줄 수 있어 파서에서 Optional 로 방어한다.
- 더 복잡한 흐름이 필요해지면 `langgraph`로 확인 상태를 명시적으로 그린다.

---

## 2. 입력 계약: destinations

목적지 원본은 `config/destinations.yaml`. 사용 필드:

```text
id
name
aliases
category1
category2
building
floor
room              # 일부 항목만
owner
authorization
is_approachable
unavailable_reason
pose              # frame_id, x, y, yaw
confirm_prompt
arrival_message
```

### 변경 기록

- `safety_level` 필드는 **제거**했다 (2026-06-30).
  접근 제한은 `is_approachable: false` + `unavailable_reason` 으로 표현한다.

### 자동 생성 규칙 (예정)

- `confirm_prompt`가 비어 있으면 → `"{name}(으)로 안내해드릴까요?"`
- `arrival_message`가 비어 있으면 → `"{name} 앞에 도착했습니다."`

---

## 3. 폴더 구조와 단계별 책임

```text
src/
  schema.py                  # VicaIntent, DestinationData 등 데이터 모양 (pydantic)
  destination_loader.py      # destinations.yaml 읽기 + confirm/arrival 자동 생성
  destination_matcher.py     # "목적지 표현" -> 목적지 id 매칭 (평범한 파이썬 함수)
  emergency_filter.py        # 긴급어 차단 (LLM 호출 전)
  langchain_intent_parser.py # Ollama Cloud + 구조화 출력 + 대화 메모리
  stt.py                     # faster-whisper 한국어 음성 인식 (마이크)
  tts.py                     # supertonic 한국어 음성 출력
  ros_node.py                # ROS2 노드: LLM intent (/vica/llm_intent_node)
  ros_stt_node.py            # ROS2 노드: 마이크 STT (-> /vica/user_text)
  ros_tts_node.py            # ROS2 노드: TTS 재생 (/vica/intent 구독)
  ros_convert.py             # pydantic <-> vica_interfaces 메시지 변환
  main.py                    # CLI 파이프라인 연결 (키보드/마이크 입력)

ros2_ws/src/vica_interfaces/ # 커스텀 메시지 패키지 (VicaIntent.msg, RobotState.msg)
```

- LangChain `tool` 기반 `destination_tool.py`는 지금 만들지 않는다.
  평범한 함수 `destination_matcher.py`로 시작하고, 나중에 agent가 정말 필요하면 이 함수를 감싼다.
- `main.py`는 마이크/STT 없이 **키보드 텍스트 입력**으로 먼저 검증한다. faster-whisper는 그 뒤에 붙인다.

---

## 4. 개발 환경 결정

- 프로토타입 단계 LLM 백엔드: **Ollama Cloud** (`.env`의 `OLLAMA_API_KEY` 사용)
  - 로컬 Ollama / 최종 온디바이스와 코드가 거의 같아 배포 전환이 쉽다.
- 최종 배포 목표: **온디바이스 LLM** (Llama 3 / EXAONE 계열, 인터넷·토큰 비용 없음)
- LangChain 버전: **1.x** (인터넷의 0.x 예제와 API가 다름에 주의)

---

## 다음에 정할 것 (TODO)

- [x] 출력/입력 계약, intent 종류, 시나리오 대응 방식 확정
- [x] `src/` 폴더 구조 확정
- [x] Ollama Cloud 연결 + 구조화 출력 + 멀티턴 구현·검증
- [x] emergency filter 긴급어 목록 확정 (CLAUDE.md 목록 채택)
- [x] emergency_filter.py + main.py 로 전체 파이프라인 연결 (③)
- [x] TTS(supertonic 한국어) 음성 출력 연결 — src/tts.py, main 통합
- [x] STT(faster-whisper small) 마이크 입력 연결 — src/stt.py, main 통합 (push-to-talk)
- [x] (A-1) question intent에 robot_state 주입 — "몇 층?" 정확히 답 (ROS2 전까지 더미 값)
- [x] (A-2) LLM/네트워크 실패 대비 — graceful fallback (긴급어는 LLM 우회라 영향 없음)
- [ ] (A-3, 선택) confidence 낮으면 되묻기 — navigate 는 이미 확인 단계가 있어 우선순위 낮음
- [x] (B) ROS2 Jazzy 노드 src/ros_node.py — /vica/user_text → VicaIntent → /vica/intent,
      /vica/robot_state 구독, 긴급어 LLM 우회. 토픽 round-trip 검증 완료 (std_msgs/String + JSON)
- [x] (B+) STT/TTS 를 ROS2 노드로 분리 — ros_stt_node, ros_tts_node. 전체 음성 그래프 round-trip 검증
- [x] launch 파일 (launch/vica_voice.launch.py) — LLM+TTS 노드 일괄 실행, 검증 완료
- [x] 커스텀 .msg 정의 — ros2_ws/src/vica_interfaces (VicaIntent.msg, RobotState.msg), colcon 빌드·import 검증
- [x] 노드를 커스텀 msg 로 전환 — ros_node/ros_tts_node 가 vica_interfaces 사용(ros_convert 변환), RobotState 입력·VicaIntent 출력 round-trip 검증
- [x] LLM/STT 백엔드 환경변수화 (PC 클라우드 ↔ Jetson 로컬 전환) + .env.example
- [x] Jetson 이식 가이드 (docs/jetson-setup.md) — ARM64/Ubuntu22.04/로컬 Ollama(gemma4 e2b)
- [x] state machine / robot_state 발행 스텁 노드 (개발 데모용) + launch 통합, 전체 그래프 검증
- [ ] (실기) Jetson 이식 실행, 로봇 팀이 스텁을 실제 노드로 교체

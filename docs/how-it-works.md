# VICA 음성/LLM 파이프라인 — 처음부터 끝까지 동작 설명서

이 문서는 이 프로그램을 **처음 보는 사람**을 위해 쓴 것이다.
어떤 파일이 어디서 실행되고, 어떤 함수가 어떤 순서로 불리는지를 하나씩 따라간다.

---

## 1. 이 프로그램은 무엇을 하는가

사용자가 말로 "윤지영 교수님 사무실로 가줘"라고 하면, 그 말을 알아듣고
로봇이 이해할 수 있는 데이터(JSON)로 바꿔서 로봇에게 **"제안"** 하는 프로그램이다.

```text
사람의 말 ──▶ 글자로 변환(STT) ──▶ 의미 해석(LLM) ──▶ VicaIntent(제안) ──▶ 로봇의 두뇌(state machine)
```

중요한 원칙 하나만 기억하면 된다:

> **이 프로그램은 로봇을 직접 움직이지 않는다.**
> "어디로 가고 싶어하는 것 같아요"라고 제안만 하고,
> 실제로 움직일지 말지는 로봇 팀의 state machine 이 결정한다.
> 긴급 정지("멈춰!")는 LLM 을 거치지 않는 별도의 빠른 길로 처리된다.

---

## 2. 실행 환경: PC 와 Jetson

**코드는 하나다.** 같은 코드가 PC 에서도 Jetson 에서도 돌아가고,
`.env` 라는 설정 파일이 "어떤 장비의 어떤 서비스를 쓸지"만 바꾼다.

| | PC (개발용) | Jetson (로봇 탑재, 현재 주력) |
|---|---|---|
| 용도 | 코드 개발/실험 | 실제 로봇에서 실행 |
| LLM | Ollama **클라우드** (인터넷 필요) | Ollama **로컬** (Jetson 안에서 실행, 인터넷 불필요) |
| STT 속도 | CPU | **CUDA GPU 가속** (직접 빌드한 ctranslate2) |
| TTS 속도 | CPU | **CUDA GPU 가속** (onnxruntime-gpu) |
| ROS2 | 보통 안 씀 (CLI 모드) | Humble 설치됨 (ROS2 모드 가능) |

`.env` 파일 예 (Jetson):

```bash
OLLAMA_HOST=http://localhost:11434   # LLM: 내 컴퓨터 안의 Ollama
VICA_LLM_MODEL=gemma4:e2b            # 사용할 LLM 모델 이름
VICA_STT_MODEL=medium                # whisper 모델 크기
VICA_STT_DEVICE=cuda                 # STT 를 GPU 로
VICA_STT_COMPUTE=float16
```

PC 에서는 `OLLAMA_HOST=https://ollama.com` + API 키로 바꾸면 클라우드 LLM 을 쓴다.
**코드 수정은 필요 없다.**

---

## 3. 실행 방법은 2가지: CLI 모드와 ROS2 모드

같은 부품(모듈)들을 두 가지 방식으로 조립해 쓴다.

- **CLI 모드** (`python -m src.main`): 터미널 하나에서 전부 실행.
  로봇 없이 파이프라인을 테스트할 때 쓴다. PC/Jetson 어디서든 가능.
- **ROS2 모드** (`ros2 launch ...`): 기능별로 쪼갠 여러 "노드"가
  토픽(우체통 같은 것)으로 메시지를 주고받는다. 로봇과 연결할 때 쓰는 실전 형태.

---

## 4. 파일 지도 — 누가 무슨 일을 하나

### 부품 (로직 모듈 — CLI/ROS 양쪽에서 재사용)

| 파일 | 역할 | 핵심 함수/클래스 |
|---|---|---|
| `src/schema.py` | 데이터의 "모양" 정의 | `VicaIntent`, `DestinationData`, `RobotState`, `EmergencyEvent` |
| `src/stt.py` | 말 → 글자 (faster-whisper) | `VicaSTT.listen()`, `VicaSTT.transcribe()` |
| `src/tts.py` | 글자 → 말 (supertonic) | `VicaTTS.speak()` |
| `src/emergency_filter.py` | 긴급어 있는지 검사 | `detect_emergency(text)` |
| `src/langchain_intent_parser.py` | LLM 으로 의미 해석 | `parse_intent(...)` |
| `src/destination_matcher.py` | 목적지 표현 → 목적지 id 확정 | `match_destination(...)` |
| `src/destination_loader.py` | 목적지 목록 읽기 (YAML 또는 API) | `load_destinations()` |
| `src/emergency_monitor.py` | 마이크 상시 감시 (Phase 4) | `EmergencyMonitor.run()` |

### 조립 방법 1: CLI 모드

| 파일 | 역할 |
|---|---|
| `src/main.py` | 위 부품을 순서대로 부르는 하나의 루프 |

### 조립 방법 2: ROS2 모드 (노드들)

| 파일 | 노드가 하는 일 | 발행(보냄) | 구독(받음) |
|---|---|---|---|
| `src/ros_stt_node.py` | 마이크 녹음 → 글자 | `/vica/user_text` | - |
| `src/ros_node.py` | 글자 → 의미 해석(LLM) | `/vica/intent` | `/vica/user_text`, `/vica/robot_state` |
| `src/ros_tts_node.py` | 답변을 소리로 재생 | - | `/vica/intent` |
| `src/ros_emergency_node.py` | 마이크 상시 감시(긴급어) | `/vica/emergency` | - |
| `src/ros_robot_state_stub.py` | (가짜) 로봇 상태 발행 — 로봇 팀이 교체 | `/vica/robot_state` | - |
| `src/ros_state_machine_stub.py` | (가짜) 이동 판단 — 로봇 팀이 교체 | - | `/vica/intent`, `/vica/emergency` |

### 부가 시스템

| 파일 | 역할 |
|---|---|
| `backend/app.py`, `backend/db.py` | 목적지 관리 API 서버 (FastAPI + SQLite, 선택 사항) |
| `config/destinations.yaml` | 목적지 원본 데이터 (이름, 별칭, 좌표 등) |
| `ros2_ws/src/vica_interfaces/` | ROS2 커스텀 메시지 정의 (.msg 파일) |

---

## 5. CLI 모드: 한 문장이 처리되는 전체 과정

`python -m src.main` 을 실행하면 `src/main.py` 의 `run()` 함수가 시작된다.

### 준비 단계 (시작할 때 1번만)

```text
run() 시작
 ├─ load_destinations()      config/destinations.yaml 을 읽어 목적지 5개를 메모리에 올림
 │                           (.env 에 VICA_DEST_API 가 있으면 FastAPI 백엔드에서 대신 읽음)
 ├─ VicaTTS() 생성           TTS 모델 로드 (Jetson 은 GPU)
 ├─ VicaSTT() 생성           whisper 모델 로드 (VICA_STT=1 일 때만, Jetson 은 GPU)
 └─ history = []             대화 기억용 빈 목록
```

### 반복 단계 (말할 때마다)

사용자가 "윤지영 교수님 사무실로 가줘"라고 말했다고 하자.

```text
1. 입력 받기
   stt.listen()                       엔터 → 녹음 → 엔터 → whisper 가 글자로 변환
   → text = "윤지영 교수님 사무실로 가줘"

2. 긴급어 검사 (LLM 보다 먼저!)
   detect_emergency(text)             "멈춰/정지/..." 가 문장에 있는지 규칙으로 검사
   → None (긴급어 없음) → 다음 단계로
   → 만약 "멈춰" 였다면: 여기서 즉시 처리하고 LLM 은 아예 부르지 않는다

3. LLM 의미 해석
   parse_intent(text, destinations, history, robot_state)
    ├─ _build_system_prompt()         LLM 에게 줄 지시문 생성:
    │                                 "너는 안내 로봇 의도 분석기다 + 목적지 목록 + 규칙"
    ├─ _get_structured_llm()          Ollama LLM 준비 (thinking 끔, JSON 형식 강제)
    ├─ structured.invoke(messages)    LLM 호출 (~3.5초). LLM 이 초안을 돌려줌:
    │                                 { intent: "navigate",
    │                                   destination_candidate: "윤지영 교수님 사무실", ... }
    └─ _finalize(draft, destinations) LLM 초안을 "코드"가 검증/완성:
        ├─ match_destination()        "윤지영 교수님 사무실" → 실제 목적지 id 로 확정
        │                             (LLM 이 지어낸 목적지면 여기서 걸러져 clarify 로 강등)
        ├─ 접근 불가 목적지면          → 안내 거절 문구로 교체
        └─ need_confirm = True        "안내해드릴까요?" 확인이 필요하다고 표시
   → VicaIntent 완성:
     { intent: "navigate", matched_destination_id: "engineering_4f_...",
       need_confirm: true, reply: "윤지영 교수님 사무실로 안내해드릴까요?" }

4. 출력
   print(...)                         터미널에 결과 표시
   tts.speak(intent.reply)            "윤지영 교수님 사무실로 안내해드릴까요?" 를 소리로

5. 대화 기억
   history 에 (내 말, 로봇 답) 추가   다음 턴에 "응 맞아"만 말해도 맥락을 이해
```

### 이어서 "응 맞아"라고 하면?

2번(긴급어 없음) 통과 → 3번에서 LLM 이 history 를 보고
"직전에 물어본 그 목적지를 수락했구나"를 알아채 `is_confirmation: true` 를 돌려준다.
`_finalize()` 가 이를 받아 `need_confirm: false` + reply "윤지영 교수님 사무실 안내를 시작합니다."
로 바꾼다. **이 시점부터 로봇이 움직여도 되는 조건이 완성**되는 것이다 (실행은 로봇 팀 몫).

---

## 6. ROS2 모드: 같은 일을 노드들이 나눠서 한다

실행 (Jetson, 터미널 2개):

```bash
# 터미널 1: 서비스 노드 일괄 실행
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
ros2 launch launch/vica_voice.launch.py
#  → LLM 노드 + TTS 노드 + 긴급감시 노드 + 스텁 2개가 한꺼번에 뜬다

# 터미널 2: 마이크 입력 (push-to-talk 라 대화형이라서 따로 띄움)
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
.venv/bin/python -m src.ros_stt_node
```

### "윤지영 교수님 사무실로 가줘" 한 마디의 여행

```text
[터미널 2] ros_stt_node
  엔터 → 녹음 → VicaSTT.listen() → "윤지영 교수님 사무실로 가줘"
  → /vica/user_text 토픽으로 발행 ─────────────────┐
                                                    ▼
[launch] ros_node (LLM 노드)                — CLI 모드 3번과 완전히 같은 함수를 쓴다
  _on_user_text() 가 메시지를 받음
   ├─ detect_emergency()   긴급어 검사 (있으면 LLM 생략하고 emergency 플래그로 발행)
   ├─ parse_intent()       LLM 해석 + 목적지 확정 (5장에서 설명한 그 함수)
   └─ intent_to_msg()      pydantic → ROS 메시지로 변환해
  → /vica/intent 토픽으로 발행 ──────┬──────────────────────┐
                                     ▼                      ▼
[launch] ros_tts_node          [launch] ros_state_machine_stub (로봇 팀이 교체할 자리)
  _on_intent() 가 받아서         _on_intent() 가 받아서
  VicaTTS.speak(reply)           "이동 확정 → Nav2 goal 생성 대상: ..." 로그만 출력
  → 스피커로 재생                 (실제 로봇에서는 여기서 이동을 결정/실행)
```

`/vica/robot_state` 는 반대 방향의 정보다: 로봇(지금은 스텁)이 "나 지금 별빛관 1층이야"를
계속 발행하고, LLM 노드가 받아뒀다가 "지금 몇 층이야?" 같은 질문에 답할 때 쓴다.

### 긴급 경로: "멈춰!" 는 다른 길로 간다

위 흐름과 **완전히 별개로**, `ros_emergency_node` 가 마이크를 항상 듣고 있다:

```text
[launch] ros_emergency_node  (src/emergency_monitor.py 의 EmergencyMonitor 사용)
  0.5초마다 반복:
   ├─ 최근 2초 분량의 소리를 봄 (슬라이딩 창)
   ├─ 음량이 작으면 → 아무것도 안 함 (조용할 땐 STT 도 안 돌림)
   ├─ 음량이 크면 → whisper 로 글자 변환 (~0.5초)
   ├─ detect_emergency() → "멈춰" 발견!
   └─ 쿨다운 2초 (같은 외침을 두 번 감지하지 않게)
  → /vica/emergency 토픽으로 EmergencyEvent 발행 ──▶ state machine / safety supervisor
                                                     (즉시 정지 판단·실행)
```

포인트: 이 길에는 **LLM 이 없다.** 외침부터 이벤트 발행까지 약 1초.
push-to-talk 대화 중이든 로봇 이동 중이든 항상 동작한다.

---

## 7. 목적지는 어디서 오는가 (선택: FastAPI 백엔드)

기본은 `config/destinations.yaml` 파일이다. `load_destinations()` 가 이 파일을 읽는다.

관리자가 목적지를 자주 바꿔야 하면 API 서버를 쓸 수 있다:

```bash
.venv/bin/uvicorn backend.app:app --port 8000     # 서버 실행 (SQLite 에 저장)
# .env 에 추가: VICA_DEST_API=http://localhost:8000
```

이러면 `load_destinations()` 가 YAML 대신 `GET /destinations` API 를 부른다.
서버가 죽어 있으면? 경고만 남기고 **자동으로 YAML 로 폴백**한다 — 로봇은 계속 동작한다.

pose(지도 좌표) 수정 예: `PATCH /destinations/{id}/pose` — calibration 도구가 쓸 API 다.
파이프라인(LLM 쪽)은 절대 목적지를 수정하지 않는다. 조회만 한다.

---

## 8. 자주 헷갈리는 것 정리

**Q. LLM 이 목적지 id 를 정하는 게 아닌가?**
아니다. LLM 은 "윤지영 교수님 사무실" 같은 **표현**까지만 고른다 (`destination_candidate`).
그 표현을 실제 id 로 확정하는 건 파이썬 코드 `match_destination()` 이다.
LLM 이 없는 목적지를 지어내도 코드가 걸러낸다.

**Q. 왜 긴급어 검사를 두 군데서 하나?**
- `ros_node`(LLM 노드) 안의 검사: push-to-talk 로 **말을 걸었을 때** LLM 보다 먼저 거르는 것
- `ros_emergency_node`: 말을 걸지 않아도 (로봇 이동 중에도) **항상** 듣는 것
같은 `detect_emergency()` 함수를 쓰지만 역할이 다르다.

**Q. 로봇은 언제 실제로 움직이나?**
이 저장소의 코드로는 절대 안 움직인다. `/vica/intent` 와 `/vica/emergency` 를 구독하는
로봇 팀의 state machine 이 결정한다. 상세 조건은 `docs/ros2-interface.md` 참고.

**Q. 응답이 느리면 어디를 보나?**
한 턴 ≈ STT 1.5초 + LLM 3.5초 + TTS 0.3초 ≈ 5초 (Jetson, 워밍업 후).
처음 한 번은 모델 로드 때문에 더 걸린다. LLM 은 `keep_alive=-1` 로 메모리에 상주시켜 둔다.

---

## 9. 더 읽을 것

- `docs/ros2-interface.md` — 토픽/메시지 상세 명세 (로봇 팀 전달용)
- `docs/jetson-setup.md` — Jetson 환경 구축 + CUDA 가속 절차
- `docs/design.md` — 설계 배경
- `docs/worklog-*.md` — 날짜별 작업 기록

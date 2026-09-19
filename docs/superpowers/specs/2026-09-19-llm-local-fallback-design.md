# LLM 클라우드→로컬 자동 폴백 설계

작성 2026-09-19. 브랜치 `feat/llm-local-fallback`(음성) · `feat/llm-cloud-probe`(ROS).
상태: 설계 확정, 구현 전. 실기 검증 전까지 dev 에 머지하지 않는다.

## 용어(쉬운 말)

| 문서의 말 | 뜻 |
| --- | --- |
| 자동 전환(절체) | 클라우드가 죽으면 로컬로, 살아나면 클라우드로 스위치를 넘기는 것 |
| 전환 담당 모듈 | 그 스위치 역할을 하는 코드 `src/llm_backend.py` 의 `LlmBackendManager` |
| 왔다 갔다 반복(플래핑) | 인터넷이 붙었다 끊겼다를 반복해 스위치가 계속 넘어가는 상태. 복귀 후 5분 안에 또 끊기면 다음 확인을 더 늦게 해서 막는다 |
| 접속 확인(probe) | 로컬 상태에서 30초마다 "클라우드 살아 있나" 문만 두드리는 것. 답을 시키지 않아 요금 0 |
| 생존 신호(heartbeat) | 클라우드로 잘 돌 때만 1초마다 내는 "이상 없음" 신호. 끊기면 앱 진단에 경고 |
| 미리 올리기(예열·warm) | 로컬 모델을 메모리에 먼저 실어 첫 답이 늦지 않게 하는 것 |
| 백엔드 | 답을 만드는 쪽 — 클라우드 GPT 또는 로컬 gemma |
| 가짜 백엔드·가짜 시계 | 시험용 대역. 진짜 GPT·시간 대신 쓴다 |

## 1. 목적과 범위

평소에는 클라우드 LLM(gpt-5.4-mini)으로 발화를 해석하고, 클라우드가 닿지 않으면
젯슨 안의 로컬 LLM(Ollama `gemma4-e2b-text`)으로 자동 전환해 안내를 이어 간다.
클라우드가 살아나면 **주행이 끝난 뒤에만** 되돌아간다.

범위에 드는 것:

- 음성 저장소: 백엔드 관리자 모듈, 파서·ROS 노드 통합, Ollama 서버 기동, 로컬 모델
  조립 스크립트, 설정·문서, 단위 시험.
- ROS 저장소: 진단 항목 1개(probes.yaml)와 결함 문구 1개(fault_catalog.py).

범위 밖: STT·TTS(이미 온디바이스), 앱 코드(진단은 기존 경로로 자동 표시), GPT
Realtime 도입(별도 스파이크), 멘트 추가(멘트 최소주의 — 대피는 로그로만).

## 2. 확정된 결정

| # | 질문 | 결정 |
| --- | --- | --- |
| 1 | 전환·복귀 판정 | 시작 점검 + 운행 중 실패 감지로 전환, 30초 주기 확인으로 자동 복귀 |
| 2 | 로컬 모델 대기 | 필요할 때 적재. 한 번 올라오면 내리지 않음(대기 0.44 GB) |
| 3 | Ollama 서버 기동 | 음성 launch 가 다른 노드처럼 함께 띄움(sudo 없음) |
| 4 | 전환 알림 | 로그 + 앱 진단 화면. 사용자 멘트 없음 |
| 5 | 복귀 시점 | 클라우드가 살아나도 해당 주행은 로컬로 마친다. 주행이 끝난 뒤에만 복귀 |

5번의 "주행이 끝남"은 도착뿐 아니라 실패·거부·취소·귀가 완료·대기(`state_idle`)를
포함한다(취소 뒤 로컬에 갇히지 않기 위해).

## 3. 현재 상태(사실)

- `src/langchain_intent_parser.py`: 백엔드는 모듈 로드 때 환경변수로 한 번 결정
  (`VICA_LLM_PROVIDER`). 실행 중 전환 불가. 실패 시 `LLM_UNAVAILABLE` 고정 멘트.
  클라우드 timeout 15초·재시도 1회(최대 30초 침묵).
- `src/ros_node.py`: 시작 시 워밍업 1회. `/vica/robot_state`(층·건물·`is_moving`)
  구독. `/vica_goal_event` 는 구독하지 않는다.
- 로컬 모델 `gemma4-e2b-text` 는 2026-09-19 젯슨 실측 8/8 정답, 발화당 3.0초, 첫
  적재 7초, 추론 중 RAM +0.46 GB, GPU 36/36층. 주행 스택은 꺼진 채 잰 값이다.
- Ollama 0.30.6(cuda_jetpack6) 설치됨. systemd 서비스 없음. 서버는 아무도 안 띄운다.
- 진단: `vica_system_monitor` 가 topic 주기·process CPU 프로브로 `/robot/health`,
  `/robot/events` 를 내고 앱이 그대로 표시한다. 결함 문구는 `fault_catalog.py`.

## 4. 구조

비유: 한전(클라우드)과 발전기(로컬) 사이의 자동 자동 전환 스위치. 스위치는 전기가
지나가는 자리, 즉 파서의 LLM 호출 바로 앞에 둔다.

```text
/vica/user_text ─▶ ros_node ─▶ parse_intent ─▶ LlmBackendManager ─┬─▶ ChatOpenAI (cloud)
                     │                              ▲              └─▶ ChatOllama (local)
/vica_goal_event ────┘ (주행 시작/종료)             │
/vica/robot_state ───┘ (is_moving·is_paused)        │ 1 Hz tick: 생존 신호 발행·클라우드 확인
                                                    ▼
                                      /vica/llm_cloud_alive (cloud 상태일 때만 1 Hz)
                                                    ▼
                          vica_system_monitor probe → LLM_CLOUD_OFFLINE → 앱 진단 화면
```

### 4.1 새 모듈 `src/llm_backend.py` — `LlmBackendManager`

ROS 를 모르는 순수 로직. 시계·백엔드·접속 확인 함수를 주입받아 pytest 로 전부 검증한다.

상태: `CLOUD` | `LOCAL`. 부가 플래그: `cloud_ready`(살아났으나 복귀 대기),
`run_active`(주행 중), `is_moving`, `is_paused`, `next_probe_at`, `probe_interval`,
`last_return_at`.

공개 동작:

- `invoke(messages)`: `CLOUD` 면 클라우드 호출. 실패하면 `_switch_to_local(reason)`
  뒤 **같은 messages 를 로컬로 재호출**해 결과를 돌려준다. `LOCAL` 이면 로컬만.
  로컬도 실패하면 예외를 올리고 파서가 `LLM_UNAVAILABLE` 로 받는다.
  폴백 모델이 비어 있으면(`local` 없음) 옛 동작 그대로: 클라우드 실패 = 예외.
- `tick(now)`: `LOCAL` 이고 `now ≥ next_probe_at` 이면 접속 확인 함수를 부른다.
  성공 → `cloud_ready=True`, 실패 → `next_probe_at = now + probe_interval`.
  이어서 `_maybe_return()`.
- `on_goal_event(name)`: 시작 사건(`goal_sent`·`goal_accepted`·`return_home_sent`)
  → `run_active=True`. 종료 사건(`goal_succeeded`·`goal_failed`·`goal_rejected`·
  `goal_canceled`·`return_home_succeeded`·`return_home_failed`·`return_home_canceled`·
  `state_idle`) → `run_active=False` 후 `_maybe_return()`. `goal_paused` 는 유지.
- `on_robot_state(is_moving, is_paused)`: 보조 신호 갱신. 노드 재시작으로
  `run_active` 를 놓친 경우를 막는 안전띠다.
- `heartbeat_enabled` → `state == CLOUD`.
- `warm_local_async()`: 백그라운드 스레드로 로컬 모델을 적재한다(`/api/generate`,
  `keep_alive=-1`, 기존 `scripts/warmup_llm.py` 와 같은 요청). 시작 워밍업이 실패해
  처음부터 `LOCAL` 일 때만 호출한다. 운행 중 대피는 재호출 자체가 적재를 겸한다.

내부 규칙:

- `_maybe_return()`: `LOCAL` ∧ `cloud_ready` ∧ ¬`run_active` ∧ ¬`is_moving` ∧
  ¬`is_paused` 일 때만 `CLOUD` 로. `last_return_at=now`. 확인 간격은 여기서 되돌리지
  않는다 — 다음 `_switch_to_local` 이 왔다 갔다 반복 여부로 정한다(아래).
- `_switch_to_local(reason)`: `now - last_return_at < 300` 이면 왔다 갔다 반복으로 보고
  `probe_interval = min(probe_interval*2, 300)`(30→60→120→240→300), 아니면 30 으로
  되돌린다. `cloud_ready=False`.
- 실패 분류(로그 등급): 연결 오류·타임아웃 → 경고 "연결 오류". 5xx·429 → 경고
  "서버 오류". 401·403 → **오류** "인증 실패(키 확인)". 그 밖의 예외 → 오류
  "요청 오류". 어느 경우든 대피는 같다 — 사용자 관점에선 모두 "클라우드가 안 됨".
- 접속 확인 함수(기본 구현): `GET https://api.openai.com/v1/models` 를 API 키와 함께
  3초 timeout 으로 호출. 200 → 살아남. 401/403 → 살아났으나 인증 실패 → 오류
  로그, 복귀하지 않음. 연결 오류·timeout → 죽음. 토큰을 쓰지 않는다.

### 4.2 파서 통합 `src/langchain_intent_parser.py`

- 모듈 상수 `PROVIDER`·`DEFAULT_MODEL` 로 백엔드를 고르던 자리를 관리자 싱글턴
  `get_backend_manager()` 로 바꾼다. 관리자는 환경변수로 클라우드·로컬 팩토리를
  만든다.
- `parse_intent(..., model=None)`: `model` 을 명시하면 지금처럼 그 모델 하나로
  직접 호출한다(`scripts/bench_models.py` 경로 보존). 명시하지 않으면 관리자를
  거친다.
- 클라우드 `ChatOpenAI`: 폴백이 켜져 있으면 `timeout=VICA_LLM_CLOUD_TIMEOUT`(기본
  6초)·`max_retries=0`. 꺼져 있으면 기존 15초·1회 유지.
- 로컬 `ChatOllama`: `base_url=VICA_LLM_FALLBACK_HOST`, `model=VICA_LLM_FALLBACK_MODEL`,
  `reasoning=False`, `keep_alive=-1`, `temperature=0`, 구조화 출력은 지금과 같은
  `with_structured_output(_IntentDraft)`.
- 실패 시 `LLM_UNAVAILABLE` 반환 로직은 그대로. 로그 문구에 어느 백엔드까지
  실패했는지 적는다.

### 4.3 ROS 노드 `src/ros_node.py`

- 구독 추가: `/vica_goal_event`(String, JSON) → `event` 키를 관리자에 전달.
- `/vica/robot_state` 콜백에서 `is_moving`·`is_paused` 를 관리자에도 전달.
  (`ros_convert`·`schema.RobotState` 에 `is_paused` 를 추가한다 — 현재 빠져 있다.)
- 1 Hz 타이머: `manager.tick()` 호출 → `heartbeat_enabled` 면
  `/vica/llm_cloud_alive`(std_msgs/Bool, True) 발행.

  **생존 신호(heartbeat)란**: 음성 LLM 노드가 "지금 클라우드로 정상 동작 중"이라는
  뜻으로 1초에 한 번 보내는 짧은 메시지다. 내용(True)에는 의미가 없고 **오고
  있는지**만 본다. 로컬로 대피하면 발행을 멈추고, 진단 노드는 이 토픽이 조용해지면
  "클라우드 LLM 오프라인"으로 판정한다. 이렇게 하는 이유는 진단 노드의 기존
  프로브가 토픽 값이 아니라 **주기**만 검사하기 때문이다(값을 읽는 프로브는 없다).
- 시작 워밍업 실패 시 관리자를 `LOCAL` 로 두고 `warm_local_async()`.
- 로그 예:

```text
[LLM] 클라우드 실패(연결 오류) → 로컬(gemma4-e2b-text)로 대피. 같은 발화 재처리
[LLM] 클라우드 살아남(확인 3회째). 주행 끝나면 복귀
[LLM] goal_succeeded → 클라우드 복귀
```

`src/main.py`(키보드 CLI) 는 타이머가 없으므로 발화마다 `tick()` 을 한 번 부른다.

### 4.4 로컬 실행

- `launch/vica_voice.launch.py` 에 `ExecuteProcess(cmd=["ollama","serve"])` 추가.
  환경: `OLLAMA_KEEP_ALIVE=-1`, `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_MAX_LOADED_MODELS=1`.
  포트 11434 가 이미 쓰이면 바인드 실패로 종료되고 기존 서버를 그대로 쓴다
  (launch 는 그 프로세스 종료를 치명으로 보지 않는다).
- `ollama/Modelfile.gemma4-e2b-text`: HF `unsloth/gemma-4-E2B-it-GGUF:Q4_K_M` 의
  가중치 blob 만 `FROM` 으로 쓰고(projector 제외), 템플릿의 모델 턴 앞에 빈 생각
  블록 `<|channel>thought\n<channel|>` 을 넣어 자발적 추론을 막는다. stop
  `<turn|>`·`<|turn>`, temperature 0.
- `scripts/setup_local_llm.sh`: `ollama pull` → manifest 에서 model 레이어 digest
  를 읽어 Modelfile 의 `FROM` 경로를 채움 → `ollama create gemma4-e2b-text`.
  재실행해도 안전하다(이미 있으면 갱신).
- `docs/jetson-setup.md` 4절: 옛 `gemma4:e2b` 안내를 위 스크립트로 교체.

### 4.5 설정 `.env`

```bash
VICA_LLM_FALLBACK_MODEL=gemma4-e2b-text   # 비우면 폴백 꺼짐(지금과 동일 동작)
VICA_LLM_FALLBACK_HOST=http://localhost:11434
VICA_LLM_CLOUD_TIMEOUT=6                  # 초. 폴백 켜졌을 때만 적용, 재시도 없음
VICA_LLM_CLOUD_PROBE_SEC=30               # LOCAL 상태의 클라우드 확인 간격(초)
```

기존 `VICA_LLM_PROVIDER=openai`·`VICA_OPENAI_MODEL`·`OPENAI_API_KEY` 는 그대로.

### 4.6 진단(ROS 저장소)

`config/probes.yaml` topic 프로브 1개:

```yaml
voice_llm_cloud:
  component: voice
  topic: /vica/llm_cloud_alive
  msg_type: std_msgs/msg/Bool
  qos: default
  min_hz: 0.5
  max_hz: 2.0
  fault_code: LLM_CLOUD_OFFLINE
  optional: true    # 경고 등급. 주행과 무관 — 로봇 안 모델로 안내 중
```

`fault_catalog.py` 항목 1개: 컴포넌트 `voice`, 등급 WARN,
상세 "LLM이 클라우드에 닿지 않아 로봇 안의 모델로 안내 중입니다. 답이 2~3초 느려질
수 있습니다.", 조치 "인터넷 연결을 확인해 주세요. 주행이 끝나면 자동으로 클라우드로
돌아갑니다."

새 토픽 계약은 `/vica/llm_cloud_alive` 하나다. 생산자 음성 노드, 소비자 진단 노드.
생존 신호가 있을 때(주기 0.5~2 Hz) 정상, 없을 때 `LLM_CLOUD_OFFLINE` 경고.

## 5. 오류 처리 요약

| 상황 | 결과 |
| --- | --- |
| 클라우드 실패, 로컬 성공 | 사용자는 로컬 답을 받음. 상태 LOCAL. 경고 로그 |
| 클라우드 실패, 로컬도 실패(서버 없음·모델 없음) | `LLM_UNAVAILABLE` 멘트. 상태 LOCAL 유지, 다음 tick 에 확인 |
| 폴백 모델 미설정 | 옛 동작. 클라우드 실패 = `LLM_UNAVAILABLE` |
| Ollama 서버 포트 충돌 | 기존 서버 사용. launch 는 계속 |
| 노드가 주행 중 재시작 | `run_active` 는 False 지만 `is_moving`/`is_paused` 가 복귀를 막음 |
| 인증 실패 | 대피 + 오류 로그. 확인이 401 을 받아도 복귀하지 않음 |

긴급어 경로(`/vica/emergency`)는 LLM 이전 단계라 이 설계의 영향을 받지 않는다.

## 6. 시험

### 6.1 단위 시험 `tests/test_llm_backend.py` (젯슨 `.venv`, pytest, 네트워크·ROS 없음)

가짜 클라우드·로컬·접속 확인 함수·시계로 다음을 고정한다.

1. 클라우드 정상 → 클라우드 결과, 상태 CLOUD, `heartbeat_enabled` True.
2. 클라우드 실패 → 같은 messages 가 로컬에 전달되고 결과 반환, 상태 LOCAL,
   `heartbeat_enabled` False.
3. 클라우드·로컬 모두 실패 → 예외(파서가 `LLM_UNAVAILABLE` 로 받음).
4. 로컬 없음(폴백 꺼짐) → 클라우드 실패가 그대로 예외.
5. LOCAL 에서 30초 전엔 확인 안 함, 30초 뒤 확인, 실패 시 다음 30초.
6. 확인 성공 + `run_active` → 복귀 안 함. 종료 사건 뒤 복귀.
7. 확인 성공 + `is_moving` 또는 `is_paused` → 복귀 안 함.
8. 종료 사건 8종 각각이 복귀를 열고, `goal_paused` 는 열지 않음.
9. 복귀 뒤 300초 안 재실패 → 간격 60→120→240→300(상한). 300초 밖이면 30.
10. 실패 분류: 연결/타임아웃 경고, 401 오류, 기타 오류 — 모두 LOCAL 전환.
11. 확인이 401 → `cloud_ready` False 유지.

### 6.2 ROS 진단 시험

worktree 에서 `vica_system_monitor` 의 기존 `test_probe_config` 실행(PYTHONPATH
덧붙이기). 새 항목의 component·fault_code·qos·hz 범위가 규칙에 맞는지 확인.

### 6.3 젯슨 실기(승인 후, 주행 스택 켠 채)

| 장면 | 합격선 |
| --- | --- |
| 평소 | 클라우드 답변, `ros2 topic hz /vica/llm_cloud_alive` ≈ 1, 앱 진단 초록 |
| 대화 중 네트워크 끊기 | 그 발화 로컬 답 12초 안, 이후 4초 안. 앱에 `LLM_CLOUD_OFFLINE` 경고 |
| 주행 중 네트워크 복구 | 도착 전 계속 로컬. 로그 "복귀 가능" 만 |
| 도착 뒤 | 다음 발화 클라우드. 경고 60초 안 해제 |
| 네트워크 없이 노드 시작 | 처음부터 로컬, 첫 발화 12초 안 |

기록: 발화당 응답 초·`free -m` 가용·스왑. 스택 없이 잰 3.0초·0.46 GB 대비 증가폭.

## 7. 되돌리기와 머지

- 되돌리기: `.env` 의 `VICA_LLM_FALLBACK_MODEL` 을 비운다. 코드 변경 없이 옛 동작.
- 머지: 6.3 통과 뒤 사용자가 시점을 정한다. 음성·ROS 두 저장소를 같은 날 머지한다
  (생존 신호 토픽은 소비자가 없어도 무해하므로 순서는 무관).

## 8. `[미검증]`

- 주행 스택과 동시 실행 시 로컬 응답 시간·스왑(6.3 에서 측정).
- 포트 충돌 시 launch 의 `ollama serve` 종료가 다른 프로세스에 영향을 주지 않는지.
- `/vica_goal_event` 는 VOLATILE 이라 노드 재시작 후 첫 사건까지 `run_active` 를
  모른다 — `is_moving`/`is_paused` 안전띠로 충분한지 실기에서 본다.

## 9. 변경 파일

음성 저장소(`feat/llm-local-fallback`):

- 신규 `src/llm_backend.py`, `tests/test_llm_backend.py`, `ollama/Modelfile.gemma4-e2b-text`,
  `scripts/setup_local_llm.sh`
- 수정 `src/langchain_intent_parser.py`, `src/ros_node.py`, `src/main.py`,
  `src/schema.py`·`src/ros_convert.py`(`is_paused`), `launch/vica_voice.launch.py`,
  `.env.example`, `docs/jetson-setup.md`

ROS 저장소(`feat/llm-cloud-probe`, worktree):

- 수정 `src/vica_system_monitor/config/probes.yaml`,
  `src/vica_system_monitor/vica_system_monitor/fault_catalog.py`

앱: 변경 없음.

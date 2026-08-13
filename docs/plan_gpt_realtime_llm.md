# 계획 보고서: gpt-realtime-2.1-mini 도입 검토

- 작성: 2026-08-11 / TONY0043 (음성·웨이크워드)
- 대상: 팀 전체 (LLM 공급자 변경은 음성 저장소 내부 사안이나, 비용·네트워크 의존은 팀 판단)
- 상태: **검토용. 파이프라인 코드는 바꾸지 않았다.**
- 근거 조사: 2026-08-11 코드 전수 확인 + 웹 조사 (출처는 부록 C)
- 실측 준비: 같은 브랜치의 `scripts/gpt_realtime_probe.py` +
  `docs/handoff-gpt-realtime-test.md` — 7절의 "실측 후 판단"을 젯슨에서 수행하는 도구다

---

## 0. 한 줄

구조를 지키면 도입은 어렵지 않다 — LLM 결합 지점이 **한 파일 15줄**뿐이다. 다만
gpt-realtime-2.1-mini 의 실익은 **음성 입력을 직결하는 2단계**에서 나오며, 지금의
구조화 출력 계약은 그때도 function calling 으로 **그대로 재현**할 수 있다.
음성 출력(로봇 목소리)까지 맡기는 것은 안전 구조와 정면 충돌하므로 하지 않는다.

---

## 1. 배경 — 왜 검토하나

| 문제 | 현재 수치 | 출처 |
| --- | --- | --- |
| LLM 왕복이 느리다 | **~3.5초** | `docs/jetson-handoff.md:116` (단축이 기존 과제로 명시) |
| 지연을 문구로 메우고 있다 | "네." 선응답 | `ros_node.py:138-140` — LLM 대기 중 침묵을 가리는 임시방편 |
| Jetson GPU 경합 | nvblox ↔ whisper STT | `devlog/2026-07-30-gpu-nvblox-stt-contention.md` 실측 |
| 대화 품질 | gemma4:cloud | 소형 모델의 한국어 뉘앙스 한계 |

gpt-realtime-2.1-mini 는 이 넷을 동시에 건드릴 수 있는 후보다. 음성을 직접 받아
이해하므로 STT+LLM 두 왕복이 한 번이 되고, whisper 의 GPU 부담이 클라우드로 내려간다.

---

## 2. 전제 사실

### 2.1 현재 구조 — LLM 은 마지막 관문이다

```text
마이크 → 웨이크워드(비카야) → whisper STT → 텍스트
    → 긴급어 선차단 (LLM 이전, ros_node.py:126)
    → 규칙 단축 (확인 응답 · pause/resume · 취소 되묻기 — 전부 LLM 우회)
    → LLM (구조화 출력 _IntentDraft)          ← 여기까지 와야 호출된다
    → _finalize (코드가 목적지 확정)
    → /vica/intent → Mission Manager 게이트
발화는 supertonic TTS(온디바이스)가 tts_queue 우선순위로 재생
```

LLM 공급자 결합은 `src/langchain_intent_parser.py` **한 파일 세 곳**이다.

| 위치 | 내용 |
| --- | --- |
| `:14` | `from langchain_ollama import ChatOllama` |
| `:29-30` | `OLLAMA_HOST` / `VICA_LLM_MODEL` 환경변수 |
| `:133-149` | `_get_structured_llm()` — Ollama 전용 kwargs 4개 (`base_url`, `reasoning`, `keep_alive`, `client_kwargs`) |

`:149`의 `with_structured_output(_IntentDraft)` 는 LangChain 표준 인터페이스라
공급자를 바꿔도 호출부(`:197`, `:204`)는 무수정이다. **"구조화 출력이라 바꾸기
어렵다"는 걱정과 반대로, 구조화 출력을 표준 인터페이스로 써 둔 것이 교체를 쉽게
만들었다.**

### 2.2 gpt-realtime-2.1-mini 사양 (2026-08 조사)

| 항목 | 값 |
| --- | --- |
| 출시 | 2026-07-07 |
| 연결 | WebRTC / WebSocket / SIP (세션 기반, chat-completions 아님) |
| 입력 | 오디오 + 텍스트 |
| 가격 | 오디오 토큰 입력 $10 / 출력 $20 per 1M — **분당 약 $0.016** |
| 기능 | function calling, reasoning 지원 |

### 2.3 이 검토의 분기점 — Realtime API 에는 Structured Output 이 없다

Realtime API 는 `response_format`(json_schema 강제)을 **지원하지 않는다.**
구조화된 데이터를 받는 채널은 **function calling 하나뿐**이다. 따라서
"지금 구조를 유지한 채 realtime 을 쓸 수 있는가"는 곧
"function calling 으로 지금의 구조화 출력 계약을 재현할 수 있는가"다. 3절이 그 답이다.

---

## 3. Function Calling vs Structured Output

### 3.1 비교

| | Structured Output | Function Calling |
| --- | --- | --- |
| 무엇 | **응답 전체**가 JSON 스키마를 100 % 준수 | 모델이 **도구를 호출**하고 인자가 스키마를 준수 |
| 보장 | 스키마 강제. 항상 구조가 나온다 | 인자는 스키마 준수(strict). **호출할지 말지는 모델 재량** |
| 어울리는 일 | 분류기·추출기 — 매 입력이 반드시 구조가 되어야 할 때 | 대화형 에이전트 — 말하다가 가끔 행동할 때 |
| 실패 모드 | 사실상 없음 | 호출 안 함 / 불필요한 호출. `tool_choice` 강제로 완화 |
| 대화 능력 | 응답이 데이터라 자유 발화 불가 | 자유 발화와 구조를 섞을 수 있음 |
| Chat API | ✅ | ✅ |
| **Realtime API** | **❌ 미지원** | **✅ 유일한 구조화 채널** |

비유하면 — Structured Output 은 **서식이 인쇄된 접수증**이다. 무엇을 쓰든 칸 안에만
쓸 수 있다. Function Calling 은 **비서에게 버튼을 쥐여 주는 것**이다. 언제 누를지는
비서가 정하지만, 누르면 정해진 형식으로만 눌린다.

### 3.2 VICA 에는 무엇이 맞는가

**VICA 의 LLM 은 대화 에이전트가 아니라 분류기다.** 계약이 "모든 발화 → 정확히
하나의 `VicaIntent`"이고(`docs/design.md`), 자유 행동이 아니라 항상 같은 구조를
내야 한다. 이 계약에는 **Structured Output 이 정확히 맞고, 지금 그렇게 돼 있다.**

Realtime 으로 가면 선택지가 없어진다 — function calling 뿐이다. 그러나 **FC 를
SO 처럼 묶어 쓰면 계약이 그대로 재현된다**:

- 도구를 **하나만** 등록: `report_intent` — 인자 스키마 = 지금의 `_IntentDraft` 그대로
  (intent / destination_candidate / is_confirmation / confidence / reply)
- `tool_choice` 로 호출을 **강제** — "호출할지 말지"라는 FC 의 재량을 제거
- 모델의 자유 발화(audio/text 응답)는 **버린다** — `reply` 필드만 쓴다 (현행과 동일)

즉 결론은 "FC vs SO 중 무엇을 고르나"가 아니라 — **"어느 API 를 쓰든 SO 의 규율을
유지한다. Realtime 에서는 그 규율을 FC 로 구현한다"** 이다.

### 3.3 안전 규칙과의 관계 — 도구는 "보고"이지 "제어"가 아니다

`CLAUDE.md` 는 `move_robot`·`send_nav2_goal` 류의 이동 도구를 금지한다.
`report_intent` 는 그 금지에 걸리지 않는다 — 실행 결과가 지금과 동일한
`VicaIntent` **제안** 하나이며, 수락 여부는 변함없이 Mission Manager 게이트가
정한다. 도구 이름도 행위가 아니라 보고로 짓는다 (`report_intent` O,
`navigate_robot` X). 권한 구조는 1 비트도 바뀌지 않는다.

---

## 4. 도입 방안 — 3단계

### 4.1 1단계 — 공급자 추상화 (최소 변경, 즉시 가능)

`_get_structured_llm()` 15줄을 교체한다.

| 변경 | 내용 |
| --- | --- |
| `:14` | `langchain_ollama` → `langchain_openai` import |
| `:29-30` | `OPENAI_API_KEY` / `VICA_LLM_MODEL` 로 환경변수 정리. **환경변수 분기로 Ollama 폴백 유지** (오프라인 대비) |
| `:133-149` | `ChatOpenAI(...)` + Ollama 전용 kwargs 4개 제거. `with_structured_output(_IntentDraft)` **유지** |
| 추가 | `@lru_cache` — 현재 발화마다 클라이언트를 새로 만든다(`:197`). OpenAI SDK 는 매번 TLS 핸드셰이크가 생기므로 캐싱 필수 |
| 의존성 | `langchain-openai` **하나만** 추가 (전이 의존은 전부 설치돼 있음을 확인) |

무수정: `ros_node.py`, `main.py`, `schema.py`, `_finalize`, `_build_system_prompt`,
규칙층 전부, 테스트 전부.

**정직한 지적**: 이 단계에서 쓸 모델은 gpt-realtime-2.1-mini 가 **아니다.** 텍스트만
넣을 거면 chat 계열(예: gpt-5-mini)이 SO 를 정식 지원하고 더 싸다. realtime 모델을
텍스트 전용으로 쓰는 것은 스포츠카를 사서 주차장에만 두는 것이다. 1단계의 목적은
품질 향상 반, **2단계를 위한 배선 정리 반**이다.

### 4.2 2단계 — 음성 입력 직결 (realtime 의 실익, 본 도입)

```text
현행:  발화 오디오 → whisper STT(수 초, GPU) → 텍스트 → LLM(3.5초) → intent
2단계: 발화 오디오 ──────────────→ gpt-realtime-2.1-mini ────────→ intent
                                   (오디오 입력, 출력은 텍스트+function call 만)
```

- 웨이크워드 노드가 지금처럼 마이크를 소유하고, **발화 창의 오디오만** 세션에 밀어
  넣는다 (`input_audio_buffer` 커밋 방식 — 상시 스트리밍이 아니라 발화 단위)
- 출력 modality 는 **텍스트로 제한** — 모델이 말하지 않는다. `report_intent` 호출만 받는다
- TTS 는 supertonic 온디바이스 유지 → `/vica/tts_state`·tts_queue·자가 트리거 검사
  전부 보존
- **긴급어 경로는 그대로 온디바이스**: 모델 B + 로컬 whisper 검증. 이 경로는 어떤
  단계에서도 클라우드에 얹지 않는다 (네트워크가 끊겨도 "멈춰"는 들려야 한다)

| 기대 이득 | 근거 |
| --- | --- |
| 지연: STT+LLM 두 왕복 → 한 왕복 | `listen_stt_sec`+`llm_sec`(~3.5초+α) → 1초 안팎 추정 `[미검증]` |
| "네." 선응답 제거 가능성 | 응답이 빨라지면 임시방편이 필요 없어진다 |
| **Jetson GPU 경합 해소** | 일반 경로 whisper(medium·CUDA)가 클라우드로 — nvblox 와의 경합 실측 문제가 구조적으로 풀린다 |
| 한국어 이해 품질 | 전사 오류를 거치지 않고 음성을 직접 이해 `[미검증 — A/B 필요]` |

신규 복잡도: WebSocket 세션 수명 관리(재접속·타임아웃), 대화 history 를 세션에
주입하는 방식, 발화 단위 커밋 타이밍. **이것이 2단계의 실제 개발 비용이다.**

### 4.3 3단계 — 음성 출력까지 (하지 않는다)

모델이 로봇 목소리까지 내는 full speech-to-speech 는 현 안전 구조와 다섯 곳에서
정면 충돌한다.

| # | 충돌 | 근거 |
| --- | --- | --- |
| 1 | 긴급어 **LLM 이전** 차단 원칙이 무너진다 | `CLAUDE.md`, `ros_node.py:126` |
| 2 | `/vica/tts_state` 자가 트리거 방지 신호의 타이밍 계약이 사라진다 | `ros_tts_node.py:9-15` |
| 3 | tts_queue 우선순위·긴급 선점이 무의미해진다 | `tts_queue.py` |
| 4 | "판정한 노드가 말한다" 계약(Mission Manager 문구) 붕괴 | `request_for_intent` |
| 5 | 고정 문구 자가 트리거 검사(`test_spoken_text`)를 적용할 수 없다 | 모델 발화는 사전 검사 불가 |

재검토 조건: 원격 오디오에 대한 mute 동기화 설계 + 긴급 선점의 대체 수단이 생겼을 때.

---

## 5. 장점과 단점

### 장점

1. **지연** — 사용자 체감 응답(`response_sec`)의 최대 병목 두 개(STT·LLM)가 한 번에 줄어든다
2. **Jetson GPU 경합 해소** — 실측으로 확인된 nvblox↔whisper 문제의 구조적 해법
3. **대화 품질** — 소형 gemma 대비 상위 모델, 음성 직접 이해로 전사 오류 단계 제거
4. **구조 보존** — 규칙층·게이트·TTS 계약을 그대로 두고 LLM 칸만 바꾼다 (15줄이 그 증거)
5. **계측 준비 완료** — `llm_sec`·`response_sec` 이 이미 측정되고 있어 도입 효과를 숫자로 비교할 수 있다

### 단점

1. **상시 네트워크 의존** — 지금도 Ollama Cloud 지만, WebSocket 세션은 지터에 더 민감. 건물 Wi-Fi 품질이 곧 대화 품질이 된다
2. **비용** — 분당 ~$0.016. 발화 창만 보내므로 상시 과금은 아니나, 사용량 기반 반복 비용 신설 (6절)
3. **세션 관리 복잡도** — stateless 한 지금과 달리 재접속·history 동기화 코드가 새로 생긴다
4. **오프라인 시험 축소** — pytest 는 규칙층만 검증 가능(지금도 동일하나, 로컬 Ollama 대안이 있는 지금보다 후퇴)
5. **벤더 종속** — 1단계의 환경변수 폴백으로 완화하되, 2단계 오디오 경로는 OpenAI 전용이 된다
6. **strict 스키마의 Realtime 지원 `[미검증]`** — 커뮤니티 자료 기준 제약이 있었고 2.1 에서 개선됐다는 서술이 있으나 실측 필요

---

## 6. 비용 개산 `[미검증]`

발화 창만 전송하는 구조 기준. 실측 전 개산이다.

| 가정 | 값 |
| --- | --- |
| 안내 1건당 사용자 발화 총량 | ~1분 (호출·목적지·확인·중간 조작 합산) |
| 하루 안내 | 50건 |
| 오디오 사용 | ~50분/일 → **~$0.8/일 → ~$24/월** |

텍스트 출력(intent JSON)은 오디오 대비 무시할 수준이다. 하루 200건 규모로 늘어도
월 $100 안팎 — 판단 근거로는 충분하되 실측으로 갱신한다.

---

## 7. 결론

1. **지금**: 1단계(공급자 추상화 15줄)만 한다. Structured Output 유지, Ollama 폴백
   유지. 위험이 거의 없고 2단계의 전제가 된다.
2. **본 도입**: 2단계에서 gpt-realtime-2.1-mini 를 오디오 입력 + **function calling
   을 SO 처럼 제약**(`report_intent` 단일 도구, `tool_choice` 강제, 출력 텍스트 한정)
   하는 형태로 붙인다. 기존 계약이 전부 보존되고 realtime 의 실익(지연·GPU 경합)이
   이때 나온다.
3. **음성 출력(3단계)은 하지 않는다.** 안전 구조 다섯 곳과 충돌하며, 재검토 조건을
   명시해 두었다.
4. **어떤 단계에서도 긴급어 경로는 온디바이스 불변이다.**
5. 도입 판단은 감이 아니라 숫자로 한다 — 기존 `metrics.py` 의 `llm_sec`·
   `response_sec` 으로 전/후 A/B 를 젯슨에서 실측한 뒤 2단계 진행을 확정한다.

**FC vs SO 질문에 대한 한 줄 답**: VICA 는 분류기라서 **SO 가 체질**이다. Realtime
API 에는 SO 가 없으므로, 갈 경우 **FC 를 SO 처럼 묶어 쓰는 것**(단일 도구·호출 강제)
이 VICA 에 맞는 형태다.

---

## 부록 A — 1단계 변경 파일 목록

| 파일 | 변경 | 성격 |
| --- | --- | --- |
| `src/langchain_intent_parser.py` `:14, :29-30, :133-149` | 공급자 교체 + `@lru_cache` | **필수, 실질 15줄** |
| `requirements.txt` | `langchain-openai` 추가 | 필수 |
| `.env.example` | `OPENAI_API_KEY` 항목 (2026-07-22 에 의도적으로 뺐던 것 — 복원 사유를 주석으로) | 필수 |
| `scripts/bench_models.py:19,36` | `OLLAMA_HOST` 심볼 import 정리 | 부수 |
| `scripts/warmup_llm.py` | 클라우드면 자동 스킵 — 주석만 갱신 | 부수 |
| `docs/jetson-setup.md` | 환경변수 절 갱신 | 부수 |

## 부록 B — 2단계에서 새로 정할 것

| 항목 | 결정 필요 |
| --- | --- |
| 세션 수명 | 상시 유지 vs 발화마다 — 재접속 지연과 비용의 트레이드오프 |
| history 주입 | `ConversationHistory` 를 세션 instructions 로 동기화하는 방식 |
| 폴백 | 세션 단절 시 whisper+텍스트 LLM 경로로 자동 강등할지 |
| strict 스키마 | Realtime 에서 `report_intent` 인자의 스키마 준수 실측 |
| 한국어 품질 | 동일 발화 세트로 whisper+LLM vs realtime A/B |

## 부록 C — 출처 (2026-08-11 조사)

- 모델 페이지: https://developers.openai.com/api/docs/models/gpt-realtime-2.1-mini
- 출시 보도 (2026-07-06): https://www.marktechpost.com/2026/07/06/openai-gpt-realtime-2-1-mini-reasoning-realtime-api/
- 가격: https://cloudprice.net/models/openai-gpt-realtime-2-1-mini
- Realtime 의 SO 부재 (개발자 커뮤니티): https://community.openai.com/t/realtime-api-workaround-for-lack-of-structured-output/998138
- Agents SDK Realtime 가이드 (SO 미지원 명시): https://openai.github.io/openai-agents-python/realtime/guide/

# 인수인계: gpt-realtime-2.1-mini 젯슨 실측

- 작성: 2026-08-11 / TONY0043
- **대상: Jetson 에서 이 시험을 진행할 AI 에이전트** — 이 문서만 읽으면 맥락 없이 시작할 수 있게 썼다
- 브랜치: `test/gpt-realtime-2026-08-11` (`origin/dev` 기준. 결과도 여기에 커밋한다)
- 상태: 탐침 스크립트는 **[미검증]** — 한 번도 실행된 적이 없다. 디버깅부터가 네 일이다

---

## 0. 먼저 읽을 것

1. 이 문서 전체
2. `CLAUDE.md`, `AGENTS.md` — 안전 규칙. 특히 **긴급어는 LLM 이전 차단, LLM 은 로봇을 움직이지 않는다**
3. `scripts/gpt_realtime_probe.py` 머리 주석

시간이 있으면: `docs/jetson-handoff.md` (저장소 전반), 워크스페이스의
`docs/plan_gpt_realtime_llm.md` (이 시험의 근거 보고서 — 아직 미커밋이라 없을 수 있다.
핵심은 아래 1절에 요약했다).

---

## 1. 지금까지의 진행 (2026-08-11 기준)

### 왜 이 시험을 하나

VICA 음성 파이프라인의 LLM(gemma4:cloud, Ollama)을 gpt-realtime-2.1-mini 로
바꾸는 것을 검토 중이다. 동기 세 가지:

| 문제 | 현재 | 근거 |
| --- | --- | --- |
| LLM 왕복 지연 | ~3.5초 | `docs/jetson-handoff.md:116` |
| 지연을 가리는 임시방편 | "네." 선응답 | `ros_node.py` (LLM 대기 중 침묵 방지) |
| Jetson GPU 경합 | nvblox ↔ whisper | 워크스페이스 devlog 2026-07-30 실측 |

### 검토 보고서의 결론 (요약)

1. LLM 결합 지점은 `src/langchain_intent_parser.py` 한 파일 15줄뿐 — 교체 자체는 쉽다
2. **Realtime API 에는 Structured Output 이 없다.** 구조화 채널은 function calling 뿐
3. 그래서 도입 형태는 "FC 를 SO 처럼 묶어 쓰기": **단일 도구 `report_intent`
   (스키마 = 기존 `_IntentDraft`) + `tool_choice` 강제 + 출력은 텍스트만**
4. 음성 **출력**(로봇 목소리)까지 모델에 맡기는 것은 금지 — 긴급어 선차단·
   `/vica/tts_state` 자가 트리거 방지·TTS 우선순위 선점과 정면 충돌
5. 긴급어 경로(모델 B + 로컬 whisper)는 어떤 경우에도 온디바이스 불변
6. **도입 확정은 감이 아니라 실측 후에 한다** ← 이 시험이 그 실측이다

모델 사양 (2026-08 조사): 2026-07-07 출시, 오디오 토큰 입력 $10/출력 $20 per 1M
(분당 약 $0.016), function calling·reasoning 지원, WebSocket/WebRTC/SIP.

### 이 저장소의 다른 브랜치 (혼동 금지)

| 브랜치 | 내용 | 관계 |
| --- | --- | --- |
| `dev` | 정본 | 이 브랜치의 기준점 |
| `test/voice-field-2026-08-10` | 음성 실기 시험 (웨이크워드·취소·mock 터치) | **무관. 건드리지 마라** |
| `test/gpt-realtime-2026-08-11` | **← 지금 여기.** realtime 탐침 | 결과를 여기 커밋 |

---

## 2. 준비

```bash
cd <workspace>/vica-voice-llm
git fetch origin && git checkout test/gpt-realtime-2026-08-11
source .venv/bin/activate
pip install websockets        # venv 안에만. requirements.txt 에 추가 금지
```

`.env` 에 `OPENAI_API_KEY` 가 필요하다. **사용자에게 키를 요청해라** — 네가 만들 수
없다. 받으면 `.env` 에 한 줄 추가한다 (`.gitignore` 에 이미 있어 커밋되지 않지만,
그래도 `git status` 로 확인해라).

ROS·로봇·마이크 노드는 **필요 없다.** 이 시험은 네트워크와 스피커 없는 터미널만으로 된다
(음성 시험용 wav 녹음에만 마이크를 쓴다).

---

## 3. 시험 절차 — 측정 4종

### 3-1. 연결이 되는가 (스크립트 디버깅 포함)

```bash
.venv/bin/python scripts/gpt_realtime_probe.py --text "화장실로 가줘"
```

스크립트는 GA(2025-08+) 이벤트 형식으로 작성됐고 **실행된 적이 없다.** `[오류]` 가
출력되면 그 JSON 이 디버깅의 출발점이다. 스크립트의 `_session_update()` 주석에
구형(beta) 키 대응표를 적어 뒀다. 고쳐서 돌게 만들고, **무엇을 고쳤는지 이 문서
7절에 기록해라.**

기록: WebSocket open 시간, session.updated 까지 시간 (세션 수명 정책 결정의 근거다)

### 3-2. 스키마 준수 — tool_choice 강제가 실제로 동작하는가

같은 명령을 **10회** 반복하고 스크립트 출력의 `[스키마]` 줄을 센다.

| 확인 | 통과 기준 |
| --- | --- |
| function_call 없이 끝난 횟수 (`[경고]` 줄) | 0 / 10 |
| 필수 필드(intent, reply) 누락 | 0 / 10 |
| intent 가 enum 밖의 값 | 0 / 10 |

하나라도 실패하면 **strict 스키마의 Realtime 지원이 불완전하다는 뜻**이고, 도입
시 재시도·검증 층이 추가로 필요해진다. 실패 사례의 JSON 을 그대로 기록해라.

### 3-3. 한국어 이해 — 발화 세트

텍스트로 먼저, 그다음 음성으로 같은 세트를 넣는다.

```bash
# 음성 샘플 녹음 (reSpeaker, 16k mono)
arecord -r 16000 -f S16_LE -c 1 -d 3 sample.wav
.venv/bin/python scripts/gpt_realtime_probe.py --wav sample.wav
```

| # | 발화 | 기대 intent | 기대 destination |
| --- | --- | --- | --- |
| 1 | "화장실로 가줘" | navigate | 화장실 |
| 2 | "배가 아파요" | navigate (간접) | 화장실 |
| 3 | "여기 몇 층이야?" | question | null |
| 4 | "저기로 가줘" | clarify | null |
| 5 | "취소해줘" | cancel | null |
| 6 | "밥 먹을 데 있어?" | navigate 또는 question | 식당 (판정 기록) |

현행(whisper+gemma)과의 비교가 목적이므로, 여유가 있으면 같은 세트를 기존
파이프라인(`.venv/bin/python -m src.main`)에도 넣어 나란히 기록해라.

### 3-4. 지연 — 도입 판단의 핵심 숫자

스크립트가 자동으로 출력한다. 10회 반복해 중앙값을 적는다.

| 항목 | 현행 | 목표 |
| --- | --- | --- |
| 텍스트: `[지연] 전체 응답` | llm_sec ~3.5초 | 유의미하게 짧을 것 |
| 음성: `[지연] 전체 응답` | listen_stt_sec + llm_sec | STT 왕복이 사라진 만큼 짧을 것 |

`[토큰]` 줄의 usage 도 함께 적어라 — 비용 개산(보고서 6절, 월 ~$24)의 검증 자료다.

---

## 4. 하지 말 것

- **`dev` 에 커밋·push 금지.** 결과는 이 브랜치에만
- **파이프라인 코드(src/) 수정 금지.** 이 시험은 탐침 스크립트만 쓴다. 통합은 별도 단계다
- **`requirements.txt` 수정 금지.** websockets 는 venv 에만 설치한다
- **API 키를 코드·문서·커밋에 남기지 마라**
- **긴급어 경로(웨이크워드·emergency_filter)를 건드리지 마라** — 이 시험과 무관하고, 안전 계약이다
- 모델의 음성 출력(audio modality) 실험은 하지 마라 — 검토 보고서에서 금지로 결론난 축이다

---

## 5. 성공 판정

| 결과 | 다음 행동 |
| --- | --- |
| 3-2 전부 통과 + 3-4 지연이 현행보다 명확히 짧음 | 도입 2단계(파이프라인 통합) 설계로 진행할 근거 확보 |
| 3-2 실패 있음 | 실패 사례를 기록하고 멈춘다 — 검증 층 설계가 선행돼야 한다 |
| 연결 자체가 불안정 (재접속 빈발) | 건물 네트워크 실측치를 기록 — 도입 반대 근거가 될 수 있다 |

**어느 쪽이 나와도 시험은 성공이다.** 숫자를 얻는 것이 목적이지, 좋은 숫자를
얻는 것이 목적이 아니다.

## 6. 결과 기록 방법

1. 아래 7절 표를 채운다
2. 이 문서와 (수정했다면) 스크립트를 **이 브랜치에** 커밋한다
3. 커밋 메시지에 실측 환경(네트워크, 시각)을 적는다

## 7. 결과 (실측 후 채울 것)

| 항목 | 값 |
| --- | --- |
| 실측 일시 / 네트워크 | |
| 스크립트 수정 사항 (있으면) | |
| 연결·세션 수립 시간 | |
| 스키마 10회: 미호출/필드누락/enum위반 | / / |
| 발화 세트 6종 판정 | |
| 지연 중앙값 (텍스트 / 음성) | / |
| usage (토큰) 예시 | |
| 총평 | |

# OpenAI Realtime API — VICA 적용 가이드라인 (초안)

작성 2026-09-20. 근거 = 공식 문서(맨 끝 URL) + 09-19 실기 기록
(`docs/worklog-2026-09-19-llm-local-fallback-field.md`) + 현재 코드.
상태: 설계 가이드. 구현 전. 실험 결과(§7)가 나오면 갱신한다.

## 0. 한 줄 결론

Realtime 은 **"소리를 직접 듣는 귀"** 다. 우리 문제(오전사·기각) 중 **귀에서 나는 오류**는
고칠 수 있지만, **뇌(LLM 분류)에서 나는 기각**은 그대로다. 그래서 먼저 whisper 자리만
Realtime 전사로 바꿔 A/B 로 재고(§4 방식 1), 기각의 주범이 LLM 층으로 확인되면 소리→의도
직행(§4 방식 2)으로 간다. 상시 스트리밍(§4 방식 3)은 안전 규칙과 부딪혀 지금은 안 한다.

## 1. 왜 하나 — 09-19 실기가 보여 준 것

| 사례 | 전사 | 결과 | 오류가 난 층 |
| --- | --- | --- | --- |
| 20:34:52 | "아멘." (실제는 다른 말) | 뜻 없음으로 기각 | **귀(STT)** |
| 20:46:36 | "아멘." | 테스트3 안내 **확정으로 통과** — 더 나쁨 | **귀(STT)** |
| 20:43:08 | "오 분." (정확) | LLM 이 unknown → 재청취 창에서 기각 | **뇌(LLM 분류)** |
| 20:46:02 | "한 시간." (정확, 20:34 엔 wait 로 통과) | 이번엔 unknown → 기각 | **뇌(흔들림)** |

기각 3건 중 2건이 뇌였다. Realtime 을 넣어도 뇌를 그대로 두면 이 둘은 남는다. 반대로
"아멘" 두 건은 귀가 원인이고, 두 번째는 오전사가 주행까지 시켰다. **소리를 함께 보관하는
비교 실험(§7)이 먼저**인 이유다.

## 2. Realtime API 요점 (공식 문서 요약)

| 항목 | 내용 |
| --- | --- |
| 모델 | 대화: `gpt-realtime-2.1`, `gpt-realtime-2.1-mini`(구 `gpt-realtime`, `-mini` 동일 가격). 전사 전용: `gpt-live-transcribe`(말하는 중 delta), `gpt-transcribe`(턴 확정 뒤 완성본 + 감지 언어). 문맥 32k·출력 4,096 토큰 |
| 연결 | 서버는 **WebSocket**(브라우저는 WebRTC). 젯슨 노드 = 서버 쪽이라 일반 API 키로 직접 연결. 임시 키(client_secrets)는 브라우저용 |
| 세션 종류 | `session.type`: `realtime`(대화, 도구 호출, 텍스트/오디오 출력) / `transcription`(전사만) |
| 입력 오디오 | `audio/pcm` **24 kHz** 16-bit mono (base64 청크를 `input_audio_buffer.append` 로 보냄). 우리 캡처는 16 kHz → 리샘플 필요 |
| 턴 판정 | `turn_detection`: `server_vad`(threshold·prefix_padding_ms·silence_duration_ms) / `semantic_vad`(eagerness) / **`null` = 우리가 `input_audio_buffer.commit` 으로 끊음** |
| 잡음 억제 | `noise_reduction` `near_field`/`far_field` — 필드 경로는 API 레퍼런스에서 재확인 `[미확인]` |
| 출력 | `output_modalities: ["text"]` 로 **글자만** 받을 수 있다(TTS 는 우리 supertonic 유지). 음성 출력은 고정 목소리 10종 |
| 도구 호출 | `session.tools`(JSON Schema) + `tool_choice`. 결과는 `response.done` 의 `output[].type == "function_call"`, `arguments`(JSON 문자열) |
| 전사 옵션 | `prompt`(상황 설명), `keywords`(고유명사·목적지 이름), `languages`(`["ko"]`), `delay`(minimal~xhigh, 지연/정확 트레이드오프) |
| 이벤트 | 전사: `conversation.item.input_audio_transcription.delta` / `.completed`. VAD: `input_audio_buffer.speech_started` / `speech_stopped` |
| 세션 수명 | 최대 60분 → 50분마다 유휴 시 재접속. 세션당 `instructions` 는 **한 번만** 보낸다 |
| 가격(1M 토큰) | 2.1: 오디오 입력 $32·출력 $64·글자 입력 $4·출력 $24·캐시 오디오 $0.40. mini: $10·$20·$0.60·$2.40·$0.30. 전사 전용 `gpt-4o-transcribe` ≈ $0.006/분, `-mini` ≈ $0.003/분 |
| 오디오 토큰 | 사용자 음성 1분 ≈ 600 토큰(100 ms = 1 토큰), 모델 음성 1분 ≈ 1,200 토큰(커뮤니티 실측·공식 표기 재확인 `[미확인]`) |
| 한도 | Tier 1 기준 200 RPM / 40k TPM. 발화당 ~2.5k 토큰이면 넉넉 |

## 3. 지켜야 할 경계 (GOVERNANCE·AGENTS 그대로)

- **긴급어("멈춰" 등)는 로봇 안 whisper 가 LLM 이전에 잡는다.** Realtime 은 절대 이 경로를
  대신하지 않는다. 네트워크가 느리거나 끊기면 긴급어가 안 들리는 구조는 금지.
- Realtime 출력은 `VicaIntent` **제안**이다. Mission Manager 가 판정하고, `/cmd_vel*`·Nav2
  goal·CAN 은 건드리지 않는다. reset 권한 없음.
- 클라우드가 끊기면 whisper(로컬) + gemma4-e2b-text(로컬 LLM) 경로로 돌아간다(09-19 폴백).
  즉 Realtime 은 **세 번째 클라우드 백엔드**이지 유일한 귀가 아니다.
- 소리가 로봇 밖으로 나간다. 마이크 소리를 클라우드에 보내는 것은 오늘 whisper(온디바이스)와
  다른 성질이므로 팀 합의·공지 문구를 별도로 정한다.
- 새 노드는 만들지 않는다(선승인 규칙). 기존 `ros_wakeword_node`(귀)와 `ros_node`(뇌) 안의
  백엔드 교체로 한다.

## 4. 적용 방식 3가지

### 방식 1 — 귀만 교체: whisper → Realtime 전사 (추천 1순위, A/B 실험용)

```text
"비카야" 호출 → 칩 VAD·DOA 관문 → 발화 클립(16 kHz)          ← 지금과 같음
       → [새] 24 kHz 리샘플 → WebSocket(type: transcription, turn_detection: null)
         input_audio_buffer.append ×N → commit → ...transcription.completed
       → /vica/user_text → LLM 파서(그대로) → VicaIntent
```

- 바꾸는 곳: `src/ros_wakeword_node.py` 의 STT 호출 한 곳. `src/stt.py` 와 같은 인터페이스
  (`transcribe(audio) -> str`) 를 가진 `src/stt_realtime.py` 를 두고 `.env` 로 고른다
  (`VICA_STT_BACKEND=whisper|realtime`). 긴급어 검증 STT 는 whisper 고정.
- 전사 힌트: `keywords` 에 목적지 이름·별칭(지금 whisper `initial_prompt` 에 넣는 것과 같은
  목록), `languages: ["ko"]`, `prompt` 에 "안내 로봇에게 짧게 말하는 시각장애인 사용자".
- 실패·지연 시: 6초 timeout 뒤 whisper 로 재전사(로컬 폴백). 클라우드 상태는 09-19 의
  `LlmBackendManager` 와 같은 규칙(끊기면 로컬, 주행 끝난 뒤 복귀)을 귀에도 적용한다.
- 장점: diff 가 작고 A/B 가 정확하다(같은 클립, 같은 뇌). 단점: 뇌 기각은 그대로.
- 지연 추정: 2초 클립 업로드(~100 KB) + 전사 ≈ 0.5~1.5초. 지금 whisper turbo 0.07초(RTFx 75)
  보다 느릴 수 있다 — 실측 필요.

### 방식 2 — 소리→의도 직행: Realtime 대화 세션 + 도구 호출 (기각이 뇌 탓이면 2순위)

```text
발화 클립 → WebSocket(type: realtime, output_modalities: ["text"])
   session.instructions = 지금 시스템 프롬프트(목적지 목록 포함) — 세션당 1회
   session.tools = [set_intent(intent, destination_candidate, is_confirmation, reply, ...)]
   append → commit → response.create(tool_choice: set_intent 강제)
   → response.done.output[0].arguments(JSON) → _IntentDraft → _finalize()(그대로)
```

- 바꾸는 곳: `src/langchain_intent_parser.py` 의 백엔드 하나 추가(관리자 `LlmBackendManager` 에
  세 번째 백엔드). 단, 입력이 글자가 아니라 **소리**라서 관리자 인터페이스가 `messages` 대신
  `(audio, history)` 를 받아야 한다 — 이 부분이 설계 변경.
- 함께 얻는 것: `input_audio_transcription` 을 켜면 전사도 같이 와서 로그·재청취 판정에
  쓸 수 있다. 1.9k 토큰짜리 시스템 프롬프트를 발화마다 안 보내고 세션당 한 번만 보낸다
  (지금은 매 발화 ~1.9k 글자 토큰).
- 멀티턴: 우리 `ConversationHistory`(40초 창, 끊기면 비움) 와 Realtime 세션 기억이 겹친다.
  단순하게 가려면 발화마다 새 `response` 만 만들고, 우리가 정한 "맥락 비움" 시점에
  세션을 새로 연다(`conversation.item.delete` 보다 명확).
- 장점: 짧은 답("오 분")을 소리째 이해할 여지, 오전사 단계 자체가 사라짐. 단점: 세션
  관리·재접속·비용이 늘고, 여전히 LLM 이 고르므로 오분류가 0 이 되진 않는다.

### 방식 3 — 상시 스트리밍(서버 VAD 가 우리 호출·말끝 판정을 대체) — 지금은 안 함

마이크를 계속 보내고 `server_vad`/`semantic_vad` 로 턴을 자르는 방식. 긴급어 경로가
클라우드로 나가고, DOA 관문·TTS 중 뮤트·barge-in 설계가 전부 다시 필요하며, 소리가 상시
외부로 나간다. 비용은 mini 기준 ≈ $0.36/시간으로 크진 않지만 안전·설계 부담이 크다.
귀 계층 재설계(8/20 합의) 때 다시 본다.

## 5. 구현 뼈대 (방식 1·2 공통)

- 모듈 `src/realtime_client.py`: WebSocket 연결·세션 유지·재접속(50분 유휴 시)·이벤트
  대기(timeout)·리샘플(16→24 kHz, `numpy` 선형 보간이면 충분)·base64 청크(100 ms 단위).
  `openai` 파이썬 SDK(설치본 3.1.0)의 realtime 연결 도우미 사용 가능 여부 확인 `[미확인]`.
- 노드 시작 때 미리 연결(워밍업)·keep-alive. 끊기면 로컬로 자동 전환, 살아나면 주행 끝난
  뒤 복귀(09-19 규칙 재사용).
- 설정(`.env`): `VICA_STT_BACKEND`, `VICA_REALTIME_MODEL`(`gpt-realtime-2.1-mini`),
  `VICA_REALTIME_TRANSCRIBE_MODEL`(`gpt-transcribe`), `VICA_REALTIME_TIMEOUT`(6),
  `VICA_REALTIME_KEYWORDS`(목적지 별칭 자동 생성이면 불필요).
- 로그: 백엔드 이름·전사·지연을 웨이크워드 계측 줄에 같이 남긴다(지금 "STT 1.83s" 자리).
- 시험: 네트워크·키 없는 단위 시험(가짜 WebSocket)으로 리샘플·청크·timeout·폴백을 고정.
  실제 API 호출 시험은 자동화하지 않는다(저장소 시험 규칙).

## 6. 비용·지연 추정 (발화 2초, 하루 300발화 가정)

| 경로 | 발화당 | 하루 | 비고 |
| --- | --- | --- | --- |
| 지금(whisper 로컬 + gpt-5.4-mini) | 글자 1.9k 입력 ≈ $0.0015 | ≈ $0.45 | 프롬프트가 매번 감 |
| 방식 1 (gpt-transcribe) | ≈ $0.0002 + LLM $0.0015 | ≈ $0.5 | 전사 $0.006/분 |
| 방식 2 (realtime-2.1-mini) | 오디오 20토큰 $0.0002 + 글자 출력 ~60토큰 $0.00015 + 세션당 프롬프트 1회 | ≈ $0.2 | 세션 재접속마다 프롬프트 재전송 |
| 방식 2 (realtime-2.1) | 위의 약 3배 | ≈ $0.6 | |

지연은 실측 전이다. 방식 1 은 whisper(0.07초)보다 느려질 가능성이 크고, 방식 2 는 "전사
1.8초 + LLM 0.9초" 두 단을 한 왕복으로 합치므로 체감 4.3초(8/28 실측 릴레이)가 줄 수 있다.

방식 2 구현은 `conversation:"none"` 이라 매 호출 instructions 를 보내므로(대화 세션을 쓰지
않는다) 위 "세션당 프롬프트 1회" 전제가 성립하지 않는다 — 이 표의 방식 2 비용은 실측
(usage 로그)으로 대체한다.

## 7. 실험 계획 (09-20 합의) — 결정 규칙까지

1. 로봇 마이크로 발화 30개 녹음(짧은 답·숫자·목적지 이름·오늘 실패한 "아멘"류 포함) +
   정답 기록. 소리 파일이 자산이다.
2. 같은 소리를 넣는다: (a) whisper turbo(현재) (b) Qwen3-ASR 0.6B/1.7B (c) Realtime 전사
   `gpt-transcribe`·`gpt-live-transcribe` (d) Realtime 소리→의도 직행(방식 2 시제품, 도구 1개).
3. 층별로 센다: 오전사(CER·오인식 건수) / LLM 오분류 / 우리 관문 기각(재청취·환각 필터).
4. 결정: (c) 가 (a) 보다 오전사가 뚜렷이 적으면 방식 1 채택. 기각의 주범이 LLM 층이면
   방식 2 로 간다. (b) 가 (a) 를 이기면 Realtime 없이 로컬 교체가 답이다.

## 8. 위험·미확인

- `[미확인]` `noise_reduction` 필드 경로와 값, `gpt-4o-transcribe` 를 전사 세션에서 계속 쓸 수
  있는지, 오디오 토큰 환산(600 토큰/분)의 공식 표기, `keywords` 가 대화 세션의
  `input_audio_transcription` 에도 있는지, 한국어 품질(`gpt-live-transcribe` 대 `gpt-transcribe`).
- 24 kHz 리샘플과 reSpeaker 처리(AGC·잡음 억제)가 클라우드 전사에 어떻게 작용하는지.
- 60분 세션 한도·재접속 중 발화가 오면? → 재접속 동안은 whisper 로 받는다.
- Tier 1 TPM 40k: 방식 2 에서 세션 재접속마다 프롬프트 1.9k 가 나가도 문제없으나,
  동시 세션을 여러 개 열지 않는다.
- 와이파이 끊김 실기는 Claude 세션도 끊는다(09-19 교훈). 실험은 녹음 파일로 책상에서 한다.

## 9. 참고 (공식)

- Realtime 안내: https://developers.openai.com/api/docs/guides/realtime
- 전사 전용 세션: https://developers.openai.com/api/docs/guides/realtime-transcription
- 턴 판정(VAD): https://developers.openai.com/api/docs/guides/realtime-vad
- 대화·도구 호출·이벤트: https://developers.openai.com/api/docs/guides/realtime-conversations
- 모델: https://developers.openai.com/api/docs/models/gpt-realtime ,
  https://developers.openai.com/api/docs/models/gpt-realtime-mini
- 가격: https://developers.openai.com/api/docs/pricing

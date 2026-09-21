# 2026-09-21 입 하나·로봇 대장·일지 설계 (Realtime 전결 2단계)

## 0. 배경과 결정

09-20 실기(모델 전결, 6회차 150발화)에서 의도 정확도는 잡혔고 남은 결함은 대화 흐름이었다:
말이 겹치고(두 입), 길고, 끼어들 수 없다. 09-21 사용자 결정으로 다음을 확정했다.

| 결정 | 내용 |
| --- | --- |
| 접근 | **A. LLM 노드가 유일한 입.** 미션은 사건(쪽지)만 내고 문장은 LLM 노드가 만든다 |
| 안내 문구 | 출발·거리·도착 같은 내레이션도 **전부 모델이 말한다** |
| 모델 부재 시 | **틀 문장(지금의 고정 문구)으로 대체.** 로컬 모델에게 문장을 짓게 하지 않는다 |
| 층 정본 | **지도마다 건물·층 한 줄**(지도 = 한 층). 방별 층은 "OO는 몇 층" 답에 쓴다 |
| 대장 소유 | 미션(현장 직원). LLM 노드는 읽기만 |
| 과거 조회 | 사건 일지를 파일로 쌓고, 모델이 **도구 호출**로 필요할 때만 찾는다 |
| 미룬 것 | 모델이 스스로 메모하는 도구(5안)는 이후 실험 항목 |

바뀌지 않는 것: 긴급어는 웨이크워드 노드의 로컬 whisper 가 잡아 안전 노드로 직행한다. 미션·Safety
경로를 우회하지 않는다. 의도 → 미션 게이트 → 주행의 순서, 미션의 상태 기계와 시계는 미션 소유
그대로다. 텍스트 모드(`VICA_INTENT_INPUT=text`)와 로컬 폴백은 유지된다.

참고한 상용 원칙(09-19 조사, Alexa ASK·Google CD·LiveKit): ① 응답은 한 묶음(말할 것 + 재요청 +
마이크 표시) ② 새 턴이 옛 출력을 지운다 ③ 무응답 사다리엔 상한 ④ 우선순위 채널 ⑤ 느린 처리엔
대체 발화 ⑥ 끼어들기의 취소 범위 명시.

관련 문서: `docs/worklog-2026-09-20-realtime-intent-field.md`(실기), `docs/superpowers/plans/2026-09-20-realtime-intent.md`(1단계 계획), 루트 `devlog/2026-09-20-Realtime-소리-의도-직행-실기.md`.

## 1. 사건 계약 — 미션 → LLM 노드

### 1.1 토픽과 쪽지 모양

미션은 말 대신 `/vica/speak_event`(`std_msgs/String`, JSON)를 낸다.

| 칸 | 형 | 뜻 | 예 |
| --- | --- | --- | --- |
| `id` | str | 사건 번호. 미션이 매기고 TTS 를 거쳐 되돌아온다 | `"e-0193"` |
| `kind` | str | 사건 종류(1.2의 고정 어휘) | `"arrived_ask"` |
| `template` | str | 틀 문장. 지금의 고정 문구 그대로(사실 채움) | `"407호 앞에 도착했습니다. 여기서 대기할까요?"` |
| `reprompt` | str | 무응답 때 다시 물을 틀 문장. 질문이 아니면 빈 문자열 | `"여기서 대기할까요?"` |
| `expects_reply` | bool | 답을 기다리는 말인가 | `true` |
| `priority` | str | TTS 큐 우선순위 그대로: `emergency` / `narration` / `response` | `"response"` |
| `budget_ms` | int | 모델 문장을 기다릴 시간(기본 1500) | `1500` |
| `facts` | dict | 문장에 반드시 들어갈 사실 | `{"place": "407호", "question": "wait"}` |

한 사건 = 한 마디다. 도착 안내와 도착 질문처럼 지금 미션이 한 발화로 합쳐 내는 것은 사건 하나
(`arrived_ask`)로 낸다.

### 1.2 사건이 되는 문구와 고정으로 남는 문구

**사건이 되는 것(미션 `MSG_*` → kind).** 안내·대화 문구 전부.

| kind | 지금 문구 | expects_reply |
| --- | --- | --- |
| `depart` | MSG_START ("{name}로 안내를 시작합니다.") | no |
| `distance` | MSG_DISTANCE_REMAINING | no |
| `arrived_ask` | 목적지 arrival_message + MSG_ASK_RESTROOM / MSG_ASK_ENTRANCE / MSG_ASK_GENERIC (`facts.question` = wait / finish / wait_time) | yes |
| `ask_wait_time` | MSG_ASK_WAIT_TIME | yes |
| `wait_confirmed` | MSG_WAIT_CONFIRM / MSG_WAIT_DEFAULT (`facts.minutes`) | no |
| `finish` | MSG_FINISH | no |
| `arrival_retry` | MSG_ARRIVAL_RETRY | yes |
| `leaving_notice` | MSG_LEAVING_NOTICE | no |
| `canceled` / `cancel_confirm` / `cancel_kept` | MSG_CANCELED / MSG_CANCEL_CONFIRM / MSG_CANCEL_KEPT | no / yes / no |
| `not_navigating` / `not_paused` | MSG_NOT_NAVIGATING / MSG_NOT_PAUSED | no |
| `paused` / `resumed` | MSG_PAUSED / MSG_RESUMED | no |
| `approach_question` | MSG_APPROACH_QUESTION | yes |
| `approach_accepted` / `approach_declined` / `approach_no_answer` | MSG_APPROACH_* | no |
| `onboarding` | MSG_APPROACH_ONBOARDING | yes |
| `busy` | MSG_APPROACH_BUSY ("지금은 다른 응대 중입니다…") | no |
| `cancel_first` | MSG_BUSY ("지금 이동 중입니다. 먼저 현재 안내를 취소해 주세요.") | no |
| `dest_retry` | MSG_DEST_RETRY | yes |
| `handle_hint` | MSG_HANDLE_HINT | no |
| `confirm_timeout` | MSG_CONFIRM_TIMEOUT | no |

**고정으로 남는 것(미션이 지금처럼 `/vica/tts_request` 로 직접).** 안전·거절 문구 9개: MSG_ESTOPPED,
MSG_ESTOP_RELEASED, MSG_NAV_FAILED, MSG_NAV_NOT_READY, MSG_ESTOP_REJECT, MSG_UNKNOWN_DEST,
MSG_PRIVATE_DEST, MSG_NOT_APPROACHABLE, MSG_POSE_INVALID. 앱 배달 모드·앱 주행의 문구(MSG_START_DELIVERY,
앱 주행 도착 멘트)도 고정 — 사용자 대화가 없는 주행이다.

### 1.3 번호가 돌아오는 길

- TTS 요청 형식을 `우선순위#번호:문장` 으로 확장한다. 번호가 없으면 옛 `우선순위:문장` 그대로 —
  `tts_queue.parse_request` 가 둘 다 읽는다.
- TTS 노드는 한 문장을 끝까지 재생하면 지금처럼 `/vica/tts_done`(문장)을 내고, 번호가 있었으면
  **추가로** `/vica/tts_done_id`(`std_msgs/String`, 번호)를 낸다. 끊긴 재생(barge-in·stop)은 둘 다 내지
  않는다(지금 규칙 그대로).
- 미션은 `/vica/tts_done_id` 로 자기 질문의 재생 완료를 알고 시계를 켠다. 지금의 문장 대조 4곳
  (접근 질문·손잡이 힌트·온보딩/되묻기·도착 질문 `_on_tts_done`)을 번호 대조로 바꾼다.
- 재청취 창은 LLM 노드가 연다: `expects_reply` 인 사건을 읽기로 정한 순간 `/vica/listen_request`(true)를
  낸다(웨이크워드 노드가 TTS 종료 직후 창을 여는 기존 기제). 미션은 `listen_request` 를 더 내지 않는다.

### 1.4 미션의 3초 보험

미션은 쪽지를 낸 뒤 LLM 노드의 접수 신호 `/vica/speak_ack`(번호)를 기다린다. **3초 안에 없으면**
LLM 노드가 없는 것으로 보고 같은 번호로 틀 문장을 직접 `/vica/tts_request` 에 낸다(번호가 붙어
있으니 `tts_done_id` 로 시계는 정상). 그 뒤 늦게 오는 `speak_ack` 는 무시한다. LLM 노드 쪽 약속:
받은 지 3초 안에 접수하지 못한 쪽지는 미션이 이미 읽은 것으로 보고 버린다(접수한 쪽지는 2.4 의
5초 규칙을 따른다) — 두 입이 같은 번호를 읽는 일은 없다. 09-20 18:55 노드 사망 사건의 재발 방지다.

## 2. 말 창구 — LLM 노드가 문장을 만들고 줄 세우는 법

### 2.1 두 종류의 말

1. **사용자 말에 대한 답**: 소리 → `set_intent` → 의도·대사. 09-20 구현 그대로.
2. **쪽지에 대한 말**: 소리 없이 모델에게 "이 일이 생겼다, 한마디로 말하라". 같은 지시문·대화 기록·
   대장을 보고 만들므로 말투가 1과 같다.

둘 다 하나의 **말 창구(SpeechDesk)** 를 거친다.

### 2.2 쪽지 문장 만들기

- Realtime out-of-band 응답 하나: 지시문 = 기존 소리 모드 지시문 + 맨 뒤 `[쪽지]` 블록(kind·facts·
  template·reprompt). 입력 항목 = 대화 기록(글자)만, 소리 항목 없음. 도구 = `compose_line`
  (`{"text": str}`), tool_choice 강제.
- 예산 `budget_ms`(기본 1500). 안에 오면 그 문장, 못 오면 틀 문장. **둘 중 하나만** 읽고 늦은 모델
  문장은 폐기한다.
- 검사: 60자 초과, `facts` 의 장소 이름 누락, 물음표 없음(expects_reply 인데) → 틀 문장.
- Realtime 이 직전 30초 안에 실패했으면(기존 `recently_failed`) 모델을 부르지 않고 바로 틀 문장.
  텍스트 모드에서는 항상 틀 문장(모델 미사용) — 지금 동작과 같다.
- "생각 중" 신호(`/vica/thinking`)는 쪽지에는 내지 않는다.

### 2.3 미리 짓기(pre-composition)

늦으면 안 되는 말은 사건이 오기 전에 만들어 주머니에 둔다.

| 사건 | 언제 짓나 | 유효 조건 |
| --- | --- | --- |
| `arrived_ask` | `depart` 쪽지를 처리한 직후, 같은 목적지·질문 유형으로 | 30분 이내, `facts.place` 일치 |
| `approach_question` | 노드 기동 시 3개 변형, 순환 사용 | 항상 |
| `onboarding` | 노드 기동 시 2개 변형 | 항상 |

주머니 문장이 유효하면 예산을 기다리지 않고 0초에 읽는다. 목적지가 바뀌었으면 버리고 2.2 로 간다.
기동 시 지은 인사 변형과 출발 때 지은 도착 문장은 TTS 노드에 `/vica/tts_prewarm`(`std_msgs/String`,
문장)으로 보내 **재생 없이 합성만** 해 두게 한다 — TTS 노드에는 이미 합성 캐시(`SynthCache`)와 기동 시
워밍업 경로가 있으므로 구독 하나를 더하는 일이다. 실제 재생 때 캐시에 맞으면 합성 지연(0.5~1 s)도 없다.

### 2.4 줄 세우기 규칙

1. **한 번에 한 마디, 대기 한 줄.** 재생 중 문장 하나 + 대기 문장 하나. 세 번째가 오면 대기 중이던
   것을 버리고 새 것으로 바꾼다.
2. **덮어쓰기 표.** `busy`·`not_navigating`·`cancel_first`·`confirm_timeout` 은 대기 중인 제안·확인
   질문(사용자 답 경로 포함)을 지운다. `arrived_ask` 는 대기 중인 `distance` 를 지운다. `canceled`·
   `finish`·`leaving_notice`·긴급은 그 안내에 딸린 대기 문장 전부를 지운다. 새 확인 질문은 옛 확인
   질문을 지운다.
3. **5초 넘게 대기한 문장은 버린다.** `expects_reply` 쪽지는 버리지 않고 `/vica/speak_dropped`(번호)로
   미션에 반납한다 — 미션은 그 질문을 다시 낼지 상태로 결정한다.
4. **모델 문장 vs 틀 문장은 둘 중 하나.**
5. **한 사건 = 한 마디, 60자.**
6. **원천 차단.** 대장의 대화 단계(3절)를 지시문에 넣어, 미션이 거절할 제안(회전 중·안내 중·인사 답
   대기 중의 navigate)은 모델이 애초에 내지 않게 한다. 규칙 2는 그물이다.
7. **끼어들기(barge-in) 시** 대기 문장과 진행 중인 `compose_line` 요청을 함께 취소한다. 대화 기록에는
   실제로 끝까지 재생된 문장만 남는다(`tts_done` 기제 그대로).

TTS 노드의 큐(최대 8·2초 중복 제거)는 그대로 두되, 창구가 위에서 정리하므로 실제로는 2개 이상 쌓이지
않는다.

### 2.5 대화 기록

실제로 재생된 문장은 `/vica/tts_done` 으로 돌아와 AI 줄로 기록된다(09-20 구현). 모델 문장이든
틀 문장이든 읽힌 쪽만 남는다. 기록 초기화 규칙(대기 중 보존, `return_home_sent` 에만 비움)은 그대로.

## 3. 로봇 대장 — 미션이 적고 방송한다

### 3.1 무엇을 적나

| 줄 | 값 | 출처 |
| --- | --- | --- |
| 건물·층 | `로봇관`, `4` | 지도 폴더의 `map.yaml`(`building`, `floor`) — 지도 = 한 층 |
| 지금 있는 곳 | `407호 앞` / `407호와 화장실 사이` / `위치 미확인` | AMCL 좌표 vs 등록 목적지 좌표. 3.0 m 안이면 "OO 앞", 아니면 가장 가까운 두 곳 "사이". 초기 위치 전(포즈 미수신·공분산 큼)은 미확인 |
| 가는 중인 곳 | 목적지 name | 출발 시 기록, 도착·실패·취소 시 비움 |
| 직전에 간 곳 · 도착 시각 | name, epoch | 도착(`goal_succeeded`) |
| 하려다 만 곳 | name | 확인 대기까지 갔다가 거절·취소·시간초과·무응답으로 안 간 곳. 다음 출발 시 비움 |
| 대화 단계 | `idle` / `awaiting_user` / `confirming` / `seeking` / `navigating` / `asking_next` / `asking_wait_time` / `waiting` / `returning` | 미션 상태 그대로 |
| 대기 | 요청 분, 남은 초 | WAITING 진입 시 |

### 3.2 전달과 보존

- `vica_interfaces/msg/RobotState.msg` 에 칸을 늘린다: `string dialog_state`, `string place_here`,
  `float32 place_here_dist_m`, `string active_destination`, `string last_destination`,
  `int32 last_arrived_age_sec`, `string aborted_destination`, `int32 wait_minutes`, `int32 wait_left_sec`.
  기존 칸(`current_floor`, `current_building`, `is_moving`, `is_paused`)은 유지하고 층·건물은 `map.yaml`
  로 채운다(launch 인자는 있으면 우선).
- 미션은 지금처럼 1 Hz 로 `/vica/robot_state` 를 낸다. 대장이 바뀔 때마다
  `~/vica_data/state/<map_id>/ledger.json` 에 적고, 기동 시 읽어 "직전에 간 곳·하려다 만 곳"을 복원한다.
  "지금 있는 곳"은 복원하지 않는다(초기 위치 뒤 좌표로 다시 계산).
- LLM 노드의 `SituationBoard` 는 `/vica/robot_state` 를 읽어 채우는 `LedgerView` 로 바뀐다. 새 칸이
  비어 오면(옛 미션) 지금의 goal-event 추정을 그대로 쓴다(호환). 지시문 맨 뒤 `[지금 상황]` 블록의
  줄이 3.1 의 일곱 줄이 된다. "답을 기다리는 중" 표시는 유지.

### 3.3 질문 → 답의 출처

"몇 층이야?" → 건물·층 / "지금 어디 있지?" → 지금 있는 곳 / "어디 가려고 했더라?" → 하려다 만 곳
(더 오래된 일은 4절) / "지금 어디 가?" → 가는 중인 곳 / "OO는 몇 층?" → 목적지 목록의 방별 층.

## 4. 사건 일지와 찾아보기 도구

### 4.1 일지

- LLM 노드가 적는다. 파일 `~/vica_data/field/journal-YYYYMMDD.jsonl`, 한 줄 = 한 사건:
  `{"t": "10:43:12", "session": 7, "kind": "depart", "place": "407호"}`. kind 는 1.2 의 사건 종류 +
  사용자 쪽 `wake`(호출), `propose`(제안), `confirm`(확정), `deny`, `wait_request`(분).
- 사실만 적는다. 모델이 지은 문장은 적지 않는다.
- `session` 은 대화 회차. 안내가 대기 없이 끝나 미션이 홈으로 갈 때(`return_home_sent`, 기록 초기화와
  같은 순간) 하나 오른다. 파일 쓰기 실패는 로그만 남기고 계속 간다.

### 4.2 도구

- Realtime 도구 둘: `set_intent`(기존), `recall_journal`
  (`{"scope": "session"|"today", "kinds": [str], "place": str, "limit": int<=10}` → 코드가 일지에서 최근
  순으로 최대 10줄을 돌려준다).
- 흐름(`RealtimeIntentClient.ask`): 1차 응답은 `tool_choice: "required"`. 모델이 `recall_journal` 을
  부르면 코드가 찾아 `function_call_output` 을 붙여 2차 응답을 만들고, 그때는 `set_intent` 를 강제한다.
  **발화당 조회 최대 1회.** 2차도 out-of-band 이며 입력 항목은 1차 것 + 호출·결과 두 항목.
- 지시문 규칙: "`[지금 상황]` 에 답이 있으면 찾지 마라. 과거의 순서·횟수·시간을 물으면 찾아라. 기본은
  지금 회차, 사용자가 '오늘' 전체를 물으면 `today`."
- 로그: `[RT] … tool=recall(session, 4줄)` 로 호출 여부·건수를 남겨 실기에서 과소/과다 호출을 본다.

## 5. 안전·오류 처리

- 긴급어·비상 멈춤·이동 실패 등 9개 고정 문구는 미션이 직접 말한다(1.2). LLM 노드가 죽어도 살아 있다.
- LLM 노드 부재: 1.4 의 3초 보험. 미션이 틀 문장을 같은 번호로 낸다.
- 인터넷 부재·느림: 2.2 의 틀 문장 대체. 텍스트 모드는 항상 틀 문장 = 09-20 이전 동작.
- 모델 문장 이상(60자 초과·사실 누락·질문인데 물음표 없음): 틀 문장.
- 파일(일지·대장) 쓰기 실패: 로그만. 어떤 로그·파일 오류도 노드를 죽이지 않는다(09-20 로거 사건의 교훈,
  `set_logger` 예외 삼킴과 같은 원칙).
- 되돌리기: `.env` `VICA_INTENT_INPUT=text` 면 쪽지는 틀 문장으로만 읽히고 의도는 whisper 경로다. 미션의
  사건 발행은 텍스트 모드에서도 동일하므로 미션을 되돌릴 필요가 없다.

## 6. 시간 상수

| 값 | 재는 구간 | 만료 시 | 근거 |
| --- | --- | --- | --- |
| 1.5 s (`budget_ms`) | 쪽지 접수 → 모델 문장 | 틀 문장 | 모델 응답 중앙값 1.3 s. 도착·인사는 미리 짓기로 0 s |
| 3 s | 쪽지 발행 → `speak_ack` | 미션이 틀 문장 직접 | 예산 1.5 + 합성 0.5~1 뒤에도 무소식 = 노드 부재 |
| 5 s | 창구 대기 → 읽기 시작 | 버림(질문은 반납) | 지난 상황을 말하지 않는다 |
| 6 s (`VICA_REALTIME_TIMEOUT`) | 소리 발송 → 의도 | whisper 경로가 답, 30 s 생략 | 09-20 7~8 s 꼬리 2건 |
| 8 s | 질문 재생 완료 → 답 | 같은 질문 1회 → 예고 → 종료 | Alexa 와 동일. 귀 홀드 중 정지 |
| 3 s (예고) | 떠나기 예고 → 출발 | 귀가 | 기존 값 |
| 30 s | Realtime 실패 → 재시도 | 재시도 | 복도 음영에서 발화마다 6 s 대기 방지 |

숫자는 서로 다른 길에 있어 더해지지 않는다. 정상 흐름에서 사용자가 기다리는 것은 자기 말의 답
(~1.3 s)과 로봇의 다음 말(0~1.5 s)뿐이다.

## 7. 시험과 합격선

- 단위: SpeechDesk 규칙(덮어쓰기·5초·한 줄·둘 중 하나), `parse_request` 옛/새 형식, TTS `tts_done_id`
  발행, 미션 `Say` 30종 → 사건 변환과 고정 9종 유지(각 문구가 사라지지 않았는지), 3초 보험,
  대장 계산(좌표→앞/사이/미확인, 복원), `LedgerView` 호환(새 칸 없을 때), 일지 회차·도구 필터,
  2회전 도구 흐름(가짜 연결).
- 실 API 스모크: 쪽지 3종 문장, 예산 초과 → 틀 문장, 미리 짓기 적중, `recall_journal` 왕복 1회.
- 실기 대본: 09-20 21:11·21:21 겹침 장면 재현 / 도착 즉시 안내 / 10분 대기 뒤 "아까 어디 갔었지?" /
  "우리 몇 층?" / 복도 오프라인 도착 / LLM 노드 강제 종료 뒤 직원 대체 발화 / 질문 중 끼어들기.
- 합격선: 겹침 0회 · 도착 안내 시작 ≤ 0.5 s · 오프라인 안내 누락 0 · 무응답 재질문 1회 뒤 종료 ·
  대장 질문 3종 정답 · 도구 호출은 기억 질문에서만.

## 8. 단계와 저장소

각 단계는 자기 브랜치·시험·실기를 거치고, 통과 뒤 다음으로 간다. 머지는 사용자 결정.

| 단계 | 내용 | 저장소·브랜치 | 크기 |
| --- | --- | --- | --- |
| P1 대장 | RobotState 칸, 미션 대장 계산·파일 보존, `map.yaml`, `LedgerView` | vica_ros2_ws(`vica_interfaces` 포함) `feat/robot-ledger`(feat/arrival-reconfirm 에서 분기), voice `feat/realtime-intent` | 반나절 |
| P2 입 하나 | 사건 토픽·번호 왕복·SpeechDesk·compose_line·미리 짓기·미션 Say→사건·3초 보험·listen_request 이관 | vica_ros2_ws `feat/speak-events`, voice `feat/realtime-intent` | 하루 반 |
| P3 일지·도구 | journal·recall_journal·2회전 흐름·지시문 | voice `feat/realtime-intent` | 반나절 |

P1 을 먼저 두는 이유: 가장 작고, P2 의 원천 차단(대화 단계)이 P1 에서 나오며, 그것만으로 "몇 층·어디·
어디 가려 했나"가 답이 된다.

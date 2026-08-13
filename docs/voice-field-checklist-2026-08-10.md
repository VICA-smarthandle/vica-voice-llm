# 음성 파이프라인 실기 시험 체크리스트 — 2026-08-10

절차와 근거는 `docs/voice-field-test.md`에 있다. 이 문서는 **현장에서 채우는 표**다.

| | |
| --- | --- |
| 시험자 | |
| 시작 / 종료 | : ~ : |
| 장소 | |
| 커밋 (`git rev-parse --short HEAD`) | voice ________ / ros2_ws ________ |

---

## A. 실행 매뉴얼 — 젯슨에서 이 순서대로

절차의 **근거**는 `docs/voice-field-test.md`에, **채울 표**는 아래 0~8절에 있다.
이 절은 **복사해서 붙여 넣는 명령**만 모았다.

> 2026-08-11 첫 기동에서 막힌 지점을 전부 반영했다. A.0 은 그때 새로 드러난
> **설치 단계**이고, A.1 의 `export` 두 줄은 **빠뜨리면 노드가 죽는다.**

`<workspace>` 는 실제 경로로 바꾼다. 이 젯슨에서는 `~/VICA-smarthandle` 이다.

### A.0 설치 확인 (기기당 한 번, 4개 전부 있어야 한다)

이미 되어 있으면 넘어간다. **하나라도 없으면 그 항목의 명령을 실행한다.**

```bash
cd <workspace>/vica-voice-llm

test -f .venv/lib/python3.10/site-packages/openwakeword/resources/models/melspectrogram.onnx \
  && echo "OK  openWakeWord 기본 모델" || echo "없음 → ①"
test -f assets/wake_greeting.wav && echo "OK  인사 음성" || echo "없음 → ②"
.venv/bin/python -c "import ctranslate2 as c; print('OK  STT CUDA' if c.get_cuda_device_count() else '없음 → ③')"
test -f .env && echo "OK  .env" || echo "없음 → ④"
```

**① openWakeWord 기본 모델** — pip 패키지에 **들어 있지 않다.** 없으면 웨이크워드
스레드가 `melspectrogram.onnx ... File doesn't exist` 로 죽고 **2·3단계가 통째로
막힌다.** VICA 자체 모델(`models/*.onnx`)과는 별개다.

```bash
.venv/bin/python -c "from openwakeword.utils import download_models; download_models(model_names=['__vica_none__'])"
```

> 더미 이름을 넘기면 공식 사전학습 모델(hey_jarvis 등)은 받지 않고 항상 받는
> feature(melspectrogram, embedding) + VAD 만 가져온다. 약 6MB.

**② 인사 음성** — 없으면 노드가 TTS 로 대체하는데 **3-2 가 재는 "네?" 까지의
시간이 왜곡된다.**

```bash
.venv/bin/python scripts/make_cue_wavs.py
```

**③ STT CUDA** — PyPI 의 aarch64 `ctranslate2` 는 **CPU 전용**이다. 이 젯슨에는
CUDA 로 빌드한 4.8.1 이 `~/jetson-builds` 에 있고, venv 의 CPU 판과 버전이 같아
교체만 하면 된다(재빌드 불필요). 자세한 절차는 `docs/jetson-handoff.md` §3.

**④ `.env`** — `.gitignore` 에 있어 **git 으로 따라오지 않는다.** 없으면 4단계
(대화)가 통째로 막힌다. 1~3단계는 없어도 된다.

```bash
cp .env.example .env      # mv 가 아니다 — 템플릿은 팀 공용이라 남겨 둔다
# .env 의 OLLAMA_API_KEY 에 실제 키를 채운다
```

### A.1 터미널 배치

터미널 5개를 쓴다. 창 4·5는 시험 항목에 따라 명령을 바꾼다.

```text
창 1  음성 노드 (launch)         ← 로그를 계속 본다
창 2  ros2 topic echo /vica/emergency   ← 긴급어. 2단계 내내 켜 둔다
창 3  ros2 topic echo /vica/wake        ← 호출. 3단계에서 본다
창 4  조작용 (topic pub 등)
창 5  터치센서 mock (5-4에서만)
```

**모든 창에서 먼저** 실행한다. 하나라도 빠뜨리면 메시지 타입을 못 찾는다.

```bash
source /opt/ros/humble/setup.bash
source <workspace>/vica_ros2_ws/install/setup.bash     # ← 빠뜨리면 노드 3개가 죽는다
cd <workspace>/vica-voice-llm
```

**창 1 에서는 추가로** 두 줄을 더 내보낸다.

```bash
export VICA_STT_DEVICE=cuda
export VICA_STT_COMPUTE=float16
```

> **이 두 줄은 선택이 아니다.** 코드 기본값은 `cpu` 이고
> (`src/wakeword_monitor.py:212`), 그러면 compute 가 `int8` 로 정해지는데 이
> CUDA 빌드는 **CPU 에서 float32 만 지원**해서 웨이크워드 스레드가
> `ValueError: Requested int8 compute type ...` 로 죽는다.
> `.env` 에 적어도 소용없다 — 이 노드들은 `.env` 를 읽지 않고 셸 환경변수만 본다.

증상별로 어디를 빠뜨렸는지 바로 알 수 있다.

| 노드 로그 | 빠뜨린 것 |
| --- | --- |
| `ModuleNotFoundError: No module named 'vica_interfaces'` | `install/setup.bash` source |
| `ValueError: Requested int8 compute type` | `export VICA_STT_DEVICE=cuda` |
| `NO_SUCHFILE ... melspectrogram.onnx` | A.0 ① |
| `목적지 catalog가 없어 빈 목록을 사용합니다` | A.4 의 `destinations_yaml:=` |

### A.2 코드 받기 (한 번만)

**`dev`가 아니라 시험 브랜치를 받는다.** 오늘 것은 `dev`에 없다.

```bash
cd <workspace>/vica-voice-llm
git fetch origin && git checkout test/voice-field-2026-08-10

cd <workspace>/vica_ros2_ws
git fetch origin && git checkout test/mock-touch-and-mic-open
colcon build --packages-select vica_interfaces vica_mission_manager vica_user_guidance --symlink-install
source install/setup.bash
```

- [ ] 두 저장소 모두 시험 브랜치인지 확인 (`git branch --show-current`)

빌드가 필요한 패키지가 셋이다.

| 패키지 | 왜 |
| --- | --- |
| `vica_interfaces` | `VicaIntent.msg` 주석에 `cancel_keep` 계약이 늘었다 |
| `vica_mission_manager` | `cancel_keep` 분기가 새로 생겼다 (5-4) |
| `vica_user_guidance` | 새 실행 파일 `mock_touch_keyboard` 가 등록됐다 (5-5) |

### A.3 마이크 확인

reSpeaker 를 **6채널로** 잡아야 한다. ch0(처리음)이 모델 입력이고, 폴백으로 기본
마이크 1채널을 잡으면 학습-실전 분포가 어긋나 3단계 수치가 무의미해진다.

```bash
lsusb | grep 2886                     # ReSpeaker 4 Mic Array 가 보여야 한다
.venv/bin/python -c "
import sounddevice as sd
hit=next((i for i,d in enumerate(sd.query_devices())
          if 'respeaker' in d['name'].lower() and d['max_input_channels']>=6), None)
print('탐지:', f'장치 {hit}, 6ch' if hit is not None else '없음 → 1ch 폴백')"
```

- [ ] `장치 N, 6ch` 로 나온다

> **다른 프로그램이 마이크를 잡고 있으면 목록에서 사라진다.** 노드가 떠 있는
> 상태로 확인하면 `없음` 으로 보이니, **반드시 노드를 내린 뒤** 확인한다.
> 녹음 도구·데모·노드를 동시에 실행하지 않는다.

### A.4 실행

**창 1 — 음성 노드**

```bash
ros2 launch launch/vica_voice.launch.py map_id:=vica_map_0630 \
  destinations_yaml:=$PWD/config/destinations.yaml
```

> `destinations_yaml` 을 붙이는 이유: launch 기본값은
> `$HOME/vica_data/destinations/<map_id>/destinations.yaml` 인데 그 저장소가 아직
> 정의되지 않았다. 빼면 목적지 0개로 떠서 **4단계가 통째로 막힌다.**

시험 2-7에서만 인자를 하나 더 바꿔 다시 띄운다.

```bash
ros2 launch launch/vica_voice.launch.py map_id:=vica_map_0630 \
  destinations_yaml:=$PWD/config/destinations.yaml suppress_during_tts:=false
```

기동 직후 **이 두 줄을 확인하고 넘어간다.**

```text
public 목적지 28개 로드: .../config/destinations.yaml
웨이크워드 상시 감시 시작 (장치 24, 6ch — Ctrl+C 종료)
```

- [ ] 목적지 개수가 0이 아니다
- [ ] `6ch` 다 (`장치 None, 1ch` 면 A.3 으로 돌아간다)

> 두 번째 줄은 `print` 라서 ROS 로그 접두사(`[INFO] [...]`)가 없다. 스크롤에
> 묻히기 쉬우니 찾아서 본다.

**창 2·3 — 감시**

```bash
ros2 topic echo /vica/emergency        # 창 2
ros2 topic echo /vica/wake             # 창 3
```

**창 4 — 조작용.** 자주 쓰는 명령을 미리 적어 둔다.

```bash
# TTS 재생 (2-1, 2-7)
ros2 topic pub -1 /vica/tts_request std_msgs/String \
  "{data: 'narration:지금 이동 중입니다. 먼저 현재 안내를 취소해 주세요.'}"

# 긴급 선점 확인 (2-5)
ros2 topic pub -1 /vica/tts_request std_msgs/String \
  "{data: 'emergency:안전을 위해 멈추겠습니다.'}"

# 재생 신호 (2-6)
ros2 topic echo /vica/tts_state

# intent 확인 (4-3)
ros2 topic echo /vica/intent
```

**창 5 — 터치센서 mock (5-5에서만).** guidance driver 를 mock 모드로 따로 띄운다.

```bash
ros2 launch vica_user_guidance user_guidance.launch.py enable_serial:=false
ros2 run vica_user_guidance mock_touch_keyboard     # 스페이스 = 잡음/놓음
```

### A.5 E-stop 이 걸렸을 때 푸는 법

2-2·2-7·5-2에서 **실제로 걸린다.** 미리 익혀 둔다.

**전제 — 안전 스택이 떠 있어야 한다.** 음성 노드는 `/vica/emergency` 를 발행할
뿐이고, 중앙 래치를 거는 것은 `emergency_stop_node` 다. 아래가 안 떠 있으면
**E-stop 이 걸리지도, 리셋 서비스가 있지도 않다.**

```bash
ros2 launch vica_safety safety_bringup.launch.py    # 별도 창
ros2 service list | grep estop                      # 서비스가 보이는지 확인
```

1. 관리자 앱 → 안전 초기화
2. 앱이 없으면 터미널에서 호출한다

```bash
ros2 service call /app_estop_reset std_srvs/srv/Trigger {}
ros2 topic echo /estop_state          # 풀렸는지 확인 (false 여야 한다)
ros2 topic echo /app_estop_state      # reset_allowed 와 safety_state 를 같이 본다
```

`success: false` 로 거부되면 **원인이 아직 살아 있는 것**이다. 물리 버튼이 눌린
채인지, 긴급어가 계속 잡히는지, 앱에서 건 E-stop 이 남았는지 본다.
`safety_state` 가 `ESTOP_RELEASED_WAIT_RESET` 이 된 뒤에야 승인된다
(`vica_safety/app_emergency_node.py:51`).

> 앱·STT 의 `false` 는 입력 해제일 뿐 **래치 reset 이 아니다.** 원인을 먼저 없애야
> reset 이 승인된다.
>
> 유지보수용 `/safety_reset` 도 같은 검사를 거치지만 호출자 인증이 없는 `[GAP]`
> 이다(`GOVERNANCE.md` §5). 시험에서는 `/app_estop_reset` 을 쓴다.

### A.6 끝낼 때 (필수)

- [ ] `suppress_during_tts` 를 **기본값 true** 로 되돌린다 (그 인자만 빼고 launch)
- [ ] 터치센서 mock 노드를 `q` 로 종료한다 (놓음 상태로 끝난다)
- [ ] 바꾼 값이 있으면 7절 표에 적는다
- [ ] `git status` 로 의도치 않은 변경이 없는지 본다 (두 저장소 각각)

---

## 0. 시작 전 (필수)

- [ ] **물리 E-stop 버튼**을 손이 닿는 곳에 둔다
- [ ] **관리자 앱**을 켜 둔다 — E-stop이 걸리면 앱 reset 외에 해제 수단이 없다
- [ ] 1~3단계는 **바퀴를 띄우거나 모터를 끈다**
- [ ] `.env`에 `OLLAMA_API_KEY`가 있다 (3단계에 필요)
- [ ] `ros2_ws` 빌드 완료 (`vica_interfaces`)

```bash
cd <workspace>/vica-voice-llm && .venv/bin/python -m pytest tests/ -q
```

- [ ] **실패 0** (숫자는 기록만) → 실패 _____ 건

> 2026-08-11 기준 **221 통과**. 크게 다르면 브랜치나 merge 상태를 먼저 본다
> (이 브랜치에는 `dev` 의 긴급어 수정 4커밋이 merge 되어 있다).

---

## 1단계 · 노드가 뜨는가 (10분)

```bash
ros2 launch launch/vica_voice.launch.py
```

- [ ] `VICA LLM intent node 시작 …`
- [ ] `VICA TTS node 시작 …`
- [ ] `VICA 웨이크워드 감시 시작 …` ← **"긴급어 상시 감시"로 나오면 옛 버전**
- [ ] `VICA 청각 안내 시작 …`

- [ ] `ls models/vica_bikaya_v1.onnx models/vica_modelb_v2.onnx` 둘 다 있음

**하나라도 안 뜨면 여기서 멈춘다.** 원인: ______________________

---

## 2단계 · 안전 ★★ (30분) — 오늘의 핵심

감시 창: `ros2 topic echo /vica/emergency`

### 2-1. 로봇이 자기 목소리에 반응하지 않는가 ★

```bash
ros2 topic pub -1 /vica/tts_request std_msgs/String \
  "{data: 'narration:지금 이동 중입니다. 먼저 현재 안내를 취소해 주세요.'}"
```

- [ ] 보통 볼륨 — `/vica/emergency` **무출력**
- [ ] **최대 볼륨** — `/vica/emergency` **무출력**
- [ ] 목적지 이름 문장도 무출력

```bash
ros2 topic pub -1 /vica/tts_request std_msgs/String \
  "{data: 'narration:행정대학건물 1층 행정지원실 앞에 도착했습니다.'}"
```

> 뜨면 **자가 트리거가 살아 있다.** 즉시 중단하고 그때의 문장·볼륨·거리를 기록한다.

기록: ______________________________________________

### 2-2. 사람 목소리에는 반응하는가 ★ — 2-1의 짝

- [ ] 로봇이 조용할 때 "멈춰" → `keyword: 멈춰` **출력됨**
- [ ] "정지" → 출력됨
- [ ] "스톱" → 출력됨

> ❌ **하나라도 안 잡히면 그날 시험은 여기서 끝난다.**
> 긴급 정지가 죽은 상태로 로봇을 움직이지 않는다.

- [ ] 로봇 발화 **직후** "멈춰" → 잡히기까지 약 _____ 초 (설계값 0.4)

> **설계값 0.4 초는 PC GPU 기준이다.** 2026-08-11 젯슨 실측에서 검증 STT
> (medium, cuda/float16)가 **2초 창 하나에 1.84초**를 썼다(무음 입력). 창 주기에
> 여유가 거의 없으므로 여기서 나오는 값이 `docs/jetson-handoff.md` §2 의
> "Jetson 예상 ~1.2초 — 실측할 것" 을 닫는 수치다. **느리게 나와도 그것이 결과다.**
> 참기 어려우면 창 1을 내리고 `VICA_VERIFY_STT_MODEL=small` 로 한 번 더 재서
> small↔medium 을 같은 날 수치로 비교해 둔다.

### 2-3. fail-safe가 사는가

TTS 노드만 종료하고 10초 뒤 "멈춰".

- [ ] 잡힘 (안 잡히면 감시가 **영구히 닫힌** 것)

### 2-4. 오탐이 없는가

전부 **무출력**이어야 한다.

- [ ] "천천히 가주세요"
- [ ] "잠깐만요"
- [ ] "행정지원실로 가줘"
- [ ] "감정지수가 뭐야"

### 2-5. 급한 말이 하던 말을 끊는가

긴 문장 재생 **중에** emergency 우선순위 요청.

- [ ] 하던 말이 **즉시 끊긴다** ← 선점 판정은 이것 하나다
- [ ] 끊긴 말이 **이어지지 않는다**
- [ ] 끊긴 뒤 긴급 문장이 나오기까지 _____ 초 (합성 지연. 판정 아님)

> **선점과 합성을 나눠서 본다.** 2026-08-11 실측에서 TTS 가 CPU 로 돌고 있어
> (`onnxruntime` 1.23.2 는 CUDA provider 가 없다) 4.46초 음성 합성에 **약 4초**가
> 걸렸다. GPU 면 10배 빠르다(`src/tts.py:19`). 하던 말이 즉시 멈추면 **선점은
> 통과**이고, 그 뒤 침묵은 합성 지연이지 결함이 아니다.

### 2-6. 말하는 중 신호가 켜지고 꺼지는가

`ros2 topic echo /vica/tts_state` · 세 문장짜리 재생.

- [ ] `true`/`false`가 **문장 수만큼** 반복 (계속 `true`면 안 됨)

---

### 2-7. 마이크를 연 채 자기 목소리를 듣는가 ★ (2026-08-10 신규)

**mute 를 끄고** 재는 계측이다. 지금 자가 트리거 방어는 mute 하나뿐인데, 그것이
없을 때 어떻게 되는지 아무도 모른다. `/vica/tts_state` 가 끊기면 fail-safe 시한
10초 동안 **실제로 마이크가 열리므로**, 그 구간의 위험도가 이 시험의 답이다.
남는 것은 AEC(reSpeaker XVF-3000) 성능이다.

launch 를 다시 띄운다.

```bash
ros2 launch launch/vica_voice.launch.py suppress_during_tts:=false
```

- [ ] 시작 로그에 **`suppress_during_tts=false — 계측 모드`** 경고가 뜬다

2-1 과 **같은 문장**을 재생하고 `/vica/emergency` 를 센다.

| 문장 | 보통 볼륨 | 최대 볼륨 |
| --- | --- | --- |
| "지금 이동 중입니다. 먼저 현재 안내를 취소해 주세요." | / 10 | / 10 |
| "행정대학건물 1층 행정지원실 앞에 도착했습니다." | / 10 | / 10 |
| "죄송합니다. 이동에 실패했습니다. 다시 시도해 주세요." | / 10 | / 10 |

**어느 쪽이 나와도 얻는 것이 있다.**

- 오인 0건 → AEC 가 잡아준다. mute 시간을 줄일 여지가 생기고 끼어들기 가능성이 열린다
- 오인 잦음 → mute 가 필수임이 증명된다. **fail-safe 10초가 위험 구간**이므로 그 값을 줄인다

> ⚠️ **AEC 는 자가 각성을 막지 못한다 (2026-08-13 측정).**
>
> 사람 목소리 "비카야" 를 스피커로 45초 되틀면서 운영과 같은 판정기로 채점했다.
>
> | 항목 | 값 |
> | --- | --- |
> | 대조군(녹음 직접 채점) | 8초에 5회 발동 |
> | 되틀기 45초 | **5회 발동** (억제 없으면 28회 예상 → 82% 억제) |
> | 환산 | **9초에 한 번꼴.** 인사말 7.24초당 약 0.8회 |
>
> 82 % 를 걸러도 **인사할 때마다 거의 매번 자기가 깨어난다.** 이유는 세 가지다 —
> 웨이크워드 모델은 크기가 아니라 패턴을 보고, AGC 가 지운 만큼 최대 31.6 dB 까지
> 되살리며, 스피커의 비선형 왜곡은 애초에 뺄 수 없다.
>
> **레벨로 AEC 를 판정하지 말 것.** 조건에 따라 ±20 dB 로 뒤집힌다. 판정기가
> 발동하느냐만 보면 된다.
>
> ➡️ **결론: mute 가 유일하게 실효 있는 방어다.** 2026-08-13 에 인사말 구간은
> 마이크를 mute 하기로 확정했다. 이 계측은 "mute 가 없으면 어떻게 되는가" 를
> 재는 것이지 mute 를 줄일 근거를 찾는 것이 아니다.
>
> ⚠️ **TTS 정규화(2026-08-13, +10.8 dB) 이후 조건이 바뀌었다.** 자기 목소리가 그만큼
> 크게 들어오므로 이전 수치는 무효다. 기록에 **"TTS 정규화 적용 후"** 를 같이 적는다.
>
> 근거: `devlog/2026-08-13-자가각성-AEC측정-TONY0043.md`

- [ ] **시험 뒤 `suppress_during_tts` 를 기본값(true)으로 되돌린다**

> ⚠️ 여기서 잡히면 **진짜 E-stop 이 걸린다.** 관리자 앱 또는 터미널로 푼다.

기록: ______________________________________________

---

## 3단계 · 웨이크워드 (30분)

감시 창: `ros2 topic echo /vica/wake`

### 3-1. "비카야"가 잡히는가 ★ — 진입 경로

각 칸 **10회**씩. 화자 2명.

| 거리 | 화자 A | 화자 B |
| --- | --- | --- |
| 0.5 m | / 10 | / 10 |
| **1.5 m** | / 10 | / 10 |
| 3 m | / 10 | / 10 |

- [ ] 1.5 m에서 **8/10 이상** (손잡이를 잡은 사용자의 실제 거리)
- [ ] 화자에 따라 크게 갈리지 않는다

### 3-2. 첫 호출과 두 번째가 다른가

- [ ] 첫 호출 → **"네?"** (말)
- [ ] 두 번째 호출 → **짧은 음**
- [ ] "네?"까지 걸린 시간 약 _____ 초

### 3-3. 비슷한 말에 열리지 않는가

전부 **무출력**이어야 한다.

- [ ] "비키야"
- [ ] "비카"
- [ ] "이거야"
- [ ] "미카엘"

> 3-1이 3-3보다 우선이다. **안 열리는 것이 잘못 열리는 것보다 나쁘다.**

---

## 4단계 · 대화 (30분, 로봇 정지 상태)

### 4-1. 전 구간 한 번

```text
"비카야"        →  "네?"
"화장실 가줘"    →  "화장실로 안내해드릴까요?"
"네"            →  "화장실로 안내를 시작합니다."
```

- [ ] 위 순서대로 진행된다
- [ ] **"네"** 한 글자를 즉시 알아듣는다 (되묻지 않는다)
- [ ] **조사가 맞는다** — "화장실**로**", "식당**으로**" ← 2026-08-05 수정, 실기 첫 확인

들린 그대로 적기: ______________________________________________

### 4-2. 실패 경로

- [ ] 없는 목적지 → 되묻기
- [ ] 알아듣지 못함 → "잘 듣지 못했습니다…"
- [ ] 안내가 끝난 뒤 다시 부르면 **"네?"** 로 돌아온다

### 4-3. 취소·일시정지·재개 (2026-08-10 신규, 실기 첫 확인)

`ros2 topic echo /vica/intent` 창을 하나 더 연다. **주행 전이므로 대부분 거부되는
것이 정상이다** — 여기서는 *intent 가 제대로 실리는가* 만 본다.

| 말 | `/vica/intent` 의 `intent` | 확인 |
| --- | --- | --- |
| "잠깐" | `pause` | ☐ |
| "잠깐만" | `pause` | ☐ |
| "다시 출발" | `resume` | ☐ |
| "취소해줘" | `cancel` | ☐ |
| "화장실 가줘" | `navigate` (pause 아님) | ☐ |

- [ ] **"멈춰"는 `/vica/intent` 가 아니라 `/vica/emergency` 로 간다** — 긴급 경로가
      먼저다. `intent` 에 `pause` 가 뜨면 잘못된 것이다
- [ ] 로봇이 **두 번 말하지 않는다** — 위 네 가지에서 음성 쪽은 침묵하고 Mission
      Manager 만 말한다(주행 중이 아니면 "지금은 안내 중이 아닙니다" 한 번)

> ⚠️ 이 경로는 **노트북에서 import 조차 되지 않아**(pydantic 미설치) 실행 검증이
> 처음이다. `/vica/intent` 에 아무것도 안 뜨면 `ros_node` 로그를 먼저 본다.

기록: ______________________________________________

---

## 5단계 · 주행 포함 (바퀴 띄운 상태에서 시작)

> ⚠️ 주변을 통제하고 물리 E-stop을 손에 둔다.

### 5-1. 안내 종단

- [ ] goal 전송 → 출발 멘트
- [ ] 회전마다 〈띵〉 + "좌회전 할게요" — **좌우가 맞는가**
- [ ] 거리 안내 ("약 10미터 남았습니다") 시점이 적절한가
- [ ] 도착음 + 도착 멘트

### 5-2. 주행 중 긴급어 ★

가장 시끄러운 조건이다 (모터 소음 + 안내 음성 + 사용자 발화).

- [ ] 주행 중 "멈춰" → 정지
- [ ] 주행 중 **자가 트리거 없음** (로봇 혼자 멈추지 않는가)

### 5-3. 주행 중 일시정지·재개 ★ (2026-08-10 신규)

> ⚠️ **"잠깐"과 "멈춰"의 결과가 다르다.** 이 차이를 확인하는 것이 목적이다.

| 말 | 기대 | 푸는 법 | 확인 |
| --- | --- | --- | --- |
| 주행 중 "잠깐" | 감속해 정지. **목적지를 기억** | 말로 "다시 출발" | ☐ |
| 이어서 "다시 출발" | 같은 목적지로 재개 | — | ☐ |
| 주행 중 "멈춰" | **비상정지 · 중앙 래치** | **관리자 앱 reset** | ☐ |

- [ ] "잠깐" 뒤 **"가자"** 로도 재개된다 (`is_paused` 문맥 판정)
- [ ] "잠깐" 정지가 **급정거로 느껴지지 않는가** — 뒤따르던 사람 기준

> 시각장애인·고령자가 뒤에서 따라온다. **예상 못 한 순간의 정지가 곧 위험**이므로
> 감속 느낌을 반드시 사람이 뒤에 선 채로 판단한다.

정지 거리 체감: ______________  재개까지 걸린 시간: ______________

---

### 5-4. 주행 중 취소 ★ (2026-08-10 신규)

취소만 **되묻는다.** 잘못 알아들으면 안내가 끊기기 때문이며, 되묻는 동안에도
**주행은 계속된다** — 확인 전에 멈추면 "아니오"일 때 되돌릴 수 없어서다
(`mission_logic.on_cancel_confirm_request`).

```text
🧑 "취소해줘"
🤖 "안내를 취소할까요?"      ← 이 동안에도 계속 간다
🧑 "취소"                    ← 다시 말해야 확정된다
🤖 "안내를 취소했습니다."
```

- [ ] "취소해줘" → **"안내를 취소할까요?"** 가 나온다
- [ ] 되묻는 동안 **로봇이 계속 간다** (여기서 멈추면 설계와 다르다)
- [ ] 다시 **"취소"** → 실제로 취소되고 정지한다
- [ ] 취소 뒤 **"비카야"** 로 새 목적지를 받을 수 있다

**되묻기를 무시했을 때**

- [ ] 아무 말 없이 30초 → 안내가 **그대로 이어진다** (취소되지 않는다)

**"네"로도 확정된다** (2026-08-10 수정, 실기 첫 확인)

원래는 `intent == "cancel"` 하나만 확정 경로여서 "취소"를 다시 말해야만 했다.
되묻는 문장을 Mission Manager 가 말하는 탓에 음성 쪽 대화 기록에 없어 "네"가
아무 데도 안 걸렸다. 취소를 내보낸 음성 노드가 직접 기억하도록 고쳤다
(`mission_command.CancelConfirm`).

- [ ] "취소해줘" → 되묻기 → **"네"** → 취소된다
- [ ] "취소해줘" → 되묻기 → **"아니요"** → **곧바로** "안내를 계속하겠습니다"
- [ ] 그 뒤 주행이 **끊기지 않고 이어진다**

> "아니요"도 즉시 응답하도록 `cancel_keep` intent 를 새로 두었다(2026-08-10).
> 전에는 부정을 전달할 값이 없어 Mission Manager 의 시한 30초가 지나야 응답이
> 나왔고, 눈으로 확인할 수 없는 사용자에게 그 30초가 침묵이었다.
>
> **로봇 저장소도 함께 받아야 한다** — `test/mock-touch-and-mic-open` 브랜치에
> 있고 `vica_mission_manager` 빌드가 필요하다(A.2).

응답까지 걸린 시간: ________ 초

기록: ______________________________________________

---

### 5-5. 터치센서 mock — 손잡이 잡기·놓기 (2026-08-10 신규)

터치센서가 미장착이라 키보드로 흉내낸다. **`enable_serial=false` 일 때만** 구독이
열린다 — 실기 배선에서는 이 경로가 아예 없다.

```bash
# 창 1 — guidance driver 를 mock 모드로
ros2 launch vica_user_guidance user_guidance.launch.py enable_serial:=false

# 창 2 — 키보드 도구 (스페이스 = 잡음/놓음)
ros2 run vica_user_guidance mock_touch_keyboard

# 창 3 — 값이 실리는지 확인
ros2 topic echo /vica/smart_handle_state --field user_contact
```

- [ ] 시작 로그에 **`터치센서 mock 활성`** 경고가 뜬다
- [ ] 스페이스를 누르면 `user_contact` 가 `true` 로 바뀐다
- [ ] 다시 누르면 `false` 로 돌아온다
- [ ] `q` 로 종료하면 **`false` 로 끝난다** (잡은 채로 남지 않는다)
- [ ] `enable_serial:=true` 로 띄우면 `/vica/mock_user_contact` 를 보내도 **바뀌지 않는다**

> 마지막 항목이 안전 확인이다. mock 이 실기 경로로 새면 **손 놓음 정지가 영구히
> 발동하지 않는다.**

**여기까지가 오늘 확인 범위다.** 3초 진입·0.5초 놓침 판정은 모드 상태 기계
(`/vica/handle_mode`)가 없어 아직 동작하지 않는다. 값이 실리는 것만 본다.

기록: ______________________________________________

---

## 6. 오늘 시험하지 않는 것

| 항목 | 이유 |
| --- | --- |
| 스마트핸들 **모드 진입·놓침 판정** | 모드 상태 기계 미구현. 5-5 는 입력이 실리는 것까지만 |
| 주행 실패 → 앱 알림 | 제안서 승인 대기 |
| 주행 중 끼어들기 | 재생 중 마이크를 닫으므로 불가 |

취소·일시정지·재개는 **오늘 시험한다.** 정지 상태는 4-3, 주행 중은 5-3(일시정지·재개)과
5-4(취소)다.

---

## 7. 조정한 값

바꾼 것만 적는다.

| 값 | 기본 | 바꾼 값 | 이유 |
| --- | --- | --- | --- |
| `suppress_during_tts` | true | | **시험 뒤 반드시 true 로 되돌린다** |
| `TAIL_SEC` | 0.4 | | |
| `gate_a` (호출) | 0.6 | | |
| `gate_b` (긴급) | 0.5 | | |
| `MAX_CHUNK_CHARS` | 40 | | |

---

## 8. 판정

- [ ] **2단계 전 항목 통과** ← 이것이 아니면 아래는 의미가 없다
- [ ] 3단계 통과
- [ ] 4단계 통과 (4-3 포함)
- [ ] 5단계 통과 (5-3 · 5-4 · 5-5 포함)

**총평 / 다음에 할 것**

______________________________________________

______________________________________________

**막힌 항목과 그때의 노드 로그**

______________________________________________

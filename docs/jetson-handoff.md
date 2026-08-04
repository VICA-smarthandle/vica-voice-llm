# 개발 Jetson 인수인계 — 2026-07-29 기준

이 문서는 **개발 PC(RTX 4070 Ti) 세션에서 개발 Jetson 세션으로 넘어가는 Claude
Code(또는 새 작업자)를 위한 것**이다. 이전 세션의 맥락 없이 이 문서만 읽고
이어서 작업할 수 있어야 한다.

## ⚠️ 브랜치 상황 (2026-08-04 갱신, 먼저 읽을 것)

**병합이 끝났다. 이제 `dev` 하나만 보면 된다** (개발 Jetson 은 `dev` 를 checkout).
7/29 시점의 "`feat/wakeword-integration` 을 쓰고 dev 는 나중에 병합" 안내는 무효이며,
그 브랜치와 `chore/jetson-only-cleanup`·`integrate/interfaces-migration` 은 정리했다
(마지막 것은 `archive/interfaces-migration-2026-07-19` 태그로 보존).

병합으로 들어온 것 (직접 다시 만들지 말 것):

- `tts_queue.py` + 새 `ros_tts_node` — **아래 5절 우선순위 2번(TTS 동기 재생 병목)이
  이걸로 해소됐다.** 워커 스레드 + 우선순위 큐 + 문장 단위 재생
- `stt.py` — Jetson CUDA ctranslate2 preload 수정 (STT GPU 가 안 되면 여기부터)
- `history.py`(멀티턴) · `replies.py`(고정 문구) · `tts_text.py`(문장 분할)

병합으로 **사라진 것**:

- `backend/`(FastAPI+SQLite) — 목적지는 `config/destinations.yaml` 단일 소스
- `ros_robot_state_stub` · `ros_state_machine_stub` — SIM 이 대체
- `ros2_ws/src/vica_interfaces/` 사본 — **정본은 `vica_ros2_ws` 다.**
  실행 전 `source ../vica_ros2_ws/install/setup.bash` (경로는 workspace 배치에 맞게)

토픽 이름도 `/vica/tts_active` → **`/vica/tts_state`** 로 통일했다 (계약 문서 등재본).
자세한 경위는 `VICA-team-workspace/devlog/2026-08-04.md`.

## 0. 지금 어디까지 왔나 (한 문단)

한국어 웨이크워드 모델 2종(호출 "비카야" / 긴급 "멈춰·정지·스톱")을 실녹음
4화자로 학습해 성능을 실측했고, 이 저장소에 **마이크 앞단(ros_wakeword_node)**
으로 통합했다(P1-a·b 완료). 로봇 없이 전체 서비스를 돌리는 **가상 로봇 +
계측(vica_sim.launch)** 도 준비됐다. **개발 Jetson 의 임무 = P1-c**: 여기서
전체를 기동해 baseline 수치를 얻고, 병목을 수치로 확인하며 서비스를 다듬는다.
로봇 이식은 그 다음이다.

## 1. 확정된 구조와 결정 (바꾸려면 사용자 승인)

```
reSpeaker ch0 (80ms 프레임, 상시)
  → openWakeWord 모델 A·B 동시 (전처리 공유, models/ 의 자립형 ONNX)
  ├─ B(긴급) ≥0.5 ×2연속 → 직전 ~2.5초+0.3초 → whisper medium → 정확 매칭
  │    → 통과 시 /vica/emergency (keyword=전사에서 추출한 정본 키워드)
  │      ⇒ 로봇 쪽 계약·코드 무변경 (설계 D1 해소)
  └─ A(호출) ≥0.6 ×2연속 → 삑 + /vica/wake → 청취 창(발화 끝 감지, 최대 6초)
       → whisper → /vica/user_text → 기존 LLM 흐름
```

- **전 구간 STT 검증** (2026-07-29 사용자 결정): 긴급은 모델 점수만으로 확정하지
  않는다. 즉시 밴드(고점수 무검증)는 보류 — 유사어(멈춤·정지야·스톡)가 모델을
  뚫는 것이 실측됐기 때문.
- 정확 매칭 규칙: 감탄사 접두 제거 후 키워드 반복과 완전 일치 + 허용 변형
  (종지·중지→정지, 맘차·마음차→멈춰). 정본: `src/wakeword_gate.py` (테스트 있음).
- 긴급어 구성(3단어 유지 vs 축소)은 **로봇 실기 라이브 시험 후 확정**. 스톱이
  약점(재현율 최저 + 오탐 대부분이 스톱 이웃)이라는 근거는 이미 확보됨.
- TTS 재생 중 자기 목소리 억제: `/vica/tts_state` (AEC 스피커 배선 전 임시).
  이름 정본은 `docs/ros2-interface.md` — dev 의 긴급어 감시도 같은 토픽을 구독한다.

## 2. 성능 스냅샷 (PC 실측, 조용한 조건 — Jetson 에서 재측정 필요)

| 항목 | 수치 |
| --- | --- |
| 종단 인식(미학습 화자, 관문→STT) | 멈춰 93% / 정지 86% / 스톱 78% |
| 함정어 최종 오탐 | 8/380 (2.1%) — 전부 whisper 도 같이 속은 경우 |
| 호출(비카야) | v1 재학습으로 일반 대화 오탐 대폭 개선 (라이브 재시험 확인) |
| STT 검증 지연 | PC GPU 0.3초 / **Jetson 예상 ~1.2초 — 실측할 것** |

## 3. Jetson 설치 (함정 주의)

```bash
# 저장소: GOVERNANCE 9절 절차로 workspace 통째 권장 (문서·도구 포함)
git clone https://github.com/VICA-smarthandle/VICA-team-workspace.git VICA-smarthandle
cd VICA-smarthandle && vcs import . < workspace.repos
git clone https://github.com/VICA-smarthandle/vica-wakeword.git   # 문서·DSP 도구
```

- **requirements.txt 는 x86 PC 의 freeze 다 — Jetson 에서 그대로 설치 금지.**
  `docs/jetson-setup.md`(기존 Jetson 구축 절차)를 따르고, 추가로:
  `openwakeword==0.6.0`, `onnxruntime==1.23.2`(aarch64 cp310 마지막 버전 — 핀 필수),
  `psutil`(계측용).
- 모델은 `models/` 에 커밋돼 있다(자립형 215KB×2). **학습 산출 원본은 가중치가
  사이드카(.onnx.data)에 분리돼 조용히 깨진다** — 새 모델을 받을 땐 반드시
  vica-wakeword 의 `consolidate_onnx.py` 를 거친 파일인지 확인.
- reSpeaker USB 접근(udev, DSP 도구용):
  `SUBSYSTEM=="usb", ATTRS{idVendor}=="2886", ATTRS{idProduct}=="0018", MODE="0666"`
  → `/etc/udev/rules.d/99-respeaker.rules`, reload+trigger.
- 마이크를 옮겨 꽂은 뒤 DSP 설정 일치 확인(학습-실전 분포, R4):
  `vica-wakeword: .venv/bin/python -m recorder.dsp_dump --check`
- **마이크는 한 프로그램만 쓴다.** 녹음 도구·데모·노드 동시 실행 금지.
- 환경변수: `VICA_STT_DEVICE=cuda`(Jetson ctranslate2 CUDA 빌드 전제),
  `VICA_WAKE_MODEL_A/B`(기본 models/ 경로), `VICA_VERIFY_STT_MODEL=medium`.

## 4. 기동과 계측

```bash
source /opt/ros/humble/setup.bash && source ../vica_ros2_ws/install/setup.bash
VICA_SIM_SESSION=baseline1 ros2 launch launch/vica_sim.launch.py
# 시험: "비카야 → 화장실 데려다줘 → 응" (이동→도착 안내), "멈춰!"(정지+래치),
#        래치 해제: ros2 topic pub --once /vica/sim/reset std_msgs/msg/Empty {}
.venv/bin/python tools/metrics_report.py logs/sim/baseline1.jsonl
```

상세: `docs/sim-guide.md`. **첫 임무 = baseline 세션 1회 돌려 보고서 확보.**

## 5. 다음 작업 우선순위 (병목 위치까지 특정돼 있음)

1. **baseline 수치 확보** (위 4절) — 이후 모든 개선은 이 수치와 비교.
   시나리오에 **"TTS 가 끝나자마자 즉시 '멈춰'"** 를 반드시 넣을 것 (아래 참고)
2. ~~TTS 동기 재생 병목~~ — **dev 병합으로 해소됨** (2026-08-04). 재생은 워커
   스레드 + 우선순위 큐이고, 문장 단위로 끊어 감시 사각지대까지 줄였다
3. **웨이크워드 검증 블로킹**: whisper 검증(~1.2초) 동안 프레임 처리 정지
   (`wakeword_monitor`) — 워커 스레드 분리 검토
4. LLM 왕복(~3.5초) 단축: 모델·스트리밍 TTS 시작 등
5. AEC 스피커(Soundcore 2, 주문됨) 도착 시: reSpeaker 3.5mm 출력 배선 →
   TTS 중 감시 유지로 전환 (vica-wakeword `docs/integration-design.md` 8절 D6)

**1번의 확인 항목** — `emergency_monitor` 는 mute 해제 시 버퍼를 비우고
`window_sec` 만큼 판정을 보류하는데, `wakeword_monitor` 는 버퍼 비우기와 게이트
리셋만 한다. 해제 직후 긴급 검증이 걸리면 참조할 과거 오디오가 짧아질 수 있다.
**결함으로 확인된 것은 아니고, 실기에서 재현되는지 볼 항목이다.**

## 6. 문서 지도

| 위치 | 내용 |
| --- | --- |
| `vica-wakeword/docs/integration-design.md` | 통합 설계 전체 + 결정 D1~D7 상태 |
| `vica-wakeword/docs/stt-gate-findings.md` | STT 검증 실측·매칭 규칙 근거 |
| `vica-wakeword/docs/modelb-loso-findings.md` | 화자 일반화·긴급어 구성 판단 재료 |
| `vica-wakeword/docs/live-test-protocol.md` | 로봇 실기 라이브 시험 절차 (P1-d) |
| `VICA-smarthandle/devlog/2026-07-2*.md` | 일자별 결정·경위 |
| `models/MODELS.md` | 모델 출처·관문값·주의사항 |

## 7. 건드리지 않는 것

- `/vica/*` 토픽 계약, `vica_interfaces` 메시지 (GOVERNANCE 5절 — 변경은 팀 승인)
- 긴급 정지 체인(vica_ros2_ws)·E-stop reset 정책
- `[SIM ONLY]` 노드(ros_robot_sim, ros_metrics_node)는 실기 이식 때 로봇 팀
  노드로 교체 대상 — 실기 코드에 섞지 말 것

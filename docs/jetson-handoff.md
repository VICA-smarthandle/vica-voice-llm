# 개발 Jetson 인수인계 — 2026-07-29 기준

이 문서는 **개발 PC(RTX 4070 Ti) 세션에서 개발 Jetson 세션으로 넘어가는 Claude
Code(또는 새 작업자)를 위한 것**이다. 이전 세션의 맥락 없이 이 문서만 읽고
이어서 작업할 수 있어야 한다.

## ⚠️ 브랜치 상황 (2026-07-29, 먼저 읽을 것)

이 문서와 웨이크워드 통합 작업은 **`feat/wakeword-integration` 브랜치**에 있다
(개발 Jetson 은 이 브랜치를 checkout 할 것). 한편 팀의 원격 **`dev` 브랜치**에는
별도의 대량 작업이 있고 **아직 병합되지 않았다** (사용자 결정: 병합은 추후):

- `tts_queue.py` — TTS 재생 큐 (아래 5절 우선순위 2번이 **이미 해결돼 있음**.
  직접 다시 만들지 말고 dev 병합으로 가져올 것)
- `stt.py` — Jetson CUDA ctranslate2 preload 수정 (Jetson 에서 STT GPU 가 안 되면
  이 픽스부터 확인)
- ros_tts_node 대폭 개편, ros_state_machine_stub 삭제, 테스트 다수

겹치는 파일 5개(ros_tts_node·launch·requirements·CLAUDE.md·.gitignore)라 병합 시
충돌 해결 필요. 병합 전까지 이 브랜치 단독으로도 시뮬레이션은 완결 동작한다.

> **2026-08-04 갱신 — ros_tts_node 재이식은 불필요해졌다.**
> 이 문서는 원래 "dev 의 새 ros_tts_node 에 이 브랜치의 `/vica/tts_active` 발행을
> 재이식하라"고 지시했으나, dev 는 **같은 신호를 `/vica/tts_state` 로 이미 발행**하며
> 모든 면에서 우수하다 (문장 단위 발행 → 감시 사각지대 축소, `TAIL_SEC=0.4` 잔향
> 처리, 워커 스레드 + 우선순위 큐, `docs/ros2-interface.md` 계약 등재).
> 그래서 이 브랜치의 토픽 이름을 `/vica/tts_state` 로 통일했다.
> **병합 시 ros_tts_node 는 dev 것을 그대로 채택하면 되고, 구독자 쪽은 손댈 필요가 없다.**

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
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
VICA_SIM_SESSION=baseline1 ros2 launch launch/vica_sim.launch.py
# 시험: "비카야 → 화장실 데려다줘 → 응" (이동→도착 안내), "멈춰!"(정지+래치),
#        래치 해제: ros2 topic pub --once /vica/sim/reset std_msgs/msg/Empty {}
.venv/bin/python tools/metrics_report.py logs/sim/baseline1.jsonl
```

상세: `docs/sim-guide.md`. **첫 임무 = baseline 세션 1회 돌려 보고서 확보.**

## 5. 다음 작업 우선순위 (병목 위치까지 특정돼 있음)

1. **baseline 수치 확보** (위 4절) — 이후 모든 개선은 이 수치와 비교
2. **TTS 동기 재생 병목**: `ros_tts_node._on_intent` 가 콜백 안에서 `speak()` 를
   동기 호출 → 재생 동안 노드 블로킹. 재생 스레드+큐로 분리
3. **웨이크워드 검증 블로킹**: whisper 검증(~1.2초) 동안 프레임 처리 정지
   (`wakeword_monitor`) — 워커 스레드 분리 검토
4. LLM 왕복(~3.5초) 단축: 모델·스트리밍 TTS 시작 등
5. AEC 스피커(Soundcore 2, 주문됨) 도착 시: reSpeaker 3.5mm 출력 배선 →
   TTS 중 감시 유지로 전환 (vica-wakeword `docs/integration-design.md` 8절 D6)

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

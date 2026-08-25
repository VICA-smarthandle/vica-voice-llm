# 비카 로봇 마이크 장착 절차서 (reSpeaker v3.0)

작성: 2026-08-25. 대상: 주행용 로봇 젯슨 (개발 젯슨 J4012 아님).
전제 브랜치: `vica-voice-llm` **dev** (`feat/aec-audio-out` 은 2026-08-25 dev 머지됨.
AEC 뮤트 제거·barge-in·VAD 말끝·방향 잠금 포함. 근거 실측: `devlog/2026-08-24-AEC뮤트제거-barge-in-TONY0043.md`).

⚠️ **로봇에서는 음성 긴급어가 진짜 E-stop 입니다.** 시험 중 "멈춰"를 말하면
중앙 래치가 걸려 관리자 앱 reset 이 필요해집니다. **6단계 전까지는
`emergency_estop_bridge`·Mission·Safety 를 띄우지 말고 음성 노드만 시험합니다.**
바퀴가 도는 검증은 AGENTS 5절 조건(바퀴 띄움·주변 통제·물리 E-stop) 필수.

## 1. 하드웨어 장착 — 방향이 전부다

- [ ] reSpeaker 를 로봇에 **수평·마이크면 노출**로 고정. **한 번 고정하면 돌리지
      않는다** — DOA 보정(4단계)이 장착 각도에 묶인다.
- [ ] 위치: 모터·팬에서 최대한 멀리, 스마트핸들(사용자 입 방향)에 가리는 것 없이.
- [ ] 스피커 AUX 를 reSpeaker 3.5mm 잭에 연결 (AEC 참조 — 다른 출력 금지).
- [ ] USB 를 로봇 젯슨에 연결 → `lsusb | grep 2886:0018` 확인.

## 2. 소프트웨어 설치 (함정 목록 — jetson-setup.md 보완)

- [ ] 3개 저장소 fetch, `vica-voice-llm` 은 `feat/aec-audio-out` checkout.
- [ ] venv 함정 (handoff 3절 + 이번 추가분):
  - `onnxruntime==1.23.2` 핀 (aarch64 cp310 마지막 GPU 호환)
  - `openwakeword==0.6.0` — `--no-deps` 로 깔았다면 `scipy` `scikit-learn`
    `joblib` `threadpoolctl` 도 `--no-deps` 로 함께
  - 전처리 모델 1회: `.venv/bin/python -c "import openwakeword.utils as u; u.download_models()"`
  - **`pyusb` (신규)** — 칩 상태(VAD·DOA) 읽기. 없으면 웨이크워드 노드가 기동 실패
  - supertonic·whisper 캐시 준비 후 `.env` 에 `HF_HUB_OFFLINE=1`
    (없으면 로드가 27분 멈춘 실측 있음)
- [ ] `.env`: `HF_HUB_OFFLINE=1` `VICA_STT_DEVICE=cuda` `VICA_STT_COMPUTE=float16`
      (ctranslate2 CUDA 확인: `python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"`
      → 0 이면 STT 가 CPU — 별도 조치 전까지 느림)
- [ ] **udev 규칙 (신규 필수)**:
      `echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="2886", ATTRS{idProduct}=="0018", MODE="0666"' | sudo tee /etc/udev/rules.d/99-respeaker.rules && sudo udevadm control --reload-rules && sudo udevadm trigger`
- [ ] 폴백 없음 정책: 재생·마이크·레지스터 중 하나라도 안 잡히면 노드가 기동
      실패한다(의도된 동작). 실패 메시지가 원인을 말해준다.

## 3. 설치 검증 (조용한 상태, 모터 꺼짐)

- [ ] `vica-wakeword: .venv/bin/python -m recorder.dsp_dump --check` — DSP 동결 확인
- [ ] `HF_HUB_OFFLINE=1 .venv/bin/python -u -m tools.aec_probe` —
      참조(ch5) 흐름 + 수렴 ≥ 6dB (개발 실측 +19.4dB). 볼륨은 원음 peak < 0.9.

## 4. DOA 장착 보정 (사용자가 핸들 잡은 자세로)

- [ ] `HF_HUB_OFFLINE=1 .venv/bin/python -u -m tools.doa_probe` —
      A구간: 핸들 위치에서 대본 낭독 / B구간: 행인 위치에서 낭독
- [ ] 출력된 `VICA_USER_DOA_CENTER` / `VICA_USER_DOA_WIDTH` 를 `.env` 에 기록.
      ("비카야" 방향 잠금이 1순위이므로 이 값은 잠금이 없을 때의 안전망이다)

## 5. 기능 검증 (음성 노드만, 각 시험은 안내 → "시작" 신호로)

`ros_tts_node` + `ros_wakeword_node` 만 기동. (감시 유지(AEC)가 기본값이다 —
2026-08-25 표준 배선 확정. 예외 환경만 `VICA_TTS_MUTE=on`.) 순서:

| # | 시험 | 통과 기준 |
| --- | --- | --- |
| 5-1 | "비카야" ×10 (정지 상태) | 인식 ≥ 9/10, "네?" 응답 |
| 5-2 | 비카야 → 목적지 발화 | 말끝에서 창이 닫히고 전사 정확 (VAD 말끝) |
| 5-3 | 비카야 → 침묵 | 30초 창 뒤 조용히 닫힘, **유령 user_text 0** |
| 5-4 | TTS 중 "멈춰" ×10 | `/vica/emergency` 발행 (개발 실측 14/14) — 브리지 없으니 래치 안 걸림 |
| 5-5 | 질문 중 barge-in | 사용자 끼어들기만 끊김, 행인 위치 발화는 무시 |
| 5-6 | **모터 구동음 재실측** (바퀴 띄우고) | 5-1~5-4 반복 — HPF·잡음억제 실효 확인. 여기부터 AGENTS 5절 조건 |

## 6. 종단 연결 (팀 조건 갖춘 뒤)

- [ ] launch 전체(LLM 포함) + Mission Manager + E-stop 브리지 — 검증 시나리오 C
- [ ] "비카야 → 잠깐만" → PAUSED 실기 (음성 intent 연결 [미검증] 해소)

## 7. 장착 후 백로그 (얹고 나서 고친다)

시간값 튜닝(VICA_LISTEN_* — 실사용), 대화 세션 모드, 시나리오 6.1·아키텍처
5.3 문서 갱신, `af22b85`(tts.py 정규화) 중복 정리, Soundcore 2 재측정.

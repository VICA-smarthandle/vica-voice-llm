# 웨이크워드 모델

| 파일 | 대상 | 학습일 | 관문(잠정) | 성능 (실측) |
| --- | --- | --- | --- | --- |
| `vica_bikaya_v1.onnx` | 호출어 "비카야" | 2026-07-29 | 0.6 ×2프레임 | 미학습 회차 재현 22/24 @0.6. 실녹음 오답 1,800개로 학습(일반 대화 오탐 억제) |
| `vica_modelb_v2.onnx` | 긴급어 멈춰·정지·스톱(스탑 포함) | 2026-07-29 | 0.5 ×2프레임 + **whisper 검증 필수** | LOSO 종단: 멈춰 93% / 정지 86% / 스톱 78%, 함정어 오탐 2.1% |

- **자립형 ONNX 다** (가중치 내장, 215KB). 학습 파이프라인이 내보내는 원본은
  가중치가 사이드카(.onnx.data)에 분리돼 있어 **합치기 전 파일을 여기 두면
  조용히 깨진다** — vica-wakeword `consolidate_onnx.py` 를 거친 파일만 배치할 것.
- 긴급 감지는 모델 단독으로 확정하지 않는다. **전 구간 whisper 검증**(2026-07-29
  결정)을 거쳐 `wakeword_gate.match_emergency_transcript()` 로 판정한다.
- Jetson 은 onnxruntime aarch64 cp310 이 1.23.2 까지만 있다 (vica-wakeword
  docs/stage0-findings.md).
- 학습 데이터·재현 절차·상세 성능: vica-wakeword 저장소
  (`docs/modelb-loso-findings.md`, `docs/stt-gate-findings.md`).

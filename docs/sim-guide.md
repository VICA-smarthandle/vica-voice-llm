# 로봇 없는 전체 서비스 시뮬레이션 + 계측 가이드

로봇 주행 없이 음성/LLM 서비스를 **실전처럼** 돌리고, 모든 시간·자원을 기록해
개선점을 숫자로 찾기 위한 환경이다. (2026-07-29, P1 개발 Jetson 단계용)

## 무엇이 실전과 같은가

```
"비카야" → 삑 → "화장실 데려다줘" → LLM → "안내해드릴까요?" → "응"
  → [가상 로봇] 이동 시작 (pose 거리/0.8㎧ 로 주행 시간 계산, 층 이동 +5초)
  → 이동 중 robot_state 발행 (is_moving=true)
  → 도착 → "…앞에 도착했습니다" (TTS)
"멈춰!" → whisper 검증 → /vica/emergency → [가상 로봇] 즉시 정지 + 래치
  → 래치 중 새 이동 거부, 자동 재개 없음 (실제 안전 규칙 그대로)
  → 해제: ros2 topic pub --once /vica/sim/reset std_msgs/msg/Empty {}
```

토픽 계약은 실기와 동일하다 — 가상 로봇(ros_robot_sim)과 계측(ros_metrics_node)만
[SIM ONLY] 이며, 실기 통합 때 로봇 팀 노드로 갈아끼운다.

## 실행

```bash
source /opt/ros/humble/setup.bash && source ../vica_ros2_ws/install/setup.bash
VICA_SIM_SESSION=평가1 ros2 launch launch/vica_sim.launch.py
```

띄우는 노드: LLM · TTS · 웨이크워드(마이크 앞단) · 가상 로봇 · 계측.

## 기록과 보고서

- 모든 이벤트·시스템 사용량이 `logs/sim/<세션>.jsonl` 에 실시간 저장된다.
- 종료 후:

```bash
.venv/bin/python tools/metrics_report.py logs/sim/평가1.jsonl
```

보고서 내용:

| 항목 | 의미 |
| --- | --- |
| 청취+STT | 삑 → 발화 텍스트 완성 |
| LLM | 텍스트 → 의도 해석 완료 |
| **체감 응답** | **부르고 나서 로봇이 말을 시작할 때까지** — 서비스 핵심 지표 |
| TTS 재생 | 말한 길이 |
| 긴급 반응 | /vica/emergency 발행 → 가상 로봇 정지 (ms) |
| 무응답 호출 | 불렀는데 아무 말 없음 = 호출 오탐 후보 수 |
| 시스템 | CPU/RAM(+PC 는 GPU) 평균·최대. Jetson GPU 는 tegrastats 별도 |

## 알아둘 것

- 마이크는 한 프로그램만 쓴다 — 녹음 도구·데모와 동시 실행 금지.
- 가상 로봇의 도착 안내는 `/vica/intent` 의 reply 필드로 TTS 에 전달된다
  ([SIM ONLY] — 실기의 안내 경로는 로봇 팀 설계를 따른다). 안내 멘트에
  긴급어를 넣지 않는다 (mission_logic 규칙과 동일 원리).
- 이동 판단 조건은 실기 명세와 동일: navigate + 목적지 확정 + 확인 완료 +
  safety normal (docs/ros2-interface.md).

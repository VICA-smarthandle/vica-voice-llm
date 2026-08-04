"""웨이크워드 2단 파이프라인의 순수 로직 (P1-a).

openWakeWord 모델(관문) → whisper 검증(정확 매칭) 구조에서 하드웨어가 필요 없는
부분만 모았다: ① 프레임 점수의 지속·쿨다운 판정 ② 검증 전사의 긴급어 매칭.
마이크·모델 로드는 ros_wakeword_node(P1-b)가 담당한다.

설계·실측 근거는 vica-wakeword 저장소에 있다:
  - docs/integration-design.md   (2단 구조, 2연속 프레임, 긴급 우선)
  - docs/stt-gate-findings.md    (정확 매칭 규칙 채택 실측 — 실녹음 1,133개:
    종단 멈춰 95% / 정지 92% / 스톱 83%, 함정어 최종 오탐 4/380)

안전 원칙 (CLAUDE.md): 이 모듈은 감지·판정까지만 한다. 정지의 결정·실행은
Safety Supervisor / State Machine 이 한다.
"""
from __future__ import annotations

import re
from typing import Optional

# 긴급어 정본은 vica_ros2_ws/vica_mission_manager/mission_logic.HARD_EMERGENCY_KEYWORDS.
# 여기 나열된 키워드는 모두 그 정본에 포함된다 — 새 키워드를 추가할 때는 정본과
# 함께 바꿔야 한다 (GOVERNANCE 5절: producer/consumer 동시 변경).
EMERGENCY_KEYWORDS = ("멈춰", "정지", "스톱", "스탑")

# whisper 오전사 허용 변형 → 정본 키워드. 실측(stt-gate-findings)에서 확인된 것만
# 넣는다. 일상어와 겹치는 변형을 임의로 추가하지 말 것.
TRANSCRIPT_VARIANTS = {"종지": "정지", "중지": "정지", "맘차": "멈춰", "마음차": "멈춰"}

# "어어 멈춰!" 처럼 감탄사가 앞에 붙는 실제 위급 발화 형태를 허용한다
# (녹음 상황 'interjection' 과 동일 근거). 문장 속 키워드("정지선 넘지 마")는
# 정확 매칭이 걸러낸다.
_INTERJECTION_PREFIX = re.compile(r"^(어+|아+|야|오+|으+|헉)+")
_STRIP = re.compile(r"[\s.,!?~'\"…]+")


def match_emergency_transcript(text: str) -> Optional[str]:
    """검증 전사에서 긴급어를 정확 매칭한다.

    감탄사 접두를 뗀 나머지가 키워드(또는 허용 변형)의 1회 이상 반복과 완전히
    일치할 때만 정본 키워드를 돌려준다. 아니면 None.

    반환된 키워드는 EmergencyEvent.keyword 로 그대로 쓸 수 있다 — 전 구간 STT
    검증(2026-07-29 결정) 덕분에 항상 전사가 있어, 어떤 단어였는지 특정하지
    못하는 문제(설계 D1)가 생기지 않는다.
    """
    if not text:
        return None
    norm = _STRIP.sub("", text)
    norm = _INTERJECTION_PREFIX.sub("", norm)
    if not norm:
        return None
    for k in (*EMERGENCY_KEYWORDS, *TRANSCRIPT_VARIANTS):
        if re.fullmatch(f"(?:{k})+", norm):
            return TRANSCRIPT_VARIANTS.get(k, k)
    return None


class FrameGate:
    """프레임 점수의 관문 판정 — 지속(연속 프레임)과 쿨다운을 관리한다.

    순간 스파이크(문 닫는 소리 등) 오탐을 막기 위해 임계값을 persist 프레임
    연속으로 넘어야 발동한다 (80ms 프레임 × 2 = 160ms). 발동 후 cooldown_sec
    동안은 같은 외침을 중복 발동하지 않는다.

    시각(now)을 인자로 받는 순수 로직이라 시계 없이 단위 테스트할 수 있다.
    """

    def __init__(self, threshold: float, persist: int = 2, cooldown_sec: float = 2.0):
        self.threshold = threshold
        self.persist = persist
        self.cooldown_sec = cooldown_sec
        self._streak = 0
        self._last_fire = float("-inf")

    def feed(self, score: float, now: float) -> bool:
        """프레임 점수 하나를 넣고, 이번 프레임에 발동해야 하면 True."""
        self._streak = self._streak + 1 if score >= self.threshold else 0
        if self._streak >= self.persist and now - self._last_fire >= self.cooldown_sec:
            self._last_fire = now
            self._streak = 0
            return True
        return False

    def reset(self) -> None:
        """다른 경로가 발동했을 때(예: 긴급 우선) 진행 중인 지속 카운트를 버린다."""
        self._streak = 0

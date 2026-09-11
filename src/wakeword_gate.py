"""웨이크워드 2단 파이프라인의 순수 로직 (P1-a)."""
from __future__ import annotations

import re
from typing import Optional

EMERGENCY_KEYWORDS = ("멈춰", "정지", "스톱", "스탑")

TRANSCRIPT_VARIANTS = {"종지": "정지", "중지": "정지", "맘차": "멈춰", "마음차": "멈춰"}

_INTERJECTION_PREFIX = re.compile(r"^(어+|아+|야|오+|으+|헉)+")
_STRIP = re.compile(r"[\s.,!?~'\"…]+")


def match_emergency_transcript(text: str) -> Optional[str]:
    """검증 전사에서 긴급어를 정확 매칭한다."""
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
    """프레임 점수의 관문 판정 — 지속(연속 프레임)과 쿨다운을 관리한다."""

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

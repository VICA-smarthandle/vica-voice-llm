"""미션이 낸 질문 기록(/vica/question, 설계 2026-11-14 4.1)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class OpenQuestion:
    id: str
    kind: str
    text: str
    option_ids: tuple
    opened_at: float


def from_fields(id: str, kind: str, text: str, option_ids: Sequence[str], opened_at: float) -> Optional[OpenQuestion]:
    """메시지 칸 → 열린 질문. id 가 비면 열린 질문 없음."""
    return OpenQuestion(id=id, kind=kind, text=text, option_ids=tuple(option_ids), opened_at=opened_at) if id else None


def log_line(q: Optional[OpenQuestion], now: float) -> str:
    """발화마다 남기는 한 줄 — 그 말이 들어올 때 열려 있던 질문."""
    if q is None:
        return "[질문] 없음"
    return f"[질문] id={q.id} kind={q.kind} 나이={max(0.0, now - q.opened_at):.1f}s"

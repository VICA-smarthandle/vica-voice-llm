"""스마트핸들 모드 첫 질문과 그 응답 판정 (순수 로직)."""
from __future__ import annotations

from typing import Optional

AFFIRMATIVES = frozenset(
    {"네", "예", "응", "어", "그래", "그래요", "맞아", "맞아요",
     "좋아", "좋아요", "네네", "네맞아요", "응응", "가자", "가줘"}
)
NEGATIVES = frozenset(
    {"아니", "아니요", "아뇨", "아니야", "아니에요", "싫어", "싫어요", "취소"}
)
SOFT_AFFIRMATIVES = frozenset({"음", "으음", "음음", "어어", "어어어"})

YES = "yes"
NO = "no"

DEFAULT_ANSWER_WINDOW_SEC = 30.0


def normalize_short_reply(text: str) -> str:
    """STT 가 붙이는 구두점·공백을 제거해 짧은 답변을 비교 가능하게 만든다."""
    return "".join(ch for ch in text if ch.isalnum())


def classify_short_reply(text: str) -> Optional[str]:
    """짧은 긍정/부정이면 `YES`/`NO`, 아니면 None."""
    word = normalize_short_reply(text)
    if word in AFFIRMATIVES or word in SOFT_AFFIRMATIVES:
        return YES
    if word in NEGATIVES:
        return NO
    return None


class ModeQuestion:
    """모드를 물었는지 기억하고, 바로 다음 발화를 그 답으로 본다."""

    def __init__(self, answer_window_sec: float = DEFAULT_ANSWER_WINDOW_SEC) -> None:
        self.answer_window_sec = answer_window_sec
        self._asked_at: Optional[float] = None

    @property
    def waiting(self) -> bool:
        """답을 기다리는 중인지. 시간 초과는 노드가 아니라 `take_answer` 가 본다."""
        return self._asked_at is not None

    def on_asked(self, now: float) -> None:
        """`MODE_ASK` 를 발화했다."""
        self._asked_at = now

    def take_answer(self, text: str, now: float) -> Optional[str]:
        """대기 중이면 발화를 답으로 해석한다. `YES`/`NO`, 아니면 None."""
        if self._asked_at is None:
            return None
        if now - self._asked_at >= self.answer_window_sec:
            self._asked_at = None
            return None
        answer = classify_short_reply(text)
        if answer is not None:
            self._asked_at = None
        return answer

    def reset(self) -> None:
        """안내가 끝났거나 취소됐다. 다음 사용자를 위해 대기를 비운다."""
        self._asked_at = None

"""대화 맥락 보관 (길이 제한 + 대화 경계)."""
from __future__ import annotations

from typing import Iterable, Optional

DEFAULT_MAX_MESSAGES = 8

DEFAULT_IDLE_RESET_SEC = 180.0


class ConversationHistory:
    def __init__(
        self,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        idle_reset_sec: float = DEFAULT_IDLE_RESET_SEC,
    ) -> None:
        self.max_messages = max_messages
        self.idle_reset_sec = idle_reset_sec
        self._messages: list = []
        self._last_turn_at: Optional[float] = None

    def __len__(self) -> int:
        return len(self._messages)

    @property
    def messages(self) -> list:
        """LLM 에 넘길 최근 맥락 (읽기용 복사본)."""
        return list(self._messages)

    def begin_turn(self, now: float) -> bool:
        """새 발화를 시작한다. 대화가 끊겼다고 판단해 비웠으면 True."""
        previous = self._last_turn_at
        self._last_turn_at = now
        if previous is None:
            return False
        if now - previous < self.idle_reset_sec:
            return False
        self._messages.clear()
        return True

    def extend(self, messages: Iterable) -> None:
        self._messages.extend(messages)
        if self.max_messages >= 0:
            del self._messages[: max(0, len(self._messages) - self.max_messages)]

    def clear(self) -> None:
        self._messages.clear()
        self._last_turn_at = None

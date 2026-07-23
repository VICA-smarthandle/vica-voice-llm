"""대화 맥락 보관 (길이 제한 + 대화 경계).

공용 안내 로봇이라 사용자가 계속 바뀐다. 맥락을 그냥 이어 두면 다음 사용자의
"거기로 가줘"가 앞사람이 말한 목적지로 해석된다. 한동안 발화가 없으면 대화가
끝난 것으로 보고 맥락을 비운다.

메시지 타입에 의존하지 않는다 (langchain 메시지든 무엇이든 그대로 담는다).
덕분에 무거운 의존성 없이 unit test 로 검증된다.
"""
from __future__ import annotations

from typing import Iterable, Optional

# 최근 몇 개를 남길지. 길어져도 프롬프트가 커지지 않게 한다.
DEFAULT_MAX_MESSAGES = 8

# 이 시간 동안 발화가 없으면 다음 발화는 새 대화로 본다.
# 안내를 마치고 사용자가 떠난 뒤 다른 사람이 말을 거는 상황을 가정한 값이다.
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
        """새 발화를 시작한다. 대화가 끊겼다고 판단해 비웠으면 True.

        판단 시점이 '발화를 받은 순간'인 이유: 타이머로 비우면 사용자가 잠깐
        생각하는 사이에 맥락이 사라질 수 있다. 실제로 다음 말이 들어왔을 때만
        경과 시간을 보고 결정한다.
        """
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

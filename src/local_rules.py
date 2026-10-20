"""로컬 LLM 전용 규칙 스위치 (2026-10-05, 인수인계 "로컬 LLM 정비")."""
from __future__ import annotations

import os
from typing import Optional

ANSWER_WAIT_STATES = frozenset({"awaiting_user", "confirming", "asking_next", "asking_wait_time"})

HISTORY_MAX_MESSAGES = 16

LOCAL_NUM_CTX = int(os.environ.get("VICA_LOCAL_NUM_CTX", "4096"))

LOCAL_PROMPT_RULES = """
[로컬 추가 규칙 — 이 규칙이 위 규칙보다 먼저다]
1. 로봇의 마지막 말이 "안내를 받으시겠어요?"면 답은 affirm 또는 deny 다. 목적지를 말하면 navigate.
2. 로봇의 마지막 말이 "여기서 대기할까요?"·"안내를 마칠까요?" 같은 도착 질문이면
   답은 affirm·deny·wait·finish 만 고른다. 목적지를 다시 제안하지 않는다.
3. 사용자가 시간만 말하면("한 5분?", "십 분") intent 는 wait 다.
4. 목적지 목록에 딱 맞는 곳이 없으면 비슷한 곳을 고르지 말고 clarify 로 답한다.
5. 로봇에게 한 말이 아니면(옆 사람 대화, 혼잣말) intent 는 unknown, reply 는 빈 문자열."""


def enabled() -> bool:
    """스위치가 켜졌는가. 켜도 VICA_INTENT_INPUT=text 일 때만 참이다."""
    on = os.environ.get("VICA_LOCAL_RULES", "").strip().lower() in ("1", "true", "on", "yes")
    text_mode = os.environ.get("VICA_INTENT_INPUT", "text").strip().lower() == "text"
    return on and text_mode


def short_answer_allowed(dialog_state: Optional[str]) -> bool:
    """짧은 대답(네·아니요·그래)을 LLM 없이 즉시 처리해도 되는가."""
    if not dialog_state:
        return True
    return dialog_state in ANSWER_WAIT_STATES


def should_clear_history(prev_state: Optional[str], new_state: Optional[str]) -> bool:
    """접근 질문이 새로 시작되면(→ awaiting_user) 앞 사람과의 대화를 비운다(수리안 나)."""
    return new_state == "awaiting_user" and prev_state != "awaiting_user"


class MisheardGate:
    """듣기 창 밖에서 LLM 이 못 알아들은 말(unknown·question)에 대한 대꾸 정책."""

    def __init__(self) -> None:
        self._strikes = 0

    def reset(self) -> None:
        self._strikes = 0

    def decide(self, retry_prompt: str) -> Optional[str]:
        """못 알아들은 말 하나에 대해 할 말을 돌려준다. None = 침묵."""
        self._strikes += 1
        return retry_prompt if self._strikes == 1 else None

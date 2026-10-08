"""로컬 LLM 전용 규칙 스위치 (2026-10-05, 인수인계 "로컬 LLM 정비").

로컬 LLM(Ollama, 믿음 등)과 OpenAI 는 같은 파서·노드 코드를 쓴다. 로컬만 고치려고
공용 함수를 그냥 고치면 OpenAI 실행도 같이 바뀐다. 그래서 이번 로컬 수정은 전부
이 스위치 뒤에 둔다 — 기본은 꺼짐이라 OpenAI 실행(Realtime·글자·대체 처리)은
지금과 같고, 로컬 실행 명령에서만 켠다.

    export VICA_LOCAL_RULES=1          # 켬
    export VICA_INTENT_INPUT=text      # 켜도 text 일 때만 동작 (audio 는 Realtime 몫)

여기는 ROS·LLM 을 모르는 순수 판단만 둔다 — 장치 없이 시험된다.
"""
from __future__ import annotations

import os
from typing import Optional

# 미션 dialog_state 중 "사용자 대답을 기다리는 단계". 짧은 대답(네·아니요)의
# 즉시 처리는 이때만 쓴다 — 아무도 묻지 않았는데 "그래"가 지나간 확인 질문에
# 붙는 일을 막는다(10-02 시연 문제 ①).
ANSWER_WAIT_STATES = frozenset({"awaiting_user", "confirming", "asking_next", "asking_wait_time"})

# 로컬 기록 길이·비우기. 로봇이 소리 낸 말 전부(미션 질문 포함)를 넣으므로
# 소리 모드와 같은 16줄로 둔다. 시간으로는 비우지 않는다(대기 10~30분 뒤에도 기억).
HISTORY_MAX_MESSAGES = 16

# 로컬 모델 문맥 길이. Ollama 기본값에 기대지 않고 명시한다 — 목적지 목록
# 지시문(1.6~2.1k 토큰, 09-28 벤치) + 16줄 기록 + 규칙이 넉넉히 들어가는 값.
LOCAL_NUM_CTX = int(os.environ.get("VICA_LOCAL_NUM_CTX", "4096"))

# 로컬 지시문에 더하는 필수 규칙 5개. OpenAI 지시문(약 3,000토큰)과 똑같이
# 맞추지 않는다 — 작은 모델은 글이 길수록 느려지고 규칙을 놓친다.
LOCAL_PROMPT_RULES = """
[로컬 추가 규칙 — 이 규칙이 위 규칙보다 먼저다]
1. 로봇의 마지막 말이 "안내를 받으시겠어요?"면 답은 affirm 또는 deny 다. 목적지를 말하면 navigate.
2. 로봇의 마지막 말이 "여기서 대기할까요?"·"안내를 마칠까요?" 같은 도착 질문이면
   답은 affirm·deny·wait·finish 만 고른다. 목적지를 다시 제안하지 않는다.
3. 사용자가 시간만 말하면("한 5분?", "십 분") intent 는 wait 다.
4. 목적지 목록에 딱 맞는 곳이 없으면 비슷한 곳을 고르지 말고 clarify 로 답한다.
5. 로봇에게 한 말이 아니면(옆 사람 대화, 혼잣말) intent 는 unknown, reply 는 빈 문자열."""


def enabled() -> bool:
    """스위치가 켜졌는가. 켜도 VICA_INTENT_INPUT=text 일 때만 참이다.

    매번 환경변수를 읽는다 — 시험이 monkeypatch 로 켜고 끌 수 있게.
    """
    on = os.environ.get("VICA_LOCAL_RULES", "").strip().lower() in ("1", "true", "on", "yes")
    text_mode = os.environ.get("VICA_INTENT_INPUT", "text").strip().lower() == "text"
    return on and text_mode


def short_answer_allowed(dialog_state: Optional[str]) -> bool:
    """짧은 대답(네·아니요·그래)을 LLM 없이 즉시 처리해도 되는가.

    dialog_state 를 모르면(옛 미션 메시지·아직 수신 전) 예전처럼 허용한다 —
    스위치를 켰다고 짧은 대답이 통째로 막히면 안 된다.
    """
    if not dialog_state:
        return True
    return dialog_state in ANSWER_WAIT_STATES


def should_clear_history(prev_state: Optional[str], new_state: Optional[str]) -> bool:
    """접근 질문이 새로 시작되면(→ awaiting_user) 앞 사람과의 대화를 비운다(수리안 나)."""
    return new_state == "awaiting_user" and prev_state != "awaiting_user"


class MisheardGate:
    """듣기 창 밖에서 LLM 이 못 알아들은 말(unknown·question)에 대한 대꾸 정책.

    첫 번째는 녹음된 고정 문구(RETRY_PROMPT) 하나만, 같은 상황이 이어지면
    두 번째부터는 침묵한다. 알아들은 말이 한 번 나오거나 "비카야" 호출이 오면
    처음으로 돌아간다. LLM 이 지어낸 대꾸(즉석 합성·엉뚱한 장소)를 내보내지
    않는 것이 목적이다.
    """

    def __init__(self) -> None:
        self._strikes = 0

    def reset(self) -> None:
        self._strikes = 0

    def decide(self, retry_prompt: str) -> Optional[str]:
        """못 알아들은 말 하나에 대해 할 말을 돌려준다. None = 침묵."""
        self._strikes += 1
        return retry_prompt if self._strikes == 1 else None

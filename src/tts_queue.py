"""발화 요청 큐 (우선순위·선점·중복 억제). 음성 모델 없이 검증되는 순수 로직.

/vica/tts_request 로 들어오는 `"{priority}:{text}"` 요청을 담는다. 계약 원본은
vica_ros2_ws 의 mission_logic.Say(priority=...) 다.

우선순위
    emergency  안전 관련. 하던 말을 끊고 즉시 재생한다.
    response   사용자 발화에 대한 답. 대기 중인 안내보다 먼저 나간다.
    narration  상태 안내. 기본값.

같은 우선순위끼리는 들어온 순서(FIFO)를 지킨다.

큐가 밀릴 때 오래된 narration 부터 버리는 이유: 상황이 끝난 뒤에 옛날 안내가
줄줄이 나오면 사용자가 현재 상태를 오해한다. 안내는 신선할 때만 쓸모가 있다.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

EMERGENCY = "emergency"
RESPONSE = "response"
NARRATION = "narration"

# 앞에 올수록 먼저 재생된다.
PRIORITIES = (EMERGENCY, RESPONSE, NARRATION)
_ORDER = {name: index for index, name in enumerate(PRIORITIES)}

DEFAULT_PRIORITY = NARRATION


@dataclass(frozen=True)
class Utterance:
    priority: str
    text: str
    queued_at: float
    seq: int  # 같은 우선순위 안에서 FIFO 를 보장하기 위한 일련번호


@dataclass(frozen=True)
class PushResult:
    accepted: bool
    preempt: bool = False  # 재생 중인 발화를 끊어야 하는가
    reason: str = ""  # 거절 사유 (로그용)


def parse_request(data: str) -> tuple[str, str]:
    """`"{priority}:{text}"` 를 (priority, text) 로 나눈다.

    접두어가 없거나 모르는 값이면 narration 으로 본다. 본문에 콜론이 들어갈 수
    있으므로 한 번만 나누고, 앞부분이 실제 우선순위일 때만 접두어로 인정한다.
    """
    if not data:
        return DEFAULT_PRIORITY, ""

    head, sep, tail = data.partition(":")
    if sep and head.strip() in _ORDER:
        return head.strip(), tail.strip()
    return DEFAULT_PRIORITY, data.strip()


def build_request(priority: str, text: str) -> str:
    """`/vica/tts_request` 로 보낼 문자열을 만든다."""
    if priority not in _ORDER:
        priority = DEFAULT_PRIORITY
    return f"{priority}:{text}"


def request_for_intent(intent) -> Optional[str]:
    """LLM 노드가 이 intent 의 reply 를 직접 말해야 하면 요청 문자열을 돌려준다.

    발화 주체를 나누는 이유: navigate 확정 요청은 Mission Manager 가 게이트를 통과
    시킨 뒤에야 결과를 알 수 있다. LLM 이 먼저 "안내하겠습니다"라고 말해 버리면,
    Mission Manager 가 거절했을 때 사용자는 안내가 시작된 줄 안다.

    - navigate + need_confirm=False → Mission Manager 가 말한다 (여기서는 침묵)
    - navigate + need_confirm=True  → 확인 질문이므로 LLM 이 말한다
    - 그 밖의 intent            → Mission Manager 가 관여하지 않으므로 LLM 이 말한다

    VicaIntent(pydantic)와 ROS 메시지 양쪽에 쓰도록 속성으로만 접근한다.
    """
    reply = (getattr(intent, "reply", "") or "").strip()
    if not reply:
        return None

    if getattr(intent, "intent", "") == "navigate" and not getattr(
        intent, "need_confirm", False
    ):
        return None

    priority = (
        EMERGENCY if getattr(intent, "safety_flag", "") == "emergency" else RESPONSE
    )
    return build_request(priority, reply)


class TtsQueue:
    """발화 대기열. ROS 콜백 스레드가 push 하고 재생 스레드가 pop 한다."""

    def __init__(self, max_len: int = 8, dedup_sec: float = 2.0) -> None:
        self.max_len = max_len
        self.dedup_sec = dedup_sec
        self._items: list[Utterance] = []
        self._seq = 0
        self._recent: dict[str, float] = {}  # 최근 발화 텍스트 -> 시각
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def push(self, priority: str, text: str, now: float) -> PushResult:
        text = (text or "").strip()
        if not text:
            return PushResult(accepted=False, reason="빈 문자열")
        if priority not in _ORDER:
            priority = DEFAULT_PRIORITY

        with self._lock:
            self._forget_old(now)
            if text in self._recent:
                return PushResult(accepted=False, reason="직전과 같은 문장")
            self._recent[text] = now

            self._seq += 1
            item = Utterance(priority=priority, text=text, queued_at=now, seq=self._seq)

            if priority == EMERGENCY:
                # 안전 관련은 대기 중인 일반 발화를 밀어내고 바로 나간다.
                self._items = [i for i in self._items if i.priority == EMERGENCY]
                self._items.append(item)
                self._sort()
                return PushResult(accepted=True, preempt=True)

            self._items.append(item)
            self._sort()
            self._trim()
            return PushResult(accepted=True)

    def pop(self) -> Optional[Utterance]:
        with self._lock:
            if not self._items:
                return None
            return self._items.pop(0)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    # -- 내부 --------------------------------------------------------------

    def _sort(self) -> None:
        self._items.sort(key=lambda i: (_ORDER[i.priority], i.seq))

    def _trim(self) -> None:
        """정원을 넘으면 가장 낮은 우선순위의 가장 오래된 항목부터 버린다."""
        while len(self._items) > self.max_len:
            lowest = max(_ORDER[i.priority] for i in self._items)
            for index, item in enumerate(self._items):
                if _ORDER[item.priority] == lowest:
                    del self._items[index]
                    break

    def _forget_old(self, now: float) -> None:
        self._recent = {
            text: at for text, at in self._recent.items() if now - at < self.dedup_sec
        }

"""발화 요청 큐 (우선순위·선점·중복 억제). 음성 모델 없이 검증되는 순수 로직."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

NARRATION_TTL_SEC = 6.0

EMERGENCY = "emergency"
RESPONSE = "response"
NARRATION = "narration"

PRIORITIES = (EMERGENCY, RESPONSE, NARRATION)
_ORDER = {name: index for index, name in enumerate(PRIORITIES)}

DEFAULT_PRIORITY = NARRATION


@dataclass(frozen=True)
class Utterance:
    priority: str
    text: str
    queued_at: float
    seq: int


@dataclass(frozen=True)
class PushResult:
    accepted: bool
    preempt: bool = False
    reason: str = ""
    dropped: tuple[str, ...] = ()


def parse_request(data: str) -> tuple[str, str]:
    """`"{priority}:{text}"` 를 (priority, text) 로 나눈다."""
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
    """LLM 노드가 이 intent 의 reply 를 직접 말해야 하면 요청 문자열을 돌려준다."""
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
        self._recent: dict[str, float] = {}
        self._expired: list[str] = []
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
                dropped = tuple(i.text for i in self._items if i.priority != EMERGENCY)
                self._items = [i for i in self._items if i.priority == EMERGENCY]
                self._items.append(item)
                self._sort()
                return PushResult(accepted=True, preempt=True, dropped=dropped)

            self._items.append(item)
            self._sort()
            return PushResult(accepted=True, dropped=self._trim())

    def pop(self, now: Optional[float] = None) -> Optional[Utterance]:
        import time as _time

        now = _time.time() if now is None else now
        with self._lock:
            self._prune_expired(now)
            if not self._items:
                return None
            return self._items.pop(0)

    def take_expired(self) -> list[str]:
        """유통기한으로 버린 발화 목록을 꺼낸다 — 호출자가 로그로 남긴다."""
        with self._lock:
            out, self._expired = self._expired, []
            return out

    def _prune_expired(self, now: float) -> None:
        def alive(item: Utterance) -> bool:
            age = now - item.queued_at
            if item.priority == NARRATION:
                return age <= NARRATION_TTL_SEC
            return True

        kept, dropped = [], []
        for item in self._items:
            (kept if alive(item) else dropped).append(item)
        if dropped:
            self._items = kept
            self._expired.extend(i.text for i in dropped)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def drop_pending(self, keep_emergency: bool = True) -> tuple[str, ...]:
        """대기 중 발화를 비운다 (barge-in — 사용자가 말을 시작했다)."""
        with self._lock:
            dropped = tuple(
                i.text for i in self._items
                if not (keep_emergency and i.priority == EMERGENCY)
            )
            self._items = [
                i for i in self._items if keep_emergency and i.priority == EMERGENCY
            ]
            return dropped

    def _sort(self) -> None:
        self._items.sort(key=lambda i: (_ORDER[i.priority], i.seq))

    def _trim(self) -> tuple[str, ...]:
        """정원을 넘으면 가장 낮은 우선순위의 가장 오래된 항목부터 버린다."""
        dropped: list[str] = []
        while len(self._items) > self.max_len:
            lowest = max(_ORDER[i.priority] for i in self._items)
            for index, item in enumerate(self._items):
                if _ORDER[item.priority] == lowest:
                    dropped.append(item.text)
                    del self._items[index]
                    break
        return tuple(dropped)

    def _forget_old(self, now: float) -> None:
        self._recent = {
            text: at for text, at in self._recent.items() if now - at < self.dedup_sec
        }

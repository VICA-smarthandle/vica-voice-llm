"""TTS 단일 큐 + 우선순위 (긴급 > 내레이션 > 응답) — 통합 진행순서 ②.

오디오/ROS 에 의존하지 않는다 (Testing Rules: 핵심 로직은 import 가능한
모듈로 분리해 unit test). 재생 스레드는 ros_tts_node 가 담당한다.

/vica/tts_request 는 std_msgs/String 이므로 우선순위는 접두어로 전달한다:
    "emergency:비상 정지합니다."   → 긴급 (최우선, 하위 큐 비움)
    "narration:이동을 시작합니다." → 내레이션
    "response:안내할 수 없습니다." → 응답 (LLM reply 와 같은 급)
    접두어 없음                    → 내레이션 (기본)

/vica/intent 의 reply 는 항상 RESPONSE 우선순위다.
"""
from __future__ import annotations

import threading
from collections import deque
from enum import IntEnum
from typing import Optional, Tuple


class TtsPriority(IntEnum):
    """값이 작을수록 먼저 재생된다."""

    EMERGENCY = 0
    NARRATION = 1
    RESPONSE = 2


_PREFIX_TO_PRIORITY = {
    "emergency:": TtsPriority.EMERGENCY,
    "narration:": TtsPriority.NARRATION,
    "response:": TtsPriority.RESPONSE,
}


def parse_tts_request(data: str) -> Tuple[TtsPriority, str]:
    """String 페이로드 → (우선순위, 텍스트). 접두어 없으면 내레이션."""
    for prefix, priority in _PREFIX_TO_PRIORITY.items():
        if data.startswith(prefix):
            return priority, data[len(prefix):].strip()
    return TtsPriority.NARRATION, data.strip()


def format_tts_request(text: str, priority: TtsPriority) -> str:
    """발행 측(예: mission_manager)이 쓰는 역변환."""
    prefix = {v: k for k, v in _PREFIX_TO_PRIORITY.items()}[priority]
    return prefix + text


class TtsQueue:
    """우선순위별 FIFO. 스레드 안전.

    - pop() 은 항상 가장 높은 우선순위(EMERGENCY 먼저)의 가장 오래된 항목.
    - 우선순위별 상한 초과 시 그 우선순위의 '가장 오래된' 항목을 드롭
      (낡은 멘트가 밀려서 재생되는 것 방지 — 실패 모드 표 'TTS 겹침/밀림').
    - 긴급 항목이 들어오면 내레이션/응답 큐를 비운다: 긴급 상황에서
      직전 상황의 멘트가 뒤이어 나오면 사용자를 혼란시킨다.
    """

    def __init__(self, max_per_priority: int = 8) -> None:
        self.max_per_priority = max_per_priority
        self._queues = {p: deque() for p in TtsPriority}
        self._lock = threading.Lock()
        self.dropped_total = 0

    def put(self, text: str, priority: TtsPriority) -> None:
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            if priority == TtsPriority.EMERGENCY:
                self.dropped_total += len(self._queues[TtsPriority.NARRATION])
                self.dropped_total += len(self._queues[TtsPriority.RESPONSE])
                self._queues[TtsPriority.NARRATION].clear()
                self._queues[TtsPriority.RESPONSE].clear()
            q = self._queues[priority]
            if len(q) >= self.max_per_priority:
                q.popleft()  # 가장 낡은 것 드롭
                self.dropped_total += 1
            q.append(text)

    def pop(self) -> Optional[Tuple[TtsPriority, str]]:
        with self._lock:
            for priority in TtsPriority:  # EMERGENCY(0) 부터
                q = self._queues[priority]
                if q:
                    return priority, q.popleft()
        return None

    def __len__(self) -> int:
        with self._lock:
            return sum(len(q) for q in self._queues.values())

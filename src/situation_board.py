"""상황판 — 코드가 미션 신호로 **확실히 아는 사실**만 적어 모델에 매번 건넨다."""
from __future__ import annotations

import json
import time
from typing import Callable, Optional

START_EVENTS = {"goal_sent", "goal_accepted"}
ARRIVE_EVENTS = {"goal_succeeded"}
FAIL_EVENTS = {"goal_failed": "실패", "goal_rejected": "거부됨", "goal_canceled": "취소"}
HOME_START = {"return_home_sent"}
HOME_END = {"return_home_succeeded", "return_home_failed", "return_home_canceled"}


def parse_goal_event_name(data: str) -> tuple[Optional[str], str]:
    """/vica_goal_event JSON → (event, 목적지 name). 못 읽으면 (None, "")."""
    try:
        payload = json.loads(data)
    except (TypeError, ValueError):
        return None, ""
    if not isinstance(payload, dict):
        return None, ""
    event = payload.get("event")
    name = payload.get("name") or ""
    return (event if isinstance(event, str) and event else None), str(name)


def _ago(seconds: float) -> str:
    if seconds < 60:
        return "방금"
    if seconds < 3600:
        return f"{int(seconds // 60)}분 전"
    return f"{int(seconds // 3600)}시간 전"


class SituationBoard:
    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self.guiding_to: Optional[str] = None
        self.guiding_since: Optional[float] = None
        self.last_arrived: Optional[str] = None
        self.arrived_at: Optional[float] = None
        self.last_outcome: Optional[str] = None
        self.waiting_minutes: Optional[int] = None
        self.returning_home: bool = False

    def on_goal_event(self, event: Optional[str], name: str = "") -> None:
        if not event:
            return
        now = self._clock()
        if event in START_EVENTS:
            if name:
                self.guiding_to = name
            self.guiding_since = now
            self.waiting_minutes = None
            self.last_outcome = None
            self.returning_home = False
        elif event in ARRIVE_EVENTS:
            self.last_arrived = name or self.guiding_to
            self.arrived_at = now
            self.guiding_to = None
            self.last_outcome = None
        elif event in FAIL_EVENTS:
            self.last_outcome = f"{name or self.guiding_to or '안내'} {FAIL_EVENTS[event]}"
            self.guiding_to = None
        elif event in HOME_START:
            self.returning_home = True
            self.guiding_to = None
        elif event in HOME_END:
            self.returning_home = False
        elif event == "state_idle":
            self.guiding_to = None
            self.returning_home = False

    def on_intent(self, intent) -> None:
        """이 노드가 발행한 의도 중 상황판에 남길 것: wait(대기 요청 분)."""
        if getattr(intent, "intent", "") == "wait":
            minutes = getattr(intent, "wait_minutes", -1)
            self.waiting_minutes = int(minutes) if minutes is not None else -1

    def render(self, awaiting_answer: bool = False) -> str:
        """프롬프트 블록. 적을 사실이 하나도 없으면 빈 문자열."""
        now = self._clock()
        lines = []
        if awaiting_answer:
            lines.append("- 로봇이 방금 질문하고 답을 기다리는 중: 예 (지금 들리는 말은 그 답일 가능성이 높다)")
        if self.guiding_to:
            since = f" ({_ago(now - self.guiding_since)} 출발)" if self.guiding_since else ""
            lines.append(f"- 안내 중: {self.guiding_to}로 이동 중{since}")
        elif self.returning_home:
            lines.append("- 안내 중: 없음 (제자리로 돌아가는 중)")
        else:
            lines.append("- 안내 중: 없음")
        if self.last_arrived:
            lines.append(f"- 마지막 도착: {self.last_arrived} ({_ago(now - (self.arrived_at or now))})")
        if self.last_outcome:
            lines.append(f"- 마지막 안내 결말: {self.last_outcome}")
        if self.waiting_minutes is not None and not self.guiding_to:
            minutes = "시간 미정" if self.waiting_minutes < 0 else f"{self.waiting_minutes}분"
            lines.append(f"- 대기 요청: {minutes} (도착 뒤 사용자가 기다려 달라고 함)")
        if (not self.last_arrived and not self.guiding_to and not self.last_outcome
                and not self.returning_home and not awaiting_answer):
            return ""
        return ("\n[지금 상황] (코드가 미션 신호로 확인한 사실 — 대화 이력이 비어 있어도 이것은 맞다. "
                "\"지금 어디 가?\"·\"아까 어디 갔었지?\"·\"몇 층이야?\"는 이것과 목적지 목록의 위치로 답한다)\n"
                + "\n".join(lines) + "\n")

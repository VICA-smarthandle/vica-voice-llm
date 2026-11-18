"""시설 예약 진행 상태(설계 2026-11-18 5.2·5.3) — 순수 모듈, 한 번에 예약 하나.

값은 LLM 이 도구 인자로 읽어 준다(ReserveCall). 이 모듈은 판단하지 않는다 — 말에 근거가 없는 값만 거르고,
날짜를 계산하고, 겹침·운영 시간을 보고, 모자란 값을 차례로 묻고, 사용자의 승낙 뒤에 저장한다.
문장은 reservation_phrases 의 확정본만 쓴다. "예약이 완료되었습니다"는 저장이 끝난 뒤에만 나온다.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable, Optional, Sequence

from . import reservation_phrases as phrases
from .question_record import OpenQuestion
from .reservation_store import ConflictError, Facility, Reservation, ReservationStore
from .reservation_time import has_date_word, has_span_word, has_time_word, resolve_date, speak_date, valid_hhmm
from .schema import VicaIntent

NOT_YET = "(아직 없음)"
# 진행 중인 예약을 버리는 다른 명령. 못 알아들은 말·질문은 버리지 않는다(잡음 한 번에 예약이 날아가면 안 된다).
_OTHER_COMMANDS = frozenset({"navigate", "pause", "resume", "wait", "finish", "help"})


@dataclass(frozen=True)
class ReserveCall:
    """LLM 이 부른 reserve_facility 의 인자. 빈 값 = 이번에 말하지 않음."""
    facility: str = ""
    date: str = ""
    start_time: str = ""
    end_time: str = ""
    name: str = ""


@dataclass(frozen=True)
class FlowOutput:
    say: str
    kind: str                               # "open" · "yesno" · ""(대답을 기다리지 않는 안내)
    saved: Optional[Reservation] = None


def start_gate(dialog_state: str, question: Optional[OpenQuestion], own_text: str) -> str:
    """새 예약을 받아도 되나. "ok" · "busy"(문장 ⑨) · "mute"(아무 말도 하지 않음)."""
    if dialog_state == "awaiting_user" or dialog_state.startswith("patrol"):
        return "mute"                       # 미션이 넘겨받은 문장을 버리는 상태
    if dialog_state != "idle":
        return "busy"
    if question is None or question.kind == "open" or (own_text and question.text == own_text):
        return "ok"
    return "busy"


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


class ReservationFlow:
    def __init__(self, store: ReservationStore, facilities: Sequence[Facility],
                 clock: Callable[[], datetime] = datetime.now):
        self.store = store
        self._facilities = list(facilities)
        self._clock = clock
        self.drop()

    # ----- 상태 --------------------------------------------------------
    def drop(self) -> None:
        self._active = False
        self._facility: Optional[Facility] = None
        self._day: Optional[date] = None
        self._start = ""
        self._end_said = ""                 # 사용자가 말한 끝 시각. 비면 기본 시간으로 채운다
        self._name = ""
        self._confirming = False
        self._seen = False                  # 넘긴 질문을 미션이 한 번 열었나
        self.asked_text = ""

    @property
    def active(self) -> bool:
        return self._active

    def _end(self) -> str:
        """끝 시각. 말한 값, 없으면 시작 + 시설 기본 시간. 자정을 넘으면 ""."""
        if self._end_said:
            return self._end_said
        if not self._start or self._facility is None:
            return ""
        minutes = int(self._start[:2]) * 60 + int(self._start[3:5]) + self._facility.default_hours * 60
        return f"{minutes // 60:02d}:{minutes % 60:02d}" if minutes <= 24 * 60 - 1 else ""

    def situation_line(self) -> str:
        """지시문의 [지금 상황]에 붙일 한 줄. 상태만 적는다 — 지시는 도구 설명에 있다."""
        if not self._active:
            return ""
        today = self._clock().date()
        parts = (f"시설 {self._facility.name if self._facility else NOT_YET}",
                 f"날짜 {speak_date(self._day, today) if self._day else NOT_YET}",
                 f"시작 {self._start or NOT_YET}", f"끝 {self._end() or NOT_YET}",
                 f"이름 {self._name or NOT_YET}")
        return f'[진행 중인 예약] {" · ".join(parts)} · 방금 물은 것: "{self.asked_text}"'

    # ----- 들어오는 것 -------------------------------------------------
    def on_call(self, call: ReserveCall, text: str, gate: str) -> Optional[FlowOutput]:
        """LLM 이 reserve_facility 를 불렀다. text 는 그 발화의 글자(받아쓴 글자 또는 heard_text)."""
        if not self._active:
            if gate == "mute":
                return None
            if gate != "ok":
                return FlowOutput(phrases.UNAVAILABLE, "")
            self._active = True
        self._merge(call, text)
        return self._advance()

    def on_general(self, intent: VicaIntent) -> Optional[FlowOutput]:
        """진행 중에 일반 해석 결과가 왔다. 승낙·거절만 예약이 받는다."""
        if not self._active:
            return None
        kind = intent.intent
        if kind == "affirm" and self._confirming:
            return self._commit()
        if kind in ("deny", "cancel"):
            self.drop()
            return FlowOutput(phrases.NOT_RESERVED, "")
        if kind in _OTHER_COMMANDS:
            self.drop()
        return None

    def on_question_record(self, text: str) -> None:
        """미션의 질문 기록이 바뀌었다("" = 열린 질문 없음). 내가 넘긴 질문이 열렸다가 사라지면 버린다."""
        if not self._active or not self.asked_text:
            return
        if text == self.asked_text:
            self._seen = True
        elif self._seen:
            self.drop()

    # ----- 안쪽 --------------------------------------------------------
    def _facility_in(self, text: str, wanted: str = "") -> Optional[Facility]:
        """글자에 이름·별칭이 있는 시설. wanted(LLM 값)가 그런 시설이면 그것을 먼저."""
        flat = _squash(text)
        said = [f for f in self._facilities if any(_squash(n) in flat for n in (f.name, *f.aliases))]
        for f in said:
            if wanted and wanted in (f.name, *f.aliases):
                return f
        return said[0] if said else None

    def _merge(self, call: ReserveCall, text: str) -> None:
        """LLM 값을 받는다. 말에 근거가 없는 값만 버린다(설계 5.2-2)."""
        facility = self._facility_in(text, call.facility.strip())
        if facility is not None:
            self._facility = facility
        name = call.name.strip()
        if name and _squash(name) in _squash(text):
            self._name = name
        if call.date and has_date_word(text):
            day = resolve_date(call.date, self._clock().date())
            if day is not None:
                self._day = day
        end = valid_hhmm(call.end_time) if call.end_time and has_span_word(text) else None
        start = valid_hhmm(call.start_time) if call.start_time and has_time_word(text) else None
        if start is not None:
            self._start = start
            self._end_said = ""             # 시작이 바뀌면 앞서 말한 끝은 맞지 않는다
        if end is not None:
            self._end_said = end

    def _ask(self, text: str, kind: str) -> FlowOutput:
        self.asked_text = text
        self._confirming = kind == "yesno"
        self._seen = False                  # 같은 문장을 다시 넘겨도 새 질문이다
        return FlowOutput(text, kind)

    def _clear_time(self, today: date) -> None:
        self._start = self._end_said = ""
        if self._day is not None and self._day < today:
            self._day = None

    def _advance(self) -> FlowOutput:
        now = self._clock()
        if self._facility is None:
            return self._ask(phrases.ASK_WHERE, "open")
        if self._day is None or not self._start:
            return self._ask(phrases.ASK_WHEN, "open")
        end = self._end()
        starts_at = datetime.combine(self._day, datetime.strptime(self._start, "%H:%M").time())
        if (not end or end <= self._start or starts_at <= now
                or self._start < self._facility.open or end > self._facility.close):
            self._clear_time(now.date())
            return self._ask(phrases.BAD_TIME, "open")
        try:
            taken = self.store.conflicts(self._facility.name, self._day.isoformat(), self._start, end)
        except OSError:
            self.drop()
            return FlowOutput(phrases.UNAVAILABLE, "")
        if taken:
            self._clear_time(now.date())
            return self._ask(phrases.TAKEN, "open")
        if not self._name:
            return self._ask(phrases.ASK_NAME, "open")
        return self._ask(phrases.confirm_sentence(self._name, self._day, self._start, end,
                                                  self._facility.name, now.date()), "yesno")

    def _commit(self) -> FlowOutput:
        """사용자가 확인 질문에 승낙했다 — 되읽은 값 그대로 저장한다."""
        now = self._clock()
        try:
            saved = self.store.add(self._facility.name, self._day.isoformat(), self._start, self._end(),
                                   self._name, now)
        except ConflictError:
            self._clear_time(now.date())
            return self._ask(phrases.TAKEN, "open")
        except OSError:
            self.drop()
            return FlowOutput(phrases.UNAVAILABLE, "")     # 저장되지 않은 예약을 완료라고 말하지 않는다
        self.drop()
        return FlowOutput(phrases.DONE, "open", saved=saved)


def to_intent(out: FlowOutput) -> VicaIntent:
    """예약 흐름의 말 → VicaIntent. 질문은 미션이 말하고(ask_text), 안내는 음성이 말한다(reply)."""
    if out.kind:
        return VicaIntent(intent="clarify", need_confirm=False, confidence=1.0, reply="",
                          ask_text=out.say, ask_kind=out.kind)
    return VicaIntent(intent="question", need_confirm=False, confidence=1.0, reply=out.say)


def created_event(res: Reservation, now: datetime) -> str:
    """/vica/reservation/event 에 실을 JSON(설계 5.5)."""
    return json.dumps({"event": "reservation_created", "id": res.id, "facility": res.facility, "date": res.date,
                       "start": res.start, "end": res.end, "name": res.name, "source": res.source,
                       "timestamp": now.isoformat(timespec="seconds")}, ensure_ascii=False)

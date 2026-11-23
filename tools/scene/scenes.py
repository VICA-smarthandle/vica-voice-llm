"""사건(Event) → 장면(Scene) 묶기와 대화 단계 시계열 (설계 2026-11-20 5.1)."""
from __future__ import annotations

import bisect
import datetime as dt
from dataclasses import dataclass

from tools.scene.events import Event

MERGE_S = 2.0
RESERVE_CHAIN_S = 15.0


@dataclass
class Scene:
    id: str
    start: float
    end: float
    trigger: str
    events: list[Event]
    linked_by: str
    question_id: str | None


def state_at(events: list[Event], t: float) -> str:
    """미션 상태 전이 줄(`a -> b`)로 만든 시각 t 의 dialog_state. 전이 전이거나 모르면 ""."""
    cur = ""
    for e in events:
        if e.t > t:
            break
        if e.kind == "state" and e.data.get("via") in ("mission", "handle") and e.data.get("to"):
            cur = e.data["to"]
    return cur


def build_scenes(events: list[Event]) -> list[Scene]:
    evs = sorted(events, key=lambda e: e.t)
    starts: list[list] = []
    start_t = None
    last_user = None
    last_qopen = None
    for i, e in enumerate(evs):
        if e.kind == "user_text":
            last_user = e.t
            continue
        if e.kind not in ("wake_open", "q_open", "listen_reserve"):
            continue
        merge = start_t is not None and e.t - start_t < MERGE_S
        cur_qid = starts[-1][2] if starts else None
        if e.kind == "wake_open":
            new, trig = (start_t is None or not merge), "wake"
        elif e.kind == "q_open":
            qid = e.data.get("id")
            if start_t is not None and merge:
                new = False
                if starts[-1][2] is None:
                    starts[-1][2] = qid
            else:
                new = start_t is None or qid != cur_qid
            trig = "question"
            if not new:
                last_qopen = e.t
        else:
            near_q = last_qopen is not None and 0 <= e.t - last_qopen < MERGE_S
            talked = last_user is not None and e.t - last_user <= RESERVE_CHAIN_S
            new, trig = (start_t is None or not (merge or near_q or talked)), "question"
        if not new:
            continue
        j = i
        if trig == "wake":
            while j > 0 and evs[j - 1].kind == "wake_fire" and e.t - evs[j - 1].t <= MERGE_S \
                    and (not starts or j - 1 > starts[-1][0]):
                j -= 1
        starts.append([j, trig, e.data.get("id") if e.kind == "q_open" else None])
        start_t, last_user = e.t, None
        last_qopen = e.t if e.kind == "q_open" else None
    scenes: list[Scene] = []
    used: dict[str, int] = {}
    for n, (idx, trig, qid) in enumerate(starts):
        stop = starts[n + 1][0] if n + 1 < len(starts) else len(evs)
        chunk = evs[idx:stop]
        start = chunk[0].t
        end = evs[stop].t if stop < len(evs) else chunk[-1].t
        sid = f"{dt.datetime.fromtimestamp(start).strftime('%Y%m%d-%H%M%S')}-{trig}"
        used[sid] = used.get(sid, 0) + 1
        if used[sid] > 1:
            sid = f"{sid}-{used[sid]}"
        scenes.append(Scene(sid, start, end, trig, chunk, "question_id" if qid else "time", qid or None))
    return scenes

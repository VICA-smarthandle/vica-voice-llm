"""사건(Event) → 장면(Scene) 묶기와 대화 단계 시계열 (설계 2026-11-20 5.1).

장면 시작:
- `wake_open` — 항상 새 장면(직전 시작과 MERGE_S 이내면 같은 장면). 바로 앞 MERGE_S 이내의 `wake_fire` 는 장면에 넣는다.
- `q_open`·`listen_reserve`(질문 예약) — 현재 장면에 `user_text` 가 아직 없을 때만 새 장면, 직전 시작과 MERGE_S 이내면 같은 장면.
  (11-17~ 로그에서 같은 질문의 예약과 q_open 은 가까워 한 장면이 된다. 사용자가 이미 말한 장면 안의 재예약은
  같은 대화 흐름이라 자르지 않는다 — 11-04 15:07 '여기까지 안내를 마칠까요?' 2회가 한 장면.)
끝은 다음 시작 직전(마지막 장면은 마지막 사건). 첫 시작 앞의 사건은 버린다.
"""
from __future__ import annotations

import bisect
import datetime as dt
from dataclasses import dataclass

from tools.scene.events import Event

MERGE_S = 2.0
_QUESTION_START = {"q_open", "listen_reserve"}


@dataclass
class Scene:
    id: str
    start: float
    end: float
    trigger: str            # wake | question
    events: list[Event]
    linked_by: str          # question_id | time
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
    starts: list[tuple[int, str]] = []          # (사건 index, trigger)
    last_start_t = None
    has_user = False
    for i, e in enumerate(evs):
        if e.kind == "user_text":
            has_user = True
            continue
        if e.kind == "wake_open":
            new = last_start_t is None or e.t - last_start_t >= MERGE_S
            trig = "wake"
        elif e.kind in _QUESTION_START:
            new = last_start_t is None or (e.t - last_start_t >= MERGE_S and not has_user)
            trig = "question"
        else:
            continue
        if new:
            j = i
            if trig == "wake":                   # 바로 앞의 호출 확정 점수도 이 장면 몫
                while j > 0 and evs[j - 1].kind == "wake_fire" and e.t - evs[j - 1].t <= MERGE_S \
                        and (not starts or j - 1 >= starts[-1][0] + 1):
                    j -= 1
            starts.append((j, trig))
            last_start_t = e.t
            has_user = False
    scenes: list[Scene] = []
    used: dict[str, int] = {}
    for n, (idx, trig) in enumerate(starts):
        stop = starts[n + 1][0] if n + 1 < len(starts) else len(evs)
        chunk = evs[idx:stop]
        start = chunk[0].t
        end = evs[stop].t if stop < len(evs) else chunk[-1].t
        sid = f"{dt.datetime.fromtimestamp(start).strftime('%Y%m%d-%H%M%S')}-{trig}"
        used[sid] = used.get(sid, 0) + 1
        if used[sid] > 1:
            sid = f"{sid}-{used[sid]}"
        qid = next((e.data.get("id") for e in chunk if e.kind in ("q_open", "q_voice", "q_close")
                    and e.data.get("id")), None)
        scenes.append(Scene(sid, start, end, trig, chunk, "question_id" if qid else "time", qid))
    return scenes

"""사건(Event) → 장면(Scene) 묶기와 대화 단계 시계열 (설계 2026-11-20 5.1).

장면 시작 (R8):
- `wake_open` — 항상 새 장면(직전 시작과 MERGE_S 이내면 같은 장면). 바로 앞 MERGE_S 이내의 `wake_fire` 는 장면에 넣는다.
- `q_open` — 현재 장면의 question_id 와 id 가 다르면 새 장면. 장면 시작 사건 MERGE_S 이내면 시작에 합쳐지고 id 를 단다.
- `listen_reserve` — 시작·q_open 근처(MERGE_S)면 같은 장면. q_open 없는 옛 로그에서는 마지막 user_text 뒤 RESERVE_CHAIN_S 이내일 때만
  같은 장면(11-04 15:07 '여기까지 안내를 마칠까요?' 연쇄), 아니면 새 장면.
끝은 다음 시작 직전(마지막 장면은 마지막 사건). 첫 시작 앞의 사건은 버린다.
"""
from __future__ import annotations

import bisect
import datetime as dt
from dataclasses import dataclass

from tools.scene.events import Event

MERGE_S = 2.0
RESERVE_CHAIN_S = 15.0     # q_open 없는 옛 로그에서 마지막 user_text 뒤 이 안의 재예약은 같은 장면


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
    starts: list[list] = []                     # [사건 index, trigger, question_id]
    start_t = None                              # 현재 장면 시작 사건의 시각
    last_user = None                            # 현재 장면의 마지막 user_text 시각
    last_qopen = None                           # 현재 장면의 마지막 q_open 시각
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
                    starts[-1][2] = qid         # 시작에 합쳐진 q_open 의 id
            else:
                new = start_t is None or qid != cur_qid
            trig = "question"
            if not new:
                last_qopen = e.t
        else:                                   # listen_reserve
            near_q = last_qopen is not None and 0 <= e.t - last_qopen < MERGE_S
            talked = last_user is not None and e.t - last_user <= RESERVE_CHAIN_S
            new, trig = (start_t is None or not (merge or near_q or talked)), "question"
        if not new:
            continue
        j = i
        if trig == "wake":                       # 바로 앞의 호출 확정 점수도 이 장면 몫
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

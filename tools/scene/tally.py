"""집계와 분석 파이프라인 (설계 2026-11-20 6절). 정의는 11-09 손 집계(노드 로그 줄 세기)와 같다.

- wake_fire = '호출 점수 확정'. wake_near_over = '호출 점수 놓침' 중 점수 ≥ 그 줄의 문턱. 그중 확정 2.0 s 이내가 after_fire.
- after_wake = '비카야 호출 #N'(창 안 구제 제외)마다 다음 호출 전·25 s 안의 첫 결과:
  user_text=heard · wake-ignored=locate · 빈손 종료+1 s 안 ghost=quiet · 빈손 종료만=silent · 없음=none.
- ghost = listen_ghost 전부. ghost_after_wake = quiet 에 쓰인 ghost, 나머지가 ghost_followup.
"""
from __future__ import annotations

import bisect
import re
from typing import Optional

from src import wakeword_monitor as _wm
from tools.scene.causes import Cause, flagged, judge
from tools.scene.events import Event, read_events
from tools.scene.scenes import Scene, build_scenes

AFTER_WAKE_WINDOW_S = 25.0
GHOST_AFTER_EMPTY_S = 1.0
NEAR_AFTER_FIRE_S = 2.0
STARTUP_LOOKBACK_S = 86400.0         # 분석 구간이 실행 도중에 시작할 때 기동 줄을 찾으려고 더 읽는 길이

_RE_WAKE = re.compile(r"호출 문턱 ([\d.]+)")
_RE_LISTEN = re.compile(r"수음 최소 ([\d.]+)")
_RE_APPROACH = re.compile(r"접근 대답 최소 ([\d.]+)")


def ctx_at(events: list[Event], t: float) -> dict:
    """t 앞의 가장 가까운 `호출 문턱 …` 기동 줄 하나에서 세 문턱을 읽는다. 없으면 wakeword_monitor 코드 기본값.

    approach_min_rms 는 기동 줄에 없으면 None(수음 최소를 따른다)."""
    ctx = {"wake_threshold": _wm.WAKE_THRESHOLD_DEFAULT, "listen_min_rms": _wm.LISTEN_MIN_RMS,
           "approach_min_rms": None}
    starts = [e for e in events if e.kind == "startup" and (e.data.get("text") or e.raw).startswith("호출 문턱 ")]
    i = bisect.bisect_right([e.t for e in starts], t)
    if i:
        text = starts[i - 1].data.get("text") or starts[i - 1].raw
        for pat, key in ((_RE_WAKE, "wake_threshold"), (_RE_LISTEN, "listen_min_rms"), (_RE_APPROACH, "approach_min_rms")):
            m = pat.search(text)
            if m:
                ctx[key] = float(m.group(1))
    return ctx


def tally(events: list[Event], scenes: list[Scene], causes: dict[str, list[Cause]]) -> dict:
    evs = sorted(events, key=lambda e: e.t)
    fires = [e.t for e in evs if e.kind == "wake_fire"]
    near = [e for e in evs if e.kind == "wake_near" and e.data.get("peak", 0.0) >= e.data.get("threshold", 1e9)]
    after_fire = sum(1 for e in near if any(0 <= e.t - f <= NEAR_AFTER_FIRE_S for f in fires))

    ghosts = [e for e in evs if e.kind == "listen_ghost"]
    used_ghost: set[int] = set()                # id(Event) of ghosts counted as quiet
    buckets = {"heard": 0, "locate": 0, "quiet": 0, "silent": 0, "none": 0}
    opens = [(i, e) for i, e in enumerate(evs) if e.kind == "wake_open" and not e.data.get("rescue")]
    for k, (i, w) in enumerate(opens):
        stop_t = min(w.t + AFTER_WAKE_WINDOW_S, opens[k + 1][1].t if k + 1 < len(opens) else float("inf"))
        outcome = "none"
        for e in evs[i + 1:]:
            if e.t > stop_t or (e.kind == "wake_open" and not e.data.get("rescue")):
                break
            if e.kind == "user_text":
                outcome = "heard"
            elif e.kind == "wake_ignored":
                outcome = "locate"
            elif e.kind == "listen_empty_free":
                g = next((x for x in ghosts if 0 <= x.t - e.t <= GHOST_AFTER_EMPTY_S and id(x) not in used_ghost), None)
                if g is not None:
                    used_ghost.add(id(g))
                    outcome = "quiet"
                else:
                    outcome = "silent"
            else:
                continue
            break
        buckets[outcome] += 1

    return {
        "wake_fire": len(fires),
        "wake_near_over": len(near),
        "wake_near_over_after_fire": after_fire,
        "ghost": len(ghosts),
        "ghost_followup": len(ghosts) - len(used_ghost),
        "ghost_after_wake": len(used_ghost),
        "relisten_drop": sum(1 for e in evs if e.kind == "relisten_drop"),
        "after_wake": buckets,
        "overlap": sum(1 for s in scenes if any(c.name == "질문 겹침" for c in causes.get(s.id, []))),
        "flagged": [s.id for s in scenes if flagged(causes.get(s.id, []))],
    }


def analyze(since: float, until: float, journal: bool = True, **read_kw):
    """기간 → (사건, 장면, 장면별 원인, 집계). 문턱은 장면 앞 가장 가까운 기동 줄(기간 앞 하루까지 거슬러 찾는다)."""
    wide = read_events(since - STARTUP_LOOKBACK_S, until, journal=False, **read_kw)
    startups = [e for e in wide if e.kind == "startup"]
    events = [e for e in wide if e.t >= since]
    if journal:
        events = read_events(since, until, journal=True, **read_kw)
    scenes = build_scenes(events)
    fire_ts = [e.t for e in events if e.kind == "wake_fire"]
    causes: dict[str, list[Cause]] = {}
    for s in scenes:
        ctx = ctx_at(startups, s.start)
        ctx["all_events"] = events
        ctx["prev_fire_ts"] = [f for f in fire_ts if s.start - NEAR_AFTER_FIRE_S <= f < s.start]
        causes[s.id] = judge(s, ctx)
    return events, scenes, causes, tally(events, scenes, causes)

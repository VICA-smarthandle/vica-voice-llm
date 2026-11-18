"""6채널 녹음을 운영 감시기에 에코 제거 off·on 으로 두 번 흘려 사건을 견준다(설계 5.2·5.4, 2026-11-18).

시각표(json)는 tools/aec_record.py 가 쓴다. 항목마다 새 감시기(점검기 _play 와 같은 틀)로 돌리고,
"멈춰"·"비카야" 잡힘이 on 에서 나빠지지 않았는지, 로봇 멘트가 6 dB 이상 줄었는지를 판정한다.
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from .echo_cancel import BLOCK, EchoCanceller, EchoConfig, EchoStage
from .mic_check_rules import MicValues
from .mic_check_replay import DT, FLUSH_FRAMES, PAD_FRAMES, Models, _Run

REDUCE_MIN_DB, LATE_MAX_SEC, LEVEL_TOL_DB, LEARN_SKIP_SEC, PRE_ROLL_SEC, TTS_TAIL_SEC = 6.0, 0.08, 1.0, 4.0, 1.0, 0.4


@dataclass(frozen=True)
class Ment:
    name: str
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Item:
    id: str
    kind: str
    say: Optional[str]
    expect: Optional[str]
    start: float
    end: float


@dataclass(frozen=True)
class Session:
    wav: Optional[Path]
    ments: tuple
    items: tuple


@dataclass(frozen=True)
class ItemResult:
    id: str
    wakes: int
    emergencies: int
    first_wake: Optional[float]
    first_emergency: Optional[float]
    peak_a: float
    peak_b: float
    texts: tuple
    check_text: str
    rms: float


def load_session(json_path: Path) -> Session:
    """시각표 json 을 읽는다. wav 는 json 과 같은 폴더 기준, 쓰지 않는 키는 무시한다."""
    json_path = Path(json_path)
    d = json.loads(json_path.read_text(encoding="utf-8"))
    ments = tuple(Ment(m["name"], m["text"], float(m["start"]), float(m["end"])) for m in d["ments"])
    items = tuple(Item(i["id"], i["kind"], i.get("say"), i.get("expect"), float(i["start"]), float(i["end"]))
                  for i in d["items"])
    return Session(json_path.parent / d["wav"], ments, items)


def split_blocks(x6: np.ndarray) -> list:
    """(N, 6) int16 → (1280, 6) 블록들. 남는 꼬리는 버린다."""
    return [x6[i:i + BLOCK] for i in range(0, len(x6) - BLOCK + 1, BLOCK)]


def frames_for(blocks: Sequence[np.ndarray], mode: str, config: EchoConfig, factory=EchoCanceller) -> list:
    """off = 블록마다 ch0 · on = EchoStage 하나로 처음부터 끝까지(운영처럼 학습을 이어 간다)."""
    if mode == "off":
        return [np.ascontiguousarray(b[:, 0]) for b in blocks]
    stage = EchoStage(EchoConfig("on", config.delay_ms, config.tail_ms, config.warnings),
                      warn=lambda msg: print(f"  ({msg})"), factory=factory)
    return [stage.feed(np.ascontiguousarray(b[:, 0]), np.ascontiguousarray(b[:, 5])) for b in blocks]


def _rms(frames: Sequence[np.ndarray]) -> float:
    if not len(frames):
        return 0.0
    x = np.concatenate([np.asarray(f, dtype=np.float64) for f in frames]) / 32768.0
    return float(np.sqrt(np.mean(x * x)))


def _window(frames: Sequence, start: float, end: float) -> list:
    return list(frames[int(start / DT):min(len(frames), math.ceil(end / DT))])


def run_item(models: Models, values: MicValues, frames: Sequence[np.ndarray], item: Item,
             ments: Sequence[Ment]) -> ItemResult:
    """항목 하나를 새 감시기에 흘린다. 앞 PRE_ROLL_SEC 는 먹이되 rms·세기는 항목 구간만 본다."""
    run = _Run(models, values)
    a = max(0, int((item.start - PRE_ROLL_SEC) / DT))
    b = min(len(frames), math.ceil(item.end / DT))
    quiet = np.zeros(BLOCK, dtype=np.int16)
    t, _ = run.feed([quiet] * FLUSH_FRAMES, [None] * FLUSH_FRAMES, a * DT - FLUSH_FRAMES * DT)
    near = [m for m in ments if m.start < b * DT and m.end > a * DT]
    started, stopped = set(), set()
    wakes, first_wake, first_emergency = 0, None, None
    run.counting = True
    for i in range(a, b):
        t = i * DT
        for k, m in enumerate(near):
            if k not in started and t >= m.start:
                run.monitor.note_robot_speech(m.text, now=t)
                run.monitor.set_speaking(True, now=t)
                started.add(k)
            if k in started and k not in stopped and t >= m.end + TTS_TAIL_SEC:
                run.monitor.set_speaking(False, now=t)
                stopped.add(k)
        before = len(run.emergencies)
        _, out = run.feed([frames[i]], [None], t)
        if out[0] == "wake":
            wakes += 1
            if first_wake is None:
                first_wake = t
        if len(run.emergencies) > before and first_emergency is None:
            first_emergency = t
    run.counting = False
    run.feed([quiet] * PAD_FRAMES, [None] * PAD_FRAMES, b * DT)
    check = (run.monitor.last_emergency_check or {}).get("text", "") or ""
    return ItemResult(item.id, wakes, len(run.emergencies), first_wake, first_emergency, run.peak_a, run.peak_b,
                      tuple(run.texts), check, _rms(_window(frames, item.start, item.end)))


def ment_reductions(session: Session, off: Sequence[np.ndarray], on: Sequence[np.ndarray]) -> list:
    """robot_only 항목 안, 항목 시작 LEARN_SKIP_SEC 뒤 멘트마다 20·log10(rms(on)/rms(off)) dB."""
    out = []
    for item in session.items:
        if item.kind != "robot_only":
            continue
        for m in session.ments:
            if item.start + LEARN_SKIP_SEC <= m.start < item.end:
                r_off, r_on = _rms(_window(off, m.start, m.end)), _rms(_window(on, m.start, m.end))
                if r_off > 0 and r_on > 0:
                    out.append(20.0 * math.log10(r_on / r_off))
    return out


def _caught(expect: Optional[str], r: ItemResult) -> bool:
    if expect == "emergency":
        return r.emergencies >= 1
    if expect == "wake":
        return r.wakes >= 1
    return False


def _pairs(session: Session, off: Mapping, on: Mapping, kind: str, expect: Optional[str] = None) -> list:
    return [(i, off[i.id], on[i.id]) for i in session.items
            if i.kind == kind and (expect is None or i.expect == expect) and i.id in off and i.id in on]


def judge(session: Session, off: Mapping, on: Mapping, reductions: Sequence[float]) -> dict:
    """설계 5.4 — on 이 off 보다 나빠지지 않았는지와 로봇 멘트 줄임."""
    stop = _pairs(session, off, on, "overlap", "emergency")
    n_off, n_on = sum(_caught("emergency", f) for _, f, _ in stop), sum(_caught("emergency", n) for _, _, n in stop)
    late = [i.id for i, f, n in stop if f.emergencies and n.emergencies
            and n.first_emergency - f.first_emergency > LATE_MAX_SEC]
    overlap_stop = {"n": len(stop), "off": n_off, "on": n_on, "late": late, "pass": n_on >= n_off and not late}

    wake = _pairs(session, off, on, "overlap", "wake")
    w_off, w_on = sum(_caught("wake", f) for _, f, _ in wake), sum(_caught("wake", n) for _, _, n in wake)
    overlap_wake = {"n": len(wake), "off": w_off, "on": w_on, "pass": w_on >= w_off}

    user = _pairs(session, off, on, "user_only")
    differs = [i.id for i, f, n in user if _caught(i.expect, f) != _caught(i.expect, n) or f.texts != n.texts]
    levels = [abs(20.0 * math.log10(n.rms / f.rms)) for _, f, n in user if f.rms > 0 and n.rms > 0]
    level_max = max(levels) if levels else 0.0
    user_only = {"n": len(user), "differs": differs, "level_db_max": level_max,
                 "pass": not differs and level_max <= LEVEL_TOL_DB}

    robot = _pairs(session, off, on, "robot_only")
    new_events = sum((n.wakes + n.emergencies) - (f.wakes + f.emergencies) for _, f, n in robot)
    median = statistics.median(reductions) if reductions else None
    robot_only = {"reduce_db_median": median, "new_events": new_events,
                  "pass": median is not None and median <= -REDUCE_MIN_DB and new_events <= 0}

    parts = (overlap_stop, overlap_wake, user_only, robot_only)
    return {"overlap_stop": overlap_stop, "overlap_wake": overlap_wake, "user_only": user_only,
            "robot_only": robot_only, "pass": all(p["pass"] or p.get("n") == 0 for p in parts)}


def _ok(flag: bool) -> str:
    return "통과" if flag else "실패"


def format_report(report: dict) -> str:
    s, w, u, r = report["overlap_stop"], report["overlap_wake"], report["user_only"], report["robot_only"]
    median = r["reduce_db_median"]
    lines = [
        "묶음            수   off   on   판정",
        f"겹침 멈춰     {s['n']:4d} {s['off']:5d} {s['on']:4d}   {_ok(s['pass'])}",
        f"겹침 비카야   {w['n']:4d} {w['off']:5d} {w['on']:4d}   {_ok(w['pass'])}",
        f"사람만        {u['n']:4d}                {_ok(u['pass'])}  (소리 크기 차 최대 {u['level_db_max']:.2f} dB)",
        f"로봇만 줄임 중앙값 {'없음' if median is None else f'{median:.1f} dB'} · 새 사건 {r['new_events']}건   {_ok(r['pass'])}",
        f"늦은 항목: {', '.join(s['late']) or '없음'}",
        f"달라진 항목: {', '.join(u['differs']) or '없음'}",
        f"종합: {_ok(report['pass'])}",
    ]
    return "\n".join(lines)

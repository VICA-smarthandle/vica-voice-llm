"""python -m tools.scene report --since "2026-11-04 00:00" --until "2026-11-08 00:00" [--no-journal]"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time

from tools.scene.causes import flagged
from tools.scene.tally import analyze


def _ts(s: str) -> float:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(s, fmt).timestamp()
        except ValueError:
            pass
    raise SystemExit(f"시각 형식 오류: {s!r} (예: '2026-11-04 00:00')")


def report(since: float, until: float, journal: bool = True) -> str:
    t0 = time.time()
    events, scenes, causes, r = analyze(since, until, journal=journal)
    aw = r["after_wake"]
    out = [f"사건 {len(events)} · 장면 {len(scenes)} · 분석 {time.time() - t0:.1f} s",
           f"wake_fire {r['wake_fire']} · wake_near_over {r['wake_near_over']} (after_fire {r['wake_near_over_after_fire']})",
           f"ghost {r['ghost']} (followup {r['ghost_followup']} · after_wake {r['ghost_after_wake']}) · relisten_drop {r['relisten_drop']}",
           "after_wake " + " / ".join(f"{k} {aw[k]}" for k in ("heard", "locate", "quiet", "silent", "none")),
           f"overlap {r['overlap']}",
           f"flagged {len(r['flagged'])}:"]
    for s in scenes:
        cs = causes[s.id]
        if flagged(cs):
            t = dt.datetime.fromtimestamp(s.start).strftime("%m-%d %H:%M:%S")
            out.append(f"  {t}  {s.id}  " + " · ".join(f"{c.name}[{c.status}]" for c in cs if c.status != "정상"))
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.scene")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("report", help="기간의 집계와 의심 장면 목록")
    rp.add_argument("--since", required=True)
    rp.add_argument("--until", required=True)
    rp.add_argument("--no-journal", action="store_true")
    a = ap.parse_args(argv)
    print(report(_ts(a.since), _ts(a.until), journal=not a.no_journal))
    return 0


if __name__ == "__main__":
    sys.exit(main())

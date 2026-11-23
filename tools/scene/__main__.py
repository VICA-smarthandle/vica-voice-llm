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


def reask_child(scene_json: str, path: str) -> int:
    """서버가 요청마다 새 프로세스로 부른다(R12) — 해석기는 import 때 환경값을 읽으므로 경로마다 새로 띄운다."""
    import json
    from dataclasses import asdict

    from tools.scene import expected
    from tools.scene.events import Event
    from tools.scene.scenes import Scene

    d = json.loads(open(scene_json, encoding="utf-8").read())
    def mk(e):
        return Event(e["t"], e["node"], e["kind"], e.get("data", {}), e.get("raw", ""))
    sc = d["scene"]
    scene = Scene(sc["id"], sc["start"], sc["end"], sc["trigger"], [mk(e) for e in sc["events"]],
                  sc.get("linked_by", "time"), sc.get("question_id"))
    events = [mk(e) for e in d["events"]]
    try:
        if path == "rule":
            ut = next((e for e in scene.events if e.kind == "user_text"), None)
            dests = expected.destinations_for_scene(events, ut.t) if ut else []
            res = expected.rule_expected(scene, dests, events)
            out = asdict(res) if res else None
        else:
            out = asdict(expected.reask(scene, path, events=events))
    except RuntimeError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 3
    print(json.dumps(out, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.scene")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("report", help="기간의 집계와 의심 장면 목록")
    rp.add_argument("--since", required=True)
    rp.add_argument("--until", required=True)
    rp.add_argument("--no-journal", action="store_true")
    sp = sub.add_parser("serve", help="브라우저 화면(포트 8772)")
    sp.add_argument("--port", type=int, default=8772)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--allow-host", action="append", default=[], help="Host 헤더로 허용할 주소(예: Tailscale IP). 여러 번 가능")
    rk = sub.add_parser("reask", help="(서버용) 한 장면을 LLM·규칙에 다시 넣기")
    rk.add_argument("--scene-json", required=True)
    rk.add_argument("--path", required=True, choices=("rule", "midm", "gpt", "realtime"))
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        from tools.scene.server import serve
        serve(a.host, a.port, tuple(a.allow_host))
        return 0
    if a.cmd == "reask":
        return reask_child(a.scene_json, a.path)
    print(report(_ts(a.since), _ts(a.until), journal=not a.no_journal))
    return 0


if __name__ == "__main__":
    sys.exit(main())

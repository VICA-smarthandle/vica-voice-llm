"""계측 세션 요약 보고서 — logs/sim/<세션>.jsonl 을 표로 만든다.

실행:
    python tools/metrics_report.py logs/sim/sim_20260729_1930.jsonl
"""
from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import SpanTracker  # noqa: E402


def pctl(vals, p):
    if not vals:
        return None
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(len(vals) * p / 100))]


def fmt(v):
    return f"{v:.2f}s" if isinstance(v, (int, float)) else "—"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = Path(sys.argv[1])
    tracker = SpanTracker()
    sys_samples = []
    last_t = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        last_t = max(last_t, rec["t"])
        if rec["type"] == "event":
            tracker.feed(rec["kind"], rec["t"], rec.get("detail", ""))
        elif rec["type"] == "sys":
            sys_samples.append(rec)
    tracker.finalize(last_t)

    print(f"=== 세션 {path.stem} — 상호작용 {len(tracker.interactions)}건 / "
          f"긴급 {len(tracker.emergencies)}건 ===\n")

    print("── 호출 상호작용 (사용자 체감) ──")
    print(f"{'시각':<9}{'청취+STT':>10}{'LLM':>8}{'응답까지':>10}{'재생':>8}  발화")
    for it in tracker.interactions:
        stamp = datetime.fromtimestamp(it["t"]).strftime("%H:%M:%S")
        if it.get("wake_silent"):
            print(f"{stamp:<9}{'(무응답 — 호출 오탐 후보)':>20}")
            continue
        print(f"{stamp:<9}{fmt(it.get('listen_stt_sec')):>10}"
              f"{fmt(it.get('llm_sec')):>8}{fmt(it.get('response_sec')):>10}"
              f"{fmt(it.get('tts_play_sec')):>8}  {it.get('text', '')!r}")

    def agg(key, label):
        vals = [it[key] for it in tracker.interactions if key in it]
        if vals:
            print(f"  {label:<14} p50 {pctl(vals, 50):.2f}s / p95 {pctl(vals, 95):.2f}s"
                  f" / 최대 {max(vals):.2f}s  (n={len(vals)})")

    print("\n── 단계별 요약 ──")
    agg("listen_stt_sec", "청취+STT")
    agg("llm_sec", "LLM")
    agg("tts_wait_sec", "TTS 큐 대기")
    agg("response_sec", "체감 응답")
    agg("tts_play_sec", "TTS 재생")
    silent = sum(1 for it in tracker.interactions if it.get("wake_silent"))
    if silent:
        print(f"  무응답 호출(오탐 후보): {silent}건")

    if tracker.emergencies:
        print("\n── 긴급 반응 (발행→가상 정지) ──")
        for e in tracker.emergencies:
            stamp = datetime.fromtimestamp(e["t"]).strftime("%H:%M:%S")
            print(f"  {stamp}  '{e['keyword']}'  {e['react_sec']*1000:.0f}ms")

    if sys_samples:
        print("\n── 시스템 사용량 ──")
        for key, label, unit in (("cpu_pct", "CPU", "%"), ("mem_pct", "RAM", "%"),
                                 ("gpu_pct", "GPU", "%"), ("gpu_mem_mb", "GPU 메모리", "MB")):
            vals = [s[key] for s in sys_samples if key in s]
            if vals:
                print(f"  {label:<9} 평균 {statistics.mean(vals):.0f}{unit}"
                      f" / 최대 {max(vals):.0f}{unit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

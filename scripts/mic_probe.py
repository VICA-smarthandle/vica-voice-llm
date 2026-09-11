#!/usr/bin/env python3
"""마이크 실황판 — 칩(XVF3000)이 판정하는 VAD·DOA 를 실시간으로 보여준다."""
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dsp_state import DspState

POLL_SEC = 0.125

dsp = DspState()
if not dsp.available:
    raise SystemExit("reSpeaker 제어 통로를 못 열었다 — USB 뽑았다 꽂기 후 재시도")

duration = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
print(f"{duration:.0f}초 관찰 시작 (Ctrl+C 로 조기 종료)")
print("  두 신호 동시 표시: S=SPEECHDETECTED(잡음억제 쪽, 현재 척추) · "
      "V=VOICEACTIVITY(AEC 후단, 미사용 후보) — ●=발동 ─=꺼짐")

frames = 0
active = 0
doa_hist: Counter = Counter()
fails = 0
runs: list[float] = []
run_start: float | None = None
active_v = 0
both = 0
t_end = time.time() + duration
prev_line = ""
try:
    while time.time() < t_end:
        vad = dsp.speech_detected()
        va = dsp.voice_activity()
        doa = dsp.doa_angle()
        if vad is None and doa is None:
            fails += 1
            if fails >= 3:
                print("\n제어 읽기 3연속 실패 — 통로가 엉켰다. USB 뽑았다 꽂기.")
                break
        else:
            fails = 0
            frames += 1
            now_t = time.time()
            if vad:
                active += 1
                doa_hist[(doa // 10) * 10 if doa is not None else -1] += 1
                if run_start is None:
                    run_start = now_t
            elif run_start is not None:
                runs.append(now_t - run_start)
                run_start = None
            if va:
                active_v += 1
            if va and vad:
                both += 1
            mark = "●" if vad else "─"
            mark_v = "●" if va else "─"
            line = f"  S{mark} V{mark_v}  DOA {doa if doa is not None else '?':>3}°"
            if line != prev_line:
                print(f"[{time.strftime('%H:%M:%S')}] {line}", flush=True)
                prev_line = line
        time.sleep(POLL_SEC)
except KeyboardInterrupt:
    pass

if run_start is not None:
    runs.append(time.time() - run_start)

print("\n===== 요약 =====")
if frames:
    print(f"관찰 {frames}프레임 · S(SPEECHDETECTED) {active}회 ({active/frames:.0%}) · "
          f"V(VOICEACTIVITY) {active_v}회 ({active_v/frames:.0%}) · 동시 {both}회")
    for bucket, n in doa_hist.most_common(5):
        label = f"{bucket}~{bucket+9}°" if bucket >= 0 else "방향?"
        print(f"  말소리 중 방향 {label}: {n}회")
else:
    print("표본 없음")
if runs:
    runs.sort()
    mid = runs[len(runs) // 2]
    print(f"● 연속 구간 {len(runs)}개 · 최단 {runs[0]:.2f}s · "
          f"중앙값 {mid:.2f}s · 최장 {runs[-1]:.2f}s")
    print("  구간 목록:", " ".join(f"{r:.2f}" for r in runs))

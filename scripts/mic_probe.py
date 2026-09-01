#!/usr/bin/env python3
"""마이크 실황판 — 칩(XVF3000)이 판정하는 VAD·DOA 를 실시간으로 보여준다.

"마이크 층에서 오인식을 어디까지 잡을 수 있나"를 눈으로 확인하는 도구
(2026-09-01). 웨이크워드 노드와 같은 USB 제어 읽기를 쓰므로 함께 돌아도
되지만, 제어 통로가 민감하니 폴링은 느슨하게(8Hz)·짧게 쓴다.

시험 시나리오:
  ① 에코 시험 — 아무도 말하지 않고 로봇만 말하게 두고 관찰.
     VAD ● 가 뜨면: AEC 잔여가 칩에게 '사람 말'로 보인다는 물증이고,
     그때의 DOA 가 에코의 방향이다 (barge-in 자책골의 원인 방향).
  ② 옆사람 시험 — 옆 위치(90°/270° 부근)에서 말해 DOA 분리를 확인.
  ③ 핸들 시험 — 핸들 위치에서 작게/보통/크게 말해 VAD 안정성 확인.
  음량(rms)은 이 도구로 못 본다(마이크 스트림은 노드 점유) — 발화별
  rms 는 웨이크워드 로그의 "수음 품질" 줄로 확인한다.

사용:
    .venv/bin/python scripts/mic_probe.py            # 60초
    .venv/bin/python scripts/mic_probe.py 180        # 180초
"""
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dsp_state import DspState

POLL_SEC = 0.125   # 8Hz — 노드(12.5Hz)보다 느슨하게

dsp = DspState()
if not dsp.available:
    raise SystemExit("reSpeaker 제어 통로를 못 열었다 — USB 뽑았다 꽂기 후 재시도")

duration = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
print(f"{duration:.0f}초 관찰 시작 (Ctrl+C 로 조기 종료) — ● = 말소리 판정")

frames = 0
active = 0
doa_hist: Counter = Counter()
fails = 0
t_end = time.time() + duration
prev_line = ""
try:
    while time.time() < t_end:
        vad = dsp.speech_detected()
        doa = dsp.doa_angle()
        if vad is None and doa is None:
            fails += 1
            if fails >= 3:
                print("\n제어 읽기 3연속 실패 — 통로가 엉켰다. USB 뽑았다 꽂기.")
                break
        else:
            fails = 0
            frames += 1
            if vad:
                active += 1
                doa_hist[(doa // 10) * 10 if doa is not None else -1] += 1
            mark = "●" if vad else "─"
            line = f"  {mark}  DOA {doa if doa is not None else '?':>3}°"
            if line != prev_line:
                print(f"[{time.strftime('%H:%M:%S')}] {line}", flush=True)
                prev_line = line
        time.sleep(POLL_SEC)
except KeyboardInterrupt:
    pass

print("\n===== 요약 =====")
if frames:
    print(f"관찰 {frames}프레임 · 말소리 판정 {active}프레임 ({active/frames:.0%})")
    for bucket, n in doa_hist.most_common(5):
        label = f"{bucket}~{bucket+9}°" if bucket >= 0 else "방향?"
        print(f"  말소리 중 방향 {label}: {n}회")
else:
    print("표본 없음")

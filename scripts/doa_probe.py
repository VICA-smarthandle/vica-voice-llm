#!/usr/bin/env python3
"""핸들 방향 DOA 측정 — barge-in 장착 부채꼴(VICA_USER_DOA_CENTER) 설정용.

사용법: 핸들 위치(사용자가 서는 자리)에 서서 실행하고, 15초 동안
아무 말이나 계속 한다 (예: 하나 둘 셋 …). 끝나면 중앙값을 알려준다.

    .venv/bin/python scripts/doa_probe.py

⚠️ 웨이크워드 노드가 마이크를 잡고 있으면 USB 제어는 공유되므로 함께
돌아도 되지만, 조용한 환경에서 혼자 말하는 것이 정확하다.
"""
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dsp_state import DspState

dsp = DspState()
if not dsp.available:
    raise SystemExit("reSpeaker 를 찾지 못했다 — USB 연결을 확인할 것")

print("15초 동안 핸들 위치에서 계속 말하세요 (하나, 둘, 셋 …)")
samples = []
t_end = time.time() + 15
while time.time() < t_end:
    vad = dsp.speech_detected()
    doa = dsp.doa_angle()
    if vad and doa is not None:
        samples.append(doa)
        print(f"  말 감지 · DOA {doa:3d}°")
    time.sleep(0.25)
dsp.close()

if len(samples) < 8:
    raise SystemExit(f"표본 부족({len(samples)}개) — 더 크게, 더 오래 말하며 다시")
med = statistics.median(samples)
spread = max(samples) - min(samples)
print(f"\n표본 {len(samples)}개 · 중앙값 {med:.0f}° · 범위 {min(samples)}~{max(samples)}°")
print(f"\n.env 에 추가할 것:\n  VICA_USER_DOA_CENTER={med:.0f}")
if spread > 60:
    print("⚠️ 산포가 크다 — 주변 소음이 섞였을 수 있으니 조용할 때 다시 재볼 것")

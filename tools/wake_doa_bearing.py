"""DOA 부호 실측 — 마이크 각도가 반시계로 커지는가 시계로 커지는가.

호출 접근 설계(루트 저장소 docs/superpowers/specs/2026-09-10-voice-call-approach-design.md)
§5 의 선행 측정이다. SpinInPlace 는 양수 = 반시계다. 마이크가 반시계로
커지면 s=+1, 시계로 커지면 s=-1 이고, 틀리면 로봇이 반대로 돈다.

세 자리에서 각각 "비카야"를 부른다 (로봇은 정지):
    ① 정면      기준선 재확인 (0° 부근이어야 한다)
    ② 왼쪽 90°  약 90° -> s=+1 / 약 270° -> s=-1
    ③ 오른쪽 90° ②의 반대편이 나오는지 확인

사용:
    cd ~/VICA-smarthandle/vica-voice-llm
    .venv/bin/python -u -m tools.wake_doa_bearing
"""
from __future__ import annotations

import math
import sys
import time

sys.path.insert(0, ".")

from src.dsp_state import DspState  # noqa: E402

POLL_HZ = 20
LISTEN_SEC = 6.0
POSITIONS = [
    ("① 로봇 정면", "0° 부근이 나와야 한다 (기준선 재확인)"),
    ("② 로봇 왼쪽 90°", "약 90° 면 s=+1(반시계), 약 270° 면 s=-1(시계)"),
    ("③ 로봇 오른쪽 90°", "②의 반대편이어야 한다"),
]


def circular_mean(angles: list) -> float:
    """0/359 경계를 올바로 다루는 원형 평균(도)."""
    if not angles:
        return float("nan")
    x = sum(math.cos(math.radians(a)) for a in angles) / len(angles)
    y = sum(math.sin(math.radians(a)) for a in angles) / len(angles)
    return math.degrees(math.atan2(y, x)) % 360


def listen_once(dsp: DspState) -> tuple:
    """LISTEN_SEC 동안 발화 판정 프레임의 방향만 모은다."""
    angles: list = []
    t_end = time.time() + LISTEN_SEC
    while time.time() < t_end:
        if dsp.speech_detected():
            a = dsp.doa_angle()
            if a is not None:
                angles.append(float(a))
        time.sleep(1.0 / POLL_HZ)
    return circular_mean(angles), len(angles)


def main() -> None:
    dsp = DspState()
    if not dsp.available:
        raise SystemExit("칩 상태를 읽을 수 없다 (udev/장치 확인)")

    print("로봇을 정지시킨 상태로 진행한다. 각 자리에서 엔터 뒤 계속 말한다.\n")
    results = []
    for label, hint in POSITIONS:
        input(f"{label} 에 서서 엔터를 누르고 {LISTEN_SEC:.0f}초간 '비카야'를 반복하세요 > ")
        mean, n = listen_once(dsp)
        print(f"   -> 평균 {mean:6.1f}°  (발화 표본 {n})   {hint}\n")
        results.append((label, mean, n))
    dsp.close()

    print("===== 결과 =====")
    for label, mean, n in results:
        print(f"  {label:16s} {mean:6.1f}°  (표본 {n})")
    left = results[1][1]
    if results[1][2] < 10:
        print("\n판정 불가 — 왼쪽 표본이 너무 적다. 더 크게/가까이 말해 재시험.")
        return
    ccw = min(abs(left - 90.0), 360 - abs(left - 90.0))
    cw = min(abs(left - 270.0), 360 - abs(left - 270.0))
    if ccw < cw:
        print(f"\n판정: 마이크는 **반시계로 증가** -> wake_doa_sign = +1.0")
    else:
        print(f"\n판정: 마이크는 **시계로 증가** -> wake_doa_sign = -1.0")


if __name__ == "__main__":
    main()

"""DOA(소리 방향) 실측 — "사용자와 옆사람을 방향으로 가릴 수 있는가".

barge-in 재설계 2단계의 전제 측정이다. 칩 VAD(SPEECHDETECTED)는 메아리를
완벽히 거르지만(vad_probe: 로봇 단독 0.0%) 화자는 못 가린다 — 옆사람 대화가
질문을 끊는 것이 실측됐다(2026-08-24). DOAANGLE 이 두 화자 위치를 안정적으로
구분하면, 사용자 방향 부채꼴 안의 발화만 대답으로 인정한다.

2구간 (사이에 삑):
    A 15초 — 로봇이 말하는 동안 **사용자**가 평소 위치에서 계속 말한다
    B 15초 — 로봇이 말하는 동안 **옆사람**이 자기 위치에서 계속 말한다
              (사용자는 조용)

발화 판정(SPEECHDETECTED=1) 프레임의 방향 분포를 비교해 부채꼴 후보를 낸다.

사용:
    HF_HUB_OFFLINE=1 .venv/bin/python -u -m tools.doa_probe
"""
from __future__ import annotations

import math
import sys
import threading
import time

sys.path.insert(0, ".")

from src import audio_cue, audio_out  # noqa: E402
from src.dsp_state import DspState  # noqa: E402

POLL_HZ = 20
PHASE_SEC = 15.0

ANNOUNCEMENT = [
    "지금부터 안내를 시작하겠습니다. 좌측에 계단이 있으니 손잡이를 잡아 주세요.",
    "복도를 따라 직진하고 있습니다. 전방에 회전 구간이 있어 속도를 줄이겠습니다.",
]


def circular_stats(angles):
    """각도 목록의 원형 평균과 퍼짐(도). 0/359 경계를 올바로 다룬다."""
    if not angles:
        return float("nan"), float("nan")
    x = sum(math.cos(math.radians(a)) for a in angles) / len(angles)
    y = sum(math.sin(math.radians(a)) for a in angles) / len(angles)
    mean = math.degrees(math.atan2(y, x)) % 360
    spread = math.degrees(math.sqrt(max(0.0, -2 * math.log(max(1e-9, math.hypot(x, y))))))
    return mean, spread


def ang_diff(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def main() -> None:
    dsp = DspState()
    if not dsp.available:
        raise SystemExit("칩 상태를 읽을 수 없다 (udev/장치 확인)")

    from src.tts import VicaTTS

    print("TTS 합성 중...")
    tts = VicaTTS()
    clips = [tts._synthesize(text) for text in ANNOUNCEMENT]

    import numpy as np

    # 상시 출력 스트림 예열 — 폴링 시작 전에 연다 (USB 제어 충돌 회피)
    audio_out.play(np.zeros(320, dtype=np.float32), 16000, blocking=True)

    samples: list[dict] = []
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            samples.append({
                "t": time.time(),
                "speech": dsp.speech_detected(),
                "doa": dsp.doa_angle(),
            })
            time.sleep(1.0 / POLL_HZ)

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()

    def speak_for(seconds: float) -> None:
        t_end = time.time() + seconds
        i = 0
        while time.time() < t_end:
            wav, rate = clips[i % len(clips)]
            i += 1
            audio_out.play(wav, rate, blocking=True)

    print(f"\nA구간 {PHASE_SEC:.0f}초 — 로봇이 말하는 동안 '사용자'가 평소 위치에서 계속 말하세요")
    a0 = time.time()
    speak_for(PHASE_SEC)
    a1 = time.time()

    audio_out.play(audio_cue.arrived(), audio_cue.SAMPLE_RATE, blocking=True)
    time.sleep(1.0)
    print(f"B구간 {PHASE_SEC:.0f}초 — 삑 후, '옆사람'이 자기 위치에서 계속 말하세요 (사용자는 조용)")
    b0 = time.time()
    speak_for(PHASE_SEC)
    b1 = time.time()

    stop.set()
    poller.join(timeout=3)
    dsp.close()

    def phase_angles(t0, t1):
        return [s["doa"] for s in samples
                if t0 <= s["t"] < t1 and s["speech"] and s["doa"] is not None]

    a_angles = phase_angles(a0, a1)
    b_angles = phase_angles(b0, b1)
    a_mean, a_spread = circular_stats(a_angles)
    b_mean, b_spread = circular_stats(b_angles)

    print("\n===== 결과 (발화 판정 프레임의 방향) =====")
    print(f"  A 사용자 : 평균 {a_mean:6.1f}° 퍼짐 {a_spread:5.1f}° (표본 {len(a_angles)})")
    print(f"  B 옆사람 : 평균 {b_mean:6.1f}° 퍼짐 {b_spread:5.1f}° (표본 {len(b_angles)})")
    if a_angles and b_angles:
        sep = ang_diff(a_mean, b_mean)
        print(f"  두 위치의 각도 차: {sep:.1f}°")
        width = max(30.0, a_spread * 2 + 10)
        if sep > width + b_spread:
            print(f"판정: 구분 가능 — 사용자 부채꼴 후보: 중심 {a_mean:.0f}° ± {width:.0f}°")
            print(f"       (VICA_USER_DOA_CENTER={a_mean:.0f} VICA_USER_DOA_WIDTH={width:.0f})")
        else:
            print("판정: 구분 애매 — 퍼짐이 크거나 각도 차가 작다. 위치·거리 재시험 필요")
    else:
        print("판정 불가 — 발화 판정 표본 부족 (더 크게/가까이 말해 재시험)")


if __name__ == "__main__":
    main()

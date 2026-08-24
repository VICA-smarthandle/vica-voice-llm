"""AEC 실효성 측정 — reSpeaker 로 소리를 틀면서 동시에 6채널을 녹음한다.

원리: 스피커(3.5mm, reSpeaker 재생 경로) 소리는 원음 채널(ch1~4)에는 그대로
들어오고, 처리음 채널(ch0)에서는 AEC 가 뺀 잔여만 남는다. 두 채널의 RMS 차이
(dB)가 억제량이다. 이 수치가 곧 "뮤트를 꺼도 되는가"의 판단 근거다
(docs/jetson-handoff.md 우선순위 5, integration-design.md 8절 D6).

⚠️ 이 도구는 **배선 점검용**이다. 레벨(dB) 지표로 "자가 각성이 안 난다"를
판정하지 말 것 — AGC 가 지운 만큼 되살리고, 웨이크워드 모델은 크기가 아니라
패턴을 본다 (devlog/2026-08-13-자가각성-AEC측정: 82% 억제에도 9초당 1회 발동).
최종 판정은 발동 시험(tools/mute_off_trial, mute_off_positive)으로 한다.

주의 — ch0 에는 AGC(자동 증폭)가 걸려 있어 원음과의 절대 비교는 과소평가된다.
그래서 두 지표를 같이 본다:
    수렴량: 재생 첫 1초 대비 마지막 1초의 ch0 잔여음 감소(dB). AEC 는 적응
            필터라 켜진 직후엔 못 빼다가 몇 초 안에 수렴한다. 이 감소가
            뚜렷하면(≥6dB) AEC 가 일하고 있는 것이다.
    억제량: 수렴 후(마지막 1초) 원음 대비 처리음 비율. AGC 때문에 보수적 수치.
    ch1 peak > 0.9  스피커가 너무 큼(클리핑) — AEC 성능 저하, 볼륨 낮출 것

사용:
    .venv/bin/python -m tools.aec_probe            # 기본 6초 처프
    .venv/bin/python -m tools.aec_probe --sec 10
"""
from __future__ import annotations

import argparse
import queue
import sys

import numpy as np

sys.path.insert(0, ".")  # .venv/bin/python -m tools.aec_probe 실행용

from src import audio_out  # noqa: E402

SAMPLE_RATE = 16000
CHANNELS = 6
BASELINE_SEC = 1.0  # 재생 전 소음 바닥 측정


def make_probe_signal(seconds: float) -> np.ndarray:
    """음성 대역(200~3400Hz) 잡음 + 음절 리듬(4Hz) 진폭 변조.

    처프를 쓰지 않는 이유: 적응 필터(AEC)는 주파수 성분별로 수렴하는데,
    처프는 각 주파수를 스치듯 한 번만 지나가 학습할 시간을 주지 않는다
    (실측 2026-08-24: 처프 6초에 수렴 0dB). 잡음은 전 대역을 계속 자극한다.
    """
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    from scipy.signal import butter, sosfilt

    rng = np.random.default_rng(7)
    sos = butter(4, [200, 3400], btype="band", fs=SAMPLE_RATE, output="sos")
    wave = sosfilt(sos, rng.standard_normal(n))
    wave *= 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t)   # 음성 유사 리듬
    wave = 0.5 * wave / np.max(np.abs(wave))
    fade = int(0.05 * SAMPLE_RATE)
    wave[:fade] *= np.linspace(0, 1, fade)
    wave[-fade:] *= np.linspace(1, 0, fade)
    return wave.astype(np.float32)


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def db(ratio: float) -> float:
    if ratio <= 0:
        return float("-inf")
    return 20.0 * float(np.log10(ratio))


def find_input_device() -> int:
    import sounddevice as sd

    for i, d in enumerate(sd.query_devices()):
        if "respeaker" in d["name"].lower() and d["max_input_channels"] >= CHANNELS:
            return i
    raise SystemExit("reSpeaker 6채널 입력 장치를 찾지 못했다 (연결·udev 확인)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sec", type=float, default=6.0, help="재생 길이(초)")
    args = parser.parse_args()

    import sounddevice as sd

    in_dev = find_input_device()
    out_dev = audio_out.output_device()
    print(f"입력 장치 {in_dev} (6ch) / 출력 {'기본 장치 ⚠️ AEC 참조 안 됨' if out_dev is None else out_dev}")

    frames: queue.Queue[np.ndarray] = queue.Queue()

    def cb(indata, _frames, _time, _status):  # noqa: ANN001
        frames.put(np.frombuffer(indata, dtype=np.int16).reshape(-1, CHANNELS).copy())

    signal = make_probe_signal(args.sec)
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=1280, channels=CHANNELS,
                           dtype="int16", device=in_dev, callback=cb):
        sd.sleep(int(BASELINE_SEC * 1000))          # 1) 소음 바닥
        n_baseline = frames.qsize()
        audio_out.play(signal, SAMPLE_RATE, blocking=True)   # 2) 재생 + 동시 녹음
        sd.sleep(200)                                # 꼬리 잔향

    blocks = []
    while not frames.empty():
        blocks.append(frames.get())
    recording = np.concatenate(blocks).astype(np.float32) / 32768.0
    baseline_len = sum(len(b) for b in blocks[:n_baseline])
    quiet, active = recording[:baseline_len], recording[baseline_len:]

    ch0_q, ch1_q = rms(quiet[:, 0]), rms(quiet[:, 1])
    ch0_a, ch1_a = rms(active[:, 0]), rms(active[:, 1])
    # ch5 = 장치 내부의 재생 복사본. 여기 소리가 있으면 AEC 참조가 흐르는 것.
    ch5_q, ch5_a = rms(quiet[:, 5]), rms(active[:, 5])
    raw_peak = float(np.max(np.abs(active[:, 1]))) if active.size else 0.0

    print(f"\n소음 바닥   : ch0 {ch0_q:.5f} / ch1(원음) {ch1_q:.5f} / ch5(참조) {ch5_q:.5f}")
    print(f"재생 구간   : ch0 {ch0_a:.5f} / ch1(원음) {ch1_a:.5f} / ch5(참조) {ch5_a:.5f}"
          f"  (ch1 peak {raw_peak:.2f})")
    if ch5_a < max(ch5_q * 3, 1e-4):
        print("⚠️ 참조(ch5)에 재생음이 없다 — 재생이 이 장치의 USB 경로로 안 들어갔다."
              " AEC 는 참조 없이는 아무것도 못 뺀다.")

    if ch1_a < ch1_q * 3:
        print("\n판정: 원음 채널에 재생음이 거의 없다 → 스피커가 안 울렸다."
              "\n      3.5mm 배선·스피커 전원·볼륨을 확인할 것.")
        return

    second = int(1.0 * SAMPLE_RATE)
    ch0_head = rms(active[:second, 0])
    ch0_tail = rms(active[-second:, 0]) if len(active) > 2 * second else ch0_head
    ch1_tail = rms(active[-second:, 1]) if len(active) > 2 * second else ch1_a
    convergence = db(ch0_head / ch0_tail) if ch0_tail > 0 else float("inf")
    suppression = db(ch1_tail / ch0_tail) if ch0_tail > 0 else float("inf")

    print(f"\n수렴량      : {convergence:+.1f} dB  (ch0 첫 1초 {ch0_head:.5f} → 끝 1초 {ch0_tail:.5f})")
    print(f"억제량      : {suppression:+.1f} dB  (수렴 후 원음 대비 — AGC 탓에 보수적)")
    if raw_peak > 0.9:
        print("⚠️ 원음 클리핑 의심 — 스피커 볼륨을 낮추고 재측정할 것 (AEC 성능 저하 원인)")
    if convergence >= 6 or suppression >= 10:
        print("판정: AEC 동작 — 뮤트 제거 함정 문장 시험으로 진행 가능")
    elif convergence >= 3:
        print("판정: 약한 수렴 — --sec 10 으로 재측정, 볼륨·배선 확인")
    else:
        print("판정: 수렴 없음 — 참조(재생)가 reSpeaker 경로로 나가는지, 배선을 확인")


if __name__ == "__main__":
    main()

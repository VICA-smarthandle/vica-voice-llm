"""칩 VAD 실측 — "하드웨어 VAD 는 자기 재생음에 반응하지 않는다" 가정 검증.

음성 barge-in 재설계(RMS → 칩 VAD)의 전제를 측정한다. RMS 는 자기 잔여
에코와 사람을 못 가린다는 것이 두 번 실측됐다(2026-08-24 자책골). XVF-3000
의 VOICEACTIVITY 는 AEC 후단에서 계산되므로 로봇 자신의 재생음에는 0 이어야
하는데, 그 가정이 참인지 여기서 확인한다.

3구간 측정 (구간 사이 삑 소리로 구분):
    A 정적 5초        — 방이 조용할 때의 VAD 오븐율 (기대: ≈0%)
    B 재생+침묵 15초  — 로봇이 말하고 시험자는 조용히 (기대: 낮음 ★핵심)
    C 재생+발화 15초  — 로봇이 말하는 동안 시험자가 계속 말함 (기대: 높음)

판정: B ≤ 15% 이고 C ≥ 50% 이면 VAD 를 barge-in 방아쇠로 쓸 수 있다.

사용:
    HF_HUB_OFFLINE=1 .venv/bin/python -u -m tools.vad_probe
"""
from __future__ import annotations

import sys
import threading
import time

sys.path.insert(0, ".")

from src import audio_cue, audio_out  # noqa: E402
from src.dsp_state import DspState  # noqa: E402

POLL_HZ = 20
PHASE_A_SEC = 5.0
PHASE_BC_SEC = 15.0

ANNOUNCEMENT = [
    "지금부터 안내를 시작하겠습니다. 좌측에 계단이 있으니 손잡이를 잡아 주세요.",
    "복도를 따라 직진하고 있습니다. 전방에 회전 구간이 있어 속도를 줄이겠습니다.",
]


def duty(samples, t0, t1, key):
    window = [s[key] for s in samples if t0 <= s["t"] < t1 and s[key] is not None]
    if not window:
        return float("nan"), 0
    return 100.0 * sum(window) / len(window), len(window)


def main() -> None:
    dsp = DspState()
    if not dsp.available:
        print("reSpeaker 상태 레지스터를 읽을 수 없다 (권한/장치).")
        print("udev 규칙이 필요할 수 있다: /etc/udev/rules.d/99-respeaker.rules 에")
        print('SUBSYSTEM=="usb", ATTRS{idVendor}=="2886", ATTRS{idProduct}=="0018", MODE="0666"')
        raise SystemExit(1)

    from src.tts import VicaTTS

    print("TTS 합성 중...")
    tts = VicaTTS()
    clips = [tts._synthesize(text) for text in ANNOUNCEMENT]

    samples: list[dict] = []
    stop = threading.Event()
    # 스트림을 여닫는 순간에는 폴링을 멈춘다 — 열기/닫기는 USB 제어 채널(EP0)
    # 을 쓰는데 우리 레지스터 읽기도 같은 채널이라, 겹치면 장치가 스트림 열기를
    # 거부한다(Device unavailable 실측 2026-08-24). 스트리밍 중에는 EP0 이
    # 한가해 폴링과 공존한다 (Seeed 튜닝 도구가 녹음 중 도는 것과 같은 원리).
    poll_gate = threading.Event()
    poll_gate.set()

    def poll() -> None:
        while not stop.is_set():
            poll_gate.wait(timeout=0.5)
            if not poll_gate.is_set():
                continue
            samples.append({
                "t": time.time(),
                "vad": dsp.voice_activity(),
                "speech": dsp.speech_detected(),
            })
            time.sleep(1.0 / POLL_HZ)

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()

    import numpy as np
    import sounddevice as sd

    dev_index, dev_rate, dev_channels = audio_out.output_device()

    def phase_wave(seconds: float) -> np.ndarray:
        parts, total = [], 0
        i = 0
        while total < int(seconds * dev_rate):
            wav, rate = clips[i % len(clips)]
            i += 1
            out = audio_out.prepare(wav, rate, dev_rate, dev_channels)
            parts.append(out)
            total += len(out)
        return np.concatenate(parts)[: int(seconds * dev_rate)]

    def play_wave(wave: np.ndarray) -> None:
        """구간당 스트림을 한 번만 열고, 여닫는 동안 폴링을 멈춘다."""
        poll_gate.clear()
        time.sleep(0.1)
        stream = sd.OutputStream(samplerate=dev_rate, device=dev_index,
                                 channels=wave.shape[1], dtype="float32")
        stream.start()
        poll_gate.set()
        for i in range(0, len(wave), 1600):
            stream.write(np.ascontiguousarray(wave[i:i + 1600]))
        poll_gate.clear()
        time.sleep(0.1)
        stream.stop()
        stream.close()
        poll_gate.set()

    def speak_for(seconds: float) -> None:
        play_wave(phase_wave(seconds))

    print(f"\nA구간 {PHASE_A_SEC:.0f}초 — 조용히 계세요")
    a0 = time.time()
    time.sleep(PHASE_A_SEC)

    print(f"B구간 {PHASE_BC_SEC:.0f}초 — 로봇이 말합니다. 계속 조용히! ★핵심 구간")
    b0 = time.time()
    speak_for(PHASE_BC_SEC)
    b1 = time.time()

    beep = audio_out.prepare(audio_cue.arrived(), audio_cue.SAMPLE_RATE,
                             dev_rate, dev_channels)
    play_wave(beep)   # 구간 전환 알림음
    time.sleep(1.0)
    print(f"C구간 {PHASE_BC_SEC:.0f}초 — 삑 소리 후 로봇이 말하는 동안 아무 말이나 계속 하세요")
    c0 = time.time()
    speak_for(PHASE_BC_SEC)
    c1 = time.time()

    time.sleep(0.5)
    stop.set()
    poller.join(timeout=3)
    dsp.close()

    print("\n===== 결과 (VOICEACTIVITY 켜짐 비율) =====")
    for name, (t0, t1) in {
        "A 정적": (a0, b0), "B 재생+침묵 ★": (b0, b1), "C 재생+발화": (c0, c1),
    }.items():
        vad, n = duty(samples, t0, t1, "vad")
        speech, _ = duty(samples, t0, t1, "speech")
        print(f"  {name:14} VAD {vad:5.1f}%  (SPEECHDETECTED {speech:5.1f}%, 표본 {n})")

    vad_b, _ = duty(samples, b0, b1, "vad")
    vad_c, _ = duty(samples, c0, c1, "vad")
    if vad_b <= 15 and vad_c >= 50:
        print("판정: 사용 가능 — 칩 VAD 를 barge-in 방아쇠로 쓴다")
    elif vad_b <= 15:
        print("판정: 절반 — 자기 소리엔 안 속지만 사람 감도도 낮다. 지속시간·보조 조건 재설계 필요")
    else:
        print("판정: 부적합 — 칩 VAD 도 자기 재생음에 반응한다. DOA/이중채널 등 다른 증거 필요")


if __name__ == "__main__":
    main()

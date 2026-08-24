"""뮤트 제거 양성 대조 — 로봇이 말하는 도중 사람의 "멈춰"가 들리는지 확인한다.

mute_off_trial(함정 시험)의 반대쪽 절반이다. 함정 시험은 "로봇 소리에 속지
않는다"를, 이 시험은 "로봇이 말하는 중에도 진짜 긴급어는 들린다"를 본다.
둘 다 통과해야 뮤트 제거(VICA_TTS_MUTE=off)가 확정된다.

절차: 로봇이 긴 안내(약 40초)를 연속으로 말한다. 시험자는 아무 때나 "멈춰!"를
외친다. AEC 가 로봇 목소리를 ch0 에서 빼 주므로, whisper 검증 전사에는
사람 목소리만 남아 정확 매칭을 통과해야 한다.

사용:
    HF_HUB_OFFLINE=1 VICA_STT_DEVICE=cuda VICA_STT_COMPUTE=float16 \
        .venv/bin/python -m tools.mute_off_positive
"""
from __future__ import annotations

import queue
import sys
import threading
import time

import numpy as np

sys.path.insert(0, ".")

from src import audio_out  # noqa: E402
from src.wakeword_monitor import WakewordMonitor  # noqa: E402

SAMPLE_RATE = 16000
FRAME = 1280
CHANNELS = 6

# 함정어 없는 현실적인 안내. 두 바퀴 돌면 약 40초 — 외칠 시간은 충분하다.
ANNOUNCEMENT = [
    "지금부터 안내를 시작하겠습니다. 좌측에 계단이 있으니 손잡이를 잡아 주세요.",
    "복도를 따라 직진하고 있습니다. 전방에 회전 구간이 있어 속도를 줄이겠습니다.",
    "잠시 후 우회전합니다. 벽이 가까우니 오른손을 안쪽으로 모아 주세요.",
    "엘리베이터 앞을 지나 이동 중입니다. 주변이 혼잡하니 천천히 이동하겠습니다.",
]
LAPS = 2


def main() -> None:
    import sounddevice as sd

    from src.tts import VicaTTS

    print("1/3 TTS 합성 중...")
    tts = VicaTTS()
    clips = [tts._synthesize(text) for text in ANNOUNCEMENT]

    print("2/3 감시 모델 로드 중 (openWakeWord + whisper)...")
    events, wakes, outcomes, transcripts = [], [], [], []
    monitor = WakewordMonitor(
        on_emergency=lambda e: (
            events.append((time.time(), e)),
            print(f"\n  🚨🚨 긴급 확정: '{e.keyword}' (전사: {e.source_text!r}) 🚨🚨\n"),
        ),
        on_user_text=lambda t: None,
        on_wake=lambda: wakes.append(time.time()),
    )
    monitor._load_real()

    # 검증 전사를 전부 기록한다 — 기각된 외침의 원인(전사 내용)을 봐야 한다.
    orig_transcribe = monitor._transcribe

    def spy(audio: np.ndarray) -> str:
        text = orig_transcribe(audio)
        transcripts.append((time.time(), text))
        return text

    monitor._transcribe = spy

    in_dev = next(i for i, d in enumerate(sd.query_devices())
                  if "respeaker" in d["name"].lower() and d["max_input_channels"] >= CHANNELS)
    frames: queue.Queue[np.ndarray] = queue.Queue()
    stop = threading.Event()

    def cb(indata, _frames, _time, _status):  # noqa: ANN001
        block = np.frombuffer(indata, dtype=np.int16).reshape(-1, CHANNELS)
        frames.put(block[:, 0].copy())

    def consume() -> None:
        while not stop.is_set():
            try:
                frame = frames.get(timeout=0.2)
            except queue.Empty:
                continue
            r = monitor.process_frame(frame)
            if r is not None:
                outcomes.append((time.time(), r))

    consumer = threading.Thread(target=consume, daemon=True)

    print("3/3 시험 시작 — 로봇이 말하는 동안 아무 때나 '멈춰!'라고 외치세요\n")
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=FRAME, channels=CHANNELS,
                           dtype="int16", device=in_dev, callback=cb):
        consumer.start()
        time.sleep(1.0)
        t_start = time.time()
        for lap in range(LAPS):
            for i, (wav, rate) in enumerate(clips):
                if events:
                    break  # 이미 확정 — 나머지 안내는 생략
                print(f"  🔊 재생 중 ({lap + 1}바퀴 {i + 1}/{len(clips)})...")
                monitor.set_speaking(True)
                audio_out.play(wav, rate, blocking=True)
                monitor.set_speaking(False)
                time.sleep(0.3)
            if events:
                break
        time.sleep(4.0)  # 마지막 외침의 검증 대기
        stop.set()
    consumer.join(timeout=5)
    took = time.time() - t_start

    n_reject = sum(1 for _, r in outcomes if r == "reject")
    print(f"\n===== 결과 ({took:.0f}초) =====")
    print(f"긴급 확정   : {len(events)}건  {[(e.keyword, e.source_text) for _, e in events]}")
    print(f"관문→기각   : {n_reject}건")
    for ts, text in transcripts:
        print(f"  검증 전사 @{ts - t_start:+.1f}s: {text!r}")
    if events:
        print("판정: 통과 — 로봇이 말하는 중에도 긴급어가 들린다. 뮤트 제거 확정 조건 충족.")
    elif n_reject:
        print("판정: 관문은 넘었으나 STT 정확 매칭에서 기각 — 위 전사를 보고 원인 분석.")
    else:
        print("판정: 외침이 관문(모델 B)조차 못 넘음 — 거리·성량 또는 AEC 잔여 확인.")


if __name__ == "__main__":
    main()

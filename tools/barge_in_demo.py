"""barge-in 실기 시연 — 로봇 질문 도중 답을 시작하면 즉시 말을 끊고 듣는다.

ROS 없이 모니터와 재생을 한 프로세스에 묶어 실전 흐름을 재현한다:
    arm_followup(질문 예약) → set_speaking(True) + 질문 재생
    → 사용자가 도중에 답 시작 → 모니터 barge-in → audio_out.stop() (재생 끊김)
    → 청취 창(말머리 보존) → whisper 전사

사용:
    HF_HUB_OFFLINE=1 VICA_STT_DEVICE=cuda VICA_STT_COMPUTE=float16 \
        .venv/bin/python -m tools.barge_in_demo
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
ROUNDS = 2

QUESTION = (
    "화장실로 안내해 드릴까요? 이동 중에는 손잡이를 꼭 잡아 주시고, "
    "도착할 때까지 안내 음성에 귀를 기울여 주세요. 안내를 시작해도 괜찮을까요?"
)


def main() -> None:
    import sounddevice as sd

    from src.tts import VicaTTS

    print("1/3 TTS 합성 중...")
    wav, rate = VicaTTS()._synthesize(QUESTION)
    clip_sec = len(wav) / rate

    print("2/3 감시 모델 로드 중...")
    texts, outcomes, transcripts = [], [], []
    barge_at: list[float] = []

    def on_barge_in() -> None:
        barge_at.append(time.time())
        audio_out.stop()          # 실전에서는 /vica/tts_stop → TTS 노드가 한다
        print("  ✂️  barge-in — 질문 재생 끊음, 듣는 중...")

    monitor = WakewordMonitor(
        on_emergency=lambda e: print(f"  🚨 긴급 '{e.keyword}'"),
        on_user_text=lambda t: texts.append((time.time(), t)),
        on_barge_in=on_barge_in,
    )
    monitor._load_real()
    orig = monitor._transcribe

    def spy(audio: np.ndarray) -> str:
        text = orig(audio)
        transcripts.append(text)
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

    print("3/3 시작 — 질문이 끝나기 전에 답을 말하세요 (예: '응, 가자')\n")
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=FRAME, channels=CHANNELS,
                           dtype="int16", device=in_dev, callback=cb):
        consumer.start()
        time.sleep(1.0)
        for round_no in range(1, ROUNDS + 1):
            print(f"  🔊 질문 재생 ({round_no}/{ROUNDS}, 길이 {clip_sec:.1f}초)...")
            monitor.arm_followup()
            monitor.set_speaking(True)
            t0 = time.time()
            audio_out.play(wav, rate, blocking=True)
            played = time.time() - t0
            monitor.set_speaking(False)
            cut = played < clip_sec - 0.5
            print(f"     재생 {played:.1f}초 / 원본 {clip_sec:.1f}초 → "
                  f"{'끊김 ✂️' if cut else '끝까지 재생됨'}")
            deadline = time.time() + 10.0   # 답 전사 대기
            while time.time() < deadline and not texts:
                time.sleep(0.2)
            if texts:
                break
            print("     (이번 회차엔 끼어들기가 없었습니다)")
        stop.set()
    consumer.join(timeout=5)

    print("\n===== 결과 =====")
    print(f"barge-in 발동 : {len(barge_at)}회")
    if barge_at:
        print(f"끊기까지 지연 : 재생 시작 후 {barge_at[-1] - t0:.2f}초 시점")
    print(f"들은 답        : {[t for _, t in texts]!r}")
    print(f"검증 전사 전체 : {transcripts!r}")
    if texts and barge_at:
        print("판정: 통과 — 질문을 끊고 답을 들었다 (자연스러운 대화 흐름)")
    elif texts:
        print("판정: 부분 — 답은 들었으나 끊김 없이 질문 종료 후였다")
    else:
        print("판정: 실패 — 답을 듣지 못함. 성량·타이밍 또는 임계값 확인")


if __name__ == "__main__":
    main()

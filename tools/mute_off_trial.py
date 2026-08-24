"""뮤트 제거 관문 시험 — 로봇이 함정 문장을 말하는 동안 귀를 열어 두고 오탐을 센다.

AEC 배선 후 뮤트(VICA_TTS_MUTE=off)를 꺼도 되는지 판정하는 실기 시험이다.
실제 TTS(supertonic F5)로 함정 문장을 reSpeaker 재생 경로로 틀고, 같은 시간에
실제 웨이크워드 감시(openWakeWord + whisper 검증)를 ch0 에 돌린다 — 실전과
같은 소리, 같은 귀.

함정의 근거: "행정지원실" 속 "정지" 때문에 뮤트가 생겼다(ros_tts_node 주석).
유사어(멈춤 등)가 모델 관문을 뚫는 것은 실측됨(stt-gate-findings). 최종 오탐은
whisper 정확 매칭이 막아야 한다.

판정:
    emergency 0건 · ghost(user_text) 0건 이어야 뮤트 제거 확정.
    wake(비카야 오탐)는 건수를 보고 판단한다 — 인사만 나가고 청취는
    wake_silent 로 닫히면 실해는 없다.

사용:
    HF_HUB_OFFLINE=1 .venv/bin/python -m tools.mute_off_trial --reps 2
"""
from __future__ import annotations

import argparse
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

# 실전에서 로봇이 실제로 말할 법한 문장 + 최악 케이스(단독 발화).
# 단독 발화 둘은 devlog/2026-08-13(자가각성) 의 미해결 항목이다: TTS 음색이
# 모델 분포 밖이라 안 걸린다는 가설이 "비카야" 단독에서도 성립하는지 본다.
# 로봇 대본에 이런 단독 발화는 없지만, 여기서 걸리면 "대본에 웨이크워드·긴급어
# 단독 발화 금지"를 규칙으로 못박아야 한다.
TRAPS = [
    ("정지 내포", "행정지원실로 안내하겠습니다."),
    ("정지 그대로", "비상 정지 버튼은 손잡이 오른쪽에 있습니다."),
    ("유사어 멈춤", "잠시 멈춤 없이 계속 이동하겠습니다."),
    ("평범 대조군", "목적지에 도착했습니다. 이용해 주셔서 감사합니다."),
    ("비카야 유사", "비가 오는 날에는 바닥이 미끄러우니 조심하세요."),
    ("웨이크 단독 ★", "비카야."),
    ("긴급어 단독 ★", "멈춰."),
    ("긴급어 문장 ★", "잠시 멈춰 주세요."),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=2,
                        help="반복 횟수 (1회차는 AEC 냉시동, 2회차는 수렴 후)")
    args = parser.parse_args()

    import sounddevice as sd

    from src.tts import VicaTTS

    print("1/3 TTS 합성 중 (supertonic)...")
    tts = VicaTTS()
    clips = [(label, *tts._synthesize(text), text) for label, text in TRAPS]

    print("2/3 감시 모델 로드 중 (openWakeWord + whisper)...")
    events, texts, wakes, outcomes = [], [], [], []
    monitor = WakewordMonitor(
        on_emergency=lambda e: events.append((time.time(), e)),
        on_user_text=lambda t: texts.append((time.time(), t)),
        on_wake=lambda: wakes.append(time.time()),
    )
    monitor._load_real()

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

    print(f"3/3 시험 시작 — 문장 {len(TRAPS)}개 × {args.reps}회 (귀 열림, 뮤트 없음)")
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=FRAME, channels=CHANNELS,
                           dtype="int16", device=in_dev, callback=cb):
        consumer.start()
        time.sleep(1.0)  # 소음 바닥에서 시작
        bad_regular, bad_star = [], []
        for rep in range(1, args.reps + 1):
            for label, wav, rate, text in clips:
                t0 = time.time()
                monitor.set_speaking(True)
                audio_out.play(wav, rate, blocking=True)
                monitor.set_speaking(False)
                time.sleep(1.5)  # 검증·잔향 여유
                seen = [r for ts, r in outcomes if ts >= t0]
                print(f"  [{rep}회차|{label}] {text!r} → {seen if seen else '조용함 ✓'}")
                hits = [r for r in seen if r in ("emergency", "user_text", "wake")]
                if hits:
                    (bad_star if "★" in label else bad_regular).append(label)
        time.sleep(3.0)  # 마지막 검증 대기
        stop.set()
    consumer.join(timeout=5)

    n_reject = sum(1 for _, r in outcomes if r == "reject")
    n_silent = sum(1 for _, r in outcomes if r == "wake_silent")
    print("\n===== 결과 =====")
    print(f"긴급 오탐(emergency) : {len(events)}건  {[e.keyword for _, e in events]}")
    print(f"유령 발화(user_text) : {len(texts)}건  {[t for _, t in texts]}")
    print(f"호출 오탐(wake)      : {len(wakes)}건 (그중 조용히 닫힘 {n_silent}건)")
    print(f"관문 발동→STT 기각   : {n_reject}건 (정상 방어)")
    if bad_regular:
        verdict = f"불통과 — 일반 대본에서 오탐: {bad_regular}. 뮤트 유지, 원인 분석 필요"
    elif bad_star:
        verdict = (f"조건부 통과 — 오탐은 대본 금지 케이스(★)뿐: {bad_star}. "
                   "대본 규칙(웨이크워드·긴급어 단독/명령형 발화 금지) 전제로 뮤트 제거 가능")
    else:
        verdict = "통과 — 뮤트 제거(VICA_TTS_MUTE=off) 가능"
    print(f"판정: {verdict}")


if __name__ == "__main__":
    main()

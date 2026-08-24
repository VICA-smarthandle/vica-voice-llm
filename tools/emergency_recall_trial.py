"""TTS 중 긴급 인식률 정식 실측 — 외침 하나하나의 운명을 전부 기록한다.

시험자는 로봇이 말하는 동안 "멈춰!"를 정해진 횟수(기본 10회) 외친다.
회차마다 다음 중 하나로 분류된다:

    성공        관문 통과 + whisper 정확 매칭 → 긴급 확정
    기각        관문은 넘었으나 매칭 실패 — whisper 가 받아쓴 내용을 남긴다
                (로봇 목소리가 섞였는지 여기서 보인다)
    근접 미달   모델 B 점수가 0.2~관문(0.5) 사이 — 문턱/연속 조건 문제
    무반응      점수 0.2 미만 — 소리가 모델까지 닿지 않음 (AEC/음량/거리)

2026-08-24 1차 시험에서 "멈춰 3회 중 1회"의 원인을 가릴 수 없었던 관측
공백(기각 무로그)을 메우는 도구다. 결과에 따라 다음 조치가 갈린다:
기각 다수 → 매칭 완화 검토(팀 승인), 근접 미달 다수 → TTS 중 관문 조정,
무반응 다수 → AEC/게인 조사.

사용:
    HF_HUB_OFFLINE=1 VICA_STT_DEVICE=cuda VICA_STT_COMPUTE=float16 \
        .venv/bin/python -u -m tools.emergency_recall_trial --shouts 10
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
NEAR_MISS_FLOOR = 0.2   # 이 미만이면 "무반응", 이상~관문 미만이면 "근접 미달"
CLUSTER_GAP_SEC = 1.5   # 점수 이벤트를 외침 단위로 묶는 간격

ANNOUNCEMENT = [
    "지금부터 안내를 시작하겠습니다. 좌측에 계단이 있으니 손잡이를 잡아 주세요.",
    "복도를 따라 직진하고 있습니다. 전방에 회전 구간이 있어 속도를 줄이겠습니다.",
    "엘리베이터 앞을 지나 이동 중입니다. 주변이 혼잡하니 천천히 이동하겠습니다.",
    "잠시 후 우회전합니다. 벽이 가까우니 오른손을 안쪽으로 모아 주세요.",
]


def cluster(times_scores, gap=CLUSTER_GAP_SEC):
    """(시각, 점수) 목록을 외침 단위 묶음으로 나눈다. 각 묶음은 (시작, 최고점수)."""
    clusters = []
    for ts, score in times_scores:
        if clusters and ts - clusters[-1][1] <= gap:
            start, _last, peak = clusters[-1]
            clusters[-1] = (start, ts, max(peak, score))
        else:
            clusters.append((ts, ts, score))
    return [(start, peak) for start, _end, peak in clusters]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shouts", type=int, default=10, help="시험자가 외칠 횟수")
    parser.add_argument("--max-sec", type=float, default=120.0, help="재생 총 길이")
    args = parser.parse_args()

    import sounddevice as sd

    from src.tts import VicaTTS

    print("1/3 TTS 합성 중...")
    tts = VicaTTS()
    clips = [tts._synthesize(text) for text in ANNOUNCEMENT]

    print("2/3 감시 모델 로드 중...")
    emergencies, rejects, bscores = [], [], []
    monitor = WakewordMonitor(
        on_emergency=lambda e: (
            emergencies.append((time.time(), e.source_text)),
            print(f"  🚨 성공: 긴급 확정 (전사 {e.source_text!r})"),
        ),
        on_user_text=lambda t: None,
        on_reject=lambda text: (
            rejects.append((time.time(), text)),
            print(f"  ⚠️ 기각: 관문은 넘었으나 매칭 실패 (전사 {text!r})"),
        ),
    )
    monitor._load_real()

    orig_predict = monitor._predict

    def spy_predict(frame: np.ndarray) -> dict:
        s = orig_predict(frame)
        if float(s["b"]) >= NEAR_MISS_FLOOR:
            bscores.append((time.time(), float(s["b"])))
        return s

    monitor._predict = spy_predict

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
            monitor.process_frame(frame)

    consumer = threading.Thread(target=consume, daemon=True)

    print(f"3/3 시작 — 로봇이 말하는 동안 '멈춰!'를 정확히 {args.shouts}회, "
          f"7초 이상 간격으로 외치세요 (최대 {args.max_sec:.0f}초)\n")
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=FRAME, channels=CHANNELS,
                           dtype="int16", device=in_dev, callback=cb):
        consumer.start()
        time.sleep(1.0)
        t_start = time.time()
        lap = 0
        while time.time() - t_start < args.max_sec:
            wav, rate = clips[lap % len(clips)]
            lap += 1
            monitor.set_speaking(True)
            audio_out.play(wav, rate, blocking=True)   # 긴급이 나도 계속 말한다
            monitor.set_speaking(False)                #  — 시험이니까
            time.sleep(0.3)
        time.sleep(4.0)   # 마지막 검증 대기
        stop.set()
    consumer.join(timeout=5)

    # 점수 묶음에서 성공·기각과 겹치지 않는 것 = 미달 (근접/무반응은 총수 계산으로)
    fired_times = [ts for ts, _ in emergencies] + [ts for ts, _ in rejects]
    score_clusters = cluster(sorted(bscores))
    near_miss = [
        (ts, peak) for ts, peak in score_clusters
        if peak < monitor.gate_b.threshold
        and not any(abs(ts - ft) < 3.0 for ft in fired_times)
    ]
    n_success, n_reject = len(emergencies), len(rejects)
    n_near = len(near_miss)
    n_silent = max(0, args.shouts - n_success - n_reject - n_near)

    print("\n===== 결과 =====")
    print(f"외침(신고)   : {args.shouts}회")
    print(f"성공         : {n_success}회")
    print(f"기각         : {n_reject}회")
    for ts, text in rejects:
        print(f"    @{ts - t_start:+.1f}s whisper: {text!r}")
    print(f"근접 미달    : {n_near}회 (점수 {NEAR_MISS_FLOOR}~{monitor.gate_b.threshold})")
    for ts, peak in near_miss:
        print(f"    @{ts - t_start:+.1f}s 최고점수 {peak:.2f}")
    print(f"무반응(추정) : {n_silent}회 — 점수가 {NEAR_MISS_FLOOR} 에도 못 미침")
    print(f"\n인식률: {n_success}/{args.shouts}"
          f"  (관문 통과율 {n_success + n_reject}/{args.shouts})")


if __name__ == "__main__":
    main()

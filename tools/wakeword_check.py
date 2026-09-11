"""웨이크워드("비카야") 측정 도구 — 만들기 전에 되는지부터 재 본다."""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .wakeword_score import (
    DEFAULT_VARIANTS,
    Trial,
    format_report,
    match_wake_word,
)

SAMPLE_RATE = 16000

WINDOW_SEC = 2.0
HOP_SEC = 0.5
RMS_THRESHOLD = 0.01

DEFAULT_PHRASES: list[tuple[str, bool, str]] = [
    ("비카야", True, "깨우는 말"),
    ("비켜", False, "복도에서 흔한 말 — 가장 중요한 함정"),
    ("비켜주세요", False, "위와 같은 계열"),
    ("비상", False, "앞 두 글자가 비슷하다"),
    ("이거야", False, "끝 두 글자가 비슷하다"),
]


def _load_transcribe(model: str):
    """whisper 를 올린다. 시간이 걸리므로 측정 전에 한 번만."""
    from src.stt import VicaSTT

    print(f"STT 모델 로드 중... (모델: {model})")
    return VicaSTT(model_size=model).transcribe


def _record(seconds: float):
    """마이크로 지정 시간만큼 녹음해 1차원 float32 파형을 돌려준다."""
    import sounddevice as sd

    audio = sd.rec(
        int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32"
    )
    sd.wait()
    return audio.flatten()


def _countdown(phrase: str, index: int, total: int) -> None:
    print(f"\n[{index}/{total}] 준비하세요 — 신호가 나오면 「{phrase}」")
    for n in (3, 2, 1):
        print(f"  {n}...", end="", flush=True)
        time.sleep(0.7)
    print(f"  🎤 지금 말하세요! ({WINDOW_SEC}초)", flush=True)


def run_phrase(
    transcribe, phrase: str, should_wake: bool, repeat: int, variants, matcher=None
) -> list[Trial]:
    """문구 하나를 repeat 번 말하게 하고 결과를 모은다."""
    if matcher is None:
        def matcher(text: str):
            return match_wake_word(text, variants)

    kind = "깨어나야 함" if should_wake else "깨어나면 안 됨"
    print("\n" + "=" * 62)
    print(f"「{phrase}」 을(를) {repeat}번 말합니다.  ({kind})")
    print("=" * 62)
    if input("준비되면 엔터 (건너뛰려면 s 입력 후 엔터) > ").strip().lower() == "s":
        print("  (건너뜀)")
        return []

    trials: list[Trial] = []
    for i in range(1, repeat + 1):
        _countdown(phrase, i, repeat)
        audio = _record(WINDOW_SEC)
        rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0

        if rms < RMS_THRESHOLD:
            print(f"      🔇 음량 미달 (rms {rms:.4f} < {RMS_THRESHOLD}) — STT 생략")
            trials.append(
                Trial(phrase, should_wake, "", None, rms=rms, gated=True)
            )
            continue

        text = transcribe(audio).strip()
        matched = matcher(text)

        woke = matched is not None
        mark = "✅" if woke == should_wake else "❌"
        state = "깨어남" if woke else "안 깨어남"
        print(f"      {mark} {state:<8} 받아쓰기: {text or '(빈 인식)'}")
        trials.append(
            Trial(phrase, should_wake, text, matched, rms=rms, gated=False)
        )
    return trials


def run_ambient(transcribe, seconds: float, variants) -> None:
    """아무 말도 하지 않고, 주변 소리만으로 깨어나는지 관찰한다."""
    import sounddevice as sd

    window = int(WINDOW_SEC * SAMPLE_RATE)
    buffer = np.zeros(window, dtype=np.float32)

    def callback(indata, _frames, _time, _status):
        nonlocal buffer
        buffer = np.concatenate([buffer, indata[:, 0]])[-window:]

    print("\n" + "=" * 62)
    print(f"배경 소음 관찰 {seconds:.0f}초 — 이 동안 「비카야」라고 하지 마세요.")
    print("실제 사용할 장소(로비·복도 등)에서 돌려야 의미가 있습니다. (Ctrl+C 중단)")
    print("=" * 62)

    started = time.time()
    checked = wakes = stt_runs = 0
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback
        ):
            while time.time() - started < seconds:
                time.sleep(HOP_SEC)
                snapshot = buffer.copy()
                checked += 1
                rms = float(np.sqrt(np.mean(snapshot**2)))
                if rms < RMS_THRESHOLD:
                    continue
                stt_runs += 1
                text = transcribe(snapshot).strip()
                if match_wake_word(text, variants):
                    wakes += 1
                    stamp = time.strftime("%H:%M:%S")
                    print(f"  ⚠️ [{stamp}] 헛깨움! 받아쓰기: {text}")
    except KeyboardInterrupt:
        print("\n(중단됨)")

    elapsed = max(time.time() - started, 1e-9)
    print("\n" + "-" * 62)
    print(f"관찰 시간      : {elapsed:.0f}초")
    print(f"검사한 창      : {checked}회")
    print(f"소리가 있어 STT: {stt_runs}회 ({stt_runs / max(checked, 1):.0%})")
    print(f"헛깨움         : {wakes}회  →  시간당 {wakes / elapsed * 3600:.1f}회")
    print("-" * 62)
    print("※ 시간당 헛깨움이 1회를 넘으면 실사용에서 성가시다고 본다(잠정 기준).")


def save_report(text: str, trials, emergency: bool = False, variants=DEFAULT_VARIANTS) -> Path:
    """결과를 파일로 남긴다. 팀 회의에 그대로 가져가기 위해서다."""
    kind = "긴급어" if emergency else "웨이크워드"
    out_dir = Path("docs/measurements")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "emergency" if emergency else "wakeword"
    path = out_dir / f"{prefix}-{datetime.now():%Y%m%d-%H%M}.md"

    judged = "운영 detect_emergency 그대로" if emergency else f"인정 후보 {', '.join(variants)}"
    lines = [
        f"# {kind} 측정 기록 ({datetime.now():%Y-%m-%d %H:%M})",
        "",
        f"- STT 모델: `{os.environ.get('_VICA_WW_MODEL', '?')}`",
        f"- 창 길이: {WINDOW_SEC}초 (운영 긴급어 감시와 동일)",
        f"- 판정: {judged}",
        "- 장소·소음 상황: (직접 적어 주세요)",
        "",
        "```",
        text,
        "```",
        "",
        "## 시도별 원본 기록",
        "",
        "| 말한 문구 | 깨어나야 하나 | 받아쓰기 | 깨어남 |",
        "| --- | --- | --- | --- |",
    ]
    for t in trials:
        lines.append(
            f"| {t.phrase} | {'예' if t.should_wake else '아니오'} "
            f"| {t.heard or '(빈 인식)'} | {'예' if t.woke else '아니오'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="웨이크워드 인식률·오인율 측정 (제품 코드를 바꾸지 않는다)"
    )
    parser.add_argument("--repeat", type=int, default=10, help="문구당 반복 횟수 (기본 10)")
    parser.add_argument(
        "--model",
        default=os.environ.get("VICA_EMERGENCY_STT_MODEL", "medium"),
        help="whisper 모델. 기본은 운영 긴급어 감시와 같은 값(medium)",
    )
    parser.add_argument(
        "--phrase", action="append", default=None,
        help="목표 문구를 직접 지정 (여러 번 사용 가능). 지정하면 함정 문구는 빠진다",
    )
    parser.add_argument(
        "--variants", default=",".join(DEFAULT_VARIANTS),
        help="깨움으로 인정할 표기들 (쉼표 구분)",
    )
    parser.add_argument(
        "--ambient", type=float, default=None,
        help="말하지 않고 배경 소음만 N초 관찰 (다른 측정은 하지 않음)",
    )
    parser.add_argument(
        "--emergency", action="store_true",
        help="웨이크워드 대신 '긴급어'를 측정한다. 운영 detect_emergency 판정을 그대로 쓴다",
    )
    parser.add_argument("--no-save", action="store_true", help="결과 파일을 남기지 않는다")
    args = parser.parse_args(argv)

    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    os.environ["_VICA_WW_MODEL"] = args.model

    print("=" * 62)
    print("긴급어 측정 도구" if args.emergency else "웨이크워드 측정 도구")
    print("=" * 62)
    print(f"모델      : {args.model}   (운영 긴급어 감시 기본값은 medium)")
    print(f"창 길이   : {WINDOW_SEC}초")
    if args.emergency:
        from src.emergency_filter import EMERGENCY_KEYWORDS

        print(f"판정      : 운영 detect_emergency 그대로 ({', '.join(EMERGENCY_KEYWORDS)})")
    else:
        print(f"인정 표기 : {', '.join(variants)}")
    print("\n주의: 실제 사용할 장소·거리·소음에서 재야 의미가 있습니다.")
    print("      조용한 책상 위 결과는 실기 성능을 보장하지 않습니다.")

    try:
        transcribe = _load_transcribe(args.model)
    except Exception as exc:
        print(f"\nSTT 로드 실패: {exc}", file=sys.stderr)
        return 1

    if args.ambient is not None:
        run_ambient(transcribe, args.ambient, variants)
        return 0

    matcher = None
    if args.emergency:
        from src.emergency_filter import EMERGENCY_KEYWORDS, detect_emergency

        matcher = detect_emergency
        if args.phrase:
            plan = [(p, True, "직접 지정 (긴급어 판정)") for p in args.phrase]
        else:
            plan = [(k, True, "긴급어") for k in EMERGENCY_KEYWORDS]
            plan += [
                ("행정지원실", False, "'정지'가 들어간 일반 낱말 — 오탐 함정"),
                ("감정지수", False, "위와 같은 계열"),
            ]
    elif args.phrase:
        plan = [(p, True, "직접 지정") for p in args.phrase]
    else:
        plan = DEFAULT_PHRASES

    print("\n시험할 문구:")
    for phrase, should_wake, note in plan:
        print(f"  - 「{phrase}」 {'(깨어나야 함)' if should_wake else '(깨어나면 안 됨)'} — {note}")

    trials: list[Trial] = []
    try:
        for phrase, should_wake, _note in plan:
            trials.extend(
                run_phrase(transcribe, phrase, should_wake, args.repeat, variants, matcher)
            )
    except (KeyboardInterrupt, EOFError):
        print("\n\n(중단됨 — 여기까지의 결과로 정리합니다)")

    report = format_report(trials)
    print(report)

    if trials and not args.no_save:
        path = save_report(report, trials, emergency=args.emergency, variants=variants)
        print(f"기록을 저장했습니다: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

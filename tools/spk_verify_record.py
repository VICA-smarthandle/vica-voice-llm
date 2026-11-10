"""화자 검증 녹음기 — 사람마다 '안내 중 대화'(등록용)와 '대기 중 명령'(판정용)을 reSpeaker 6채널 그대로 녹음한다.
평가는 tools/spk_verify_eval.py 가 한다.

    cd ~/VICA-smarthandle/vica-voice-llm && .venv/bin/python -m tools.spk_verify_record --person 민준
    .venv/bin/python -m tools.spk_verify_record --person 민준 --start 9     # 9번 문장부터 이어서

전제: ⑫·⑬ 꺼짐(마이크를 혼자 써야 한다) · reSpeaker USB · 말하는 사람은 손잡이를 잡은 자세(로봇 뒤).
칩에는 운영 .env 의 AGC·에코 값을 ⑫ 가 켤 때와 똑같이 쓴다 — 운영과 같은 소리를 녹음하기 위해서다.
같은 사람·같은 번호를 다시 녹음하면 파일을 덮어쓰고, 평가기는 manifest 의 마지막 줄을 쓴다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.spk_verify_eval import REC_ROOT, SAMPLE_RATE, trim_speech  # noqa: E402

CHANNELS = 6
RECORD_SEC = 4.0
ENROLL = ("민원실로 가줘", "그래", "서류 내러 가고 싶어", "아니", "엘리베이터는 어디야?", "응 맞아")   # 안내 중 대화 흉내
WAKE = "비카야"
COMMANDS = ("화장실로 가자", "안내 끝내줘", "취소해")   # 대기 중 주행·종료·취소
ROUNDS = 3


def prompts() -> list:
    """녹음할 문장 순서. 명령은 같은 말이 연달아 나오지 않게 돌아가며 섞는다."""
    out = [{"kind": "enroll", "text": t} for t in ENROLL]
    out.append({"kind": "wake", "text": WAKE})
    out += [{"kind": "cmd", "text": t} for _ in range(ROUNDS) for t in COMMANDS]
    for i, p in enumerate(out, start=1):
        p["idx"] = i
    return out


def safe_person(name: str) -> str:
    """폴더 이름으로 쓸 사람 이름. 녹음 폴더 밖으로 나가는 이름은 거절한다."""
    name = (name or "").strip()
    if not name or name.startswith(".") or any(c in name for c in "/\\"):
        raise ValueError(f"사람 이름으로 쓸 수 없습니다: {name!r}")
    return name


def save_take(rec_dir: Path, person: str, prompt: dict, six: np.ndarray) -> dict:
    """6채널 int16 녹음을 <rec_dir>/<사람>/<종류>_<번호>.wav 로 쓰고 manifest.jsonl 에 한 줄 붙인다."""
    person = safe_person(person)
    rel = f"{person}/{prompt['kind']}_{prompt['idx']:02d}.wav"
    path = Path(rec_dir) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    six = np.ascontiguousarray(six, dtype=np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(six.shape[1])
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(six.tobytes())
    ch0 = six[:, 0].astype(np.float32) / 32768.0
    line = {"person": person, "kind": prompt["kind"], "idx": prompt["idx"], "text": prompt["text"], "file": rel,
            "time": datetime.now().isoformat(timespec="seconds"),
            "speech_sec": round(len(trim_speech(ch0)) / SAMPLE_RATE, 2),
            "peak": round(float(np.max(np.abs(ch0))) if len(ch0) else 0.0, 4)}
    with open(Path(rec_dir) / "manifest.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return line


def _record(device: int, sec: float) -> np.ndarray:
    import sounddevice as sd

    return sd.rec(int(sec * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16",
                  device=device, blocking=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="화자 검증 녹음기")
    ap.add_argument("--person", required=True, help="말하는 사람 이름(폴더 이름)")
    ap.add_argument("--rec-dir", type=Path, default=REC_ROOT / datetime.now().strftime("%Y%m%d"),
                    help="녹음 폴더. 여러 날 녹음한 사람을 같이 평가하려면 같은 폴더를 준다")
    ap.add_argument("--start", type=int, default=1, help="이 번호 문장부터")
    ap.add_argument("--sec", type=float, default=RECORD_SEC, help="문장당 녹음 시간 s")
    args = ap.parse_args(argv)
    try:
        person = safe_person(args.person)
    except ValueError as exc:
        print(exc)
        return 1

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")    # ⑫ 와 같게 src 가져오기 전에 읽는다
    from src.env_file import EnvEditError, values_from
    from tools.mic_venue_check import SETTLE_SEC, _find_input, _proc_cmdlines, voice_stack_running, write_chip

    running = voice_stack_running(_proc_cmdlines())
    if running:
        print("⑫·⑬ 음성 칸이 켜져 있습니다. 끈 뒤 다시 실행하세요:")
        for line in running[:5]:
            print("   ", line[:120])
        return 1
    try:
        values = values_from(os.environ)
    except EnvEditError as exc:
        print(exc)
        return 1
    device = _find_input()
    print("칩에 운영 .env 값을 씁니다(⑫ 가 켤 때와 같은 일)…")
    for p in write_chip(values):
        print(f"  ⚠ {p}")
    time.sleep(SETTLE_SEC)

    rec_dir = args.rec_dir.expanduser()
    items = [p for p in prompts() if p["idx"] >= args.start]
    total = len(prompts())
    print(f"\n녹음 폴더 {rec_dir} · 사람 '{person}' · {len(items)}문장(문장당 {args.sec:.0f} s)")
    print("손잡이를 잡은 자세(로봇 뒤)에서 평소 목소리로 말합니다.")
    print("Enter = 녹음 시작 · r = 앞 문장 다시 · s = 건너뛰기 · q = 끝내기\n")
    i = 0
    while i < len(items):
        p = items[i]
        kind = {"enroll": "등록", "wake": "호출", "cmd": "명령"}[p["kind"]]
        key = input(f"[{p['idx']}/{total}] {kind} · \"{p['text']}\"  ").strip().lower()
        if key == "q":
            break
        if key == "s":
            i += 1
            continue
        if key == "r":
            i = max(0, i - 1)
            continue
        print("   ● 말씀하세요…", end="", flush=True)
        six = _record(device, args.sec)
        line = save_take(rec_dir, person, p, six)
        if line["speech_sec"] == 0:
            print(f"\r   ⚠ 말소리가 없습니다(최대 크기 {line['peak']:.3f}) — 같은 문장을 다시 녹음합니다.")
            continue
        print(f"\r   → {line['file']} · 말소리 {line['speech_sec']:.2f} s · 최대 크기 {line['peak']:.3f}")
        i += 1
    print(f"\n끝. 평가: ~/vica_data/spk_eval/venv/bin/python -m tools.spk_verify_eval --rec {rec_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

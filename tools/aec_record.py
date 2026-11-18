#!/usr/bin/env python3
"""에코 제거 검증용 6채널 녹음 묶음 — 로봇만 · 사람만 · 로봇 멘트 위에 사람 말(설계 5.3).

  cd ~/VICA-smarthandle/vica-voice-llm && .venv/bin/python -m tools.aec_record [--out 폴더] [--countdown 10]

전제: 음성 노드(⑫·⑬)가 꺼져 있어야 한다(마이크·스피커·칩을 혼자 쓴다). 켜져 있으면 시작하지 않는다.
출력: <out>/set.wav(6ch PCM_16, 16 kHz) · <out>/set.json(시각표 — Task 7 이 읽는다, 꼴을 바꾸지 않는다).
시각은 녹음 첫 표본부터의 초(콜백이 받은 표본 수 ÷ 16000).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ⑫ 와 같게 .env 를 src 가져오기보다 먼저 읽는다 — wakeword_monitor 가 가져오는 순간 환경에서 문턱을 읽는다.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from tools.mic_venue_check import ROBOT_CLIPS, _proc_cmdlines, load_clips, voice_stack_running  # noqa: E402

RATE = 16000
CHANNELS = 6              # ch0 = 칩 처리음 · ch1~4 = 마이크 원음 · ch5 = 재생 참조
BLOCK = 1280
OVERLAP_CLIPS = ("mission_msg_approach_question.wav", "mission_msg_estopped.wav",
                 "mission_msg_paused.wav", "mission_msg_estop_reject.wav")
LEAD_SEC, GAP_SEC, USER_WINDOW_SEC, OVERLAP_TAIL_SEC = 2.0, 0.3, 4.0, 2.0
ROBOT_TAIL_SEC = 0.5      # 로봇만 구간 끝에 더하는 여유
PROMPT_PAUSE_SEC = 1.0    # 할 말을 읽을 시간(다음 소리 전)
BEEP_HZ, BEEP_SEC, BEEP_PEAK_DBFS = 880, 0.15, -20.0
BAKED_DIR = ROOT / "assets" / "baked"
DEFAULT_ROOT = Path.home() / "vica_data" / "aec_sets"


@dataclass(frozen=True)
class Step:
    kind: str                 # "robot_only" | "user_only" | "overlap" | "cue"
    clip: Optional[str]
    say: Optional[str]
    expect: Optional[str]
    item_id: Optional[str]


def build_script() -> list:
    """녹음 순서. 로봇만 16걸음 → 삐 두 번 → 사람만 10 → 삐 두 번 → 겹침 20."""
    steps = [Step("robot_only", name, None, None, "robot_only") for name in ROBOT_CLIPS * 2]
    steps.append(Step("cue", None, None, None, None))
    for prefix, say, expect in (("wake", "비카야 화장실 가자", "wake"), ("stop", "멈춰", "emergency")):
        steps += [Step("user_only", None, say, expect, f"user_{prefix}_{i}") for i in range(1, 6)]
    steps.append(Step("cue", None, None, None, None))
    n = 0
    for prefix, say, expect in (("stop", "멈춰", "emergency"), ("wake", "비카야", "wake")):
        for i in range(1, 11):
            steps.append(Step("overlap", OVERLAP_CLIPS[n % len(OVERLAP_CLIPS)], say, expect, f"over_{prefix}_{i}"))
            n += 1
    return steps


def session_dict(started: str, wav_name: str, chip: dict, ments: list, items: list) -> dict:
    return {"version": 1, "rate": RATE, "channels": CHANNELS, "started": started, "wav": wav_name,
            "chip": chip, "ments": ments, "items": items}


def apply_chip_from_env() -> dict:
    """⑫ 가 기동 때 쓰는 칩 세 값을 같은 순서·같은 환경 키·같은 기본값으로 쓴다. 쓴 값을 돌려준다."""
    from src.dsp_state import (agc_desired_from_env, agc_max_gain_from_env, apply_agc_desired_level,
                               apply_agc_max_gain, apply_echo_tuning, echo_tuning_from_env)
    chip = {}
    desired = agc_desired_from_env(os.environ.get("VICA_MIC_AGC_DESIRED", "0.010"))
    if desired is not None:
        chip["agc_desired"] = {"value": desired, "ok": bool(apply_agc_desired_level(desired))}
    max_gain = agc_max_gain_from_env(os.environ.get("VICA_MIC_AGC_MAX_GAIN", ""))
    if max_gain is not None:
        chip["agc_max_gain"] = {"value": max_gain, "ok": bool(apply_agc_max_gain(max_gain))}
    etail, nlatten = echo_tuning_from_env(os.environ.get("VICA_MIC_GAMMA_ETAIL", ""),
                                          os.environ.get("VICA_MIC_NLATTEN", ""))
    if etail is not None or nlatten is not None:
        wrote = apply_echo_tuning(etail, nlatten)
        chip["echo_tuning"] = {"gamma_etail": etail, "nlatten": nlatten, "wrote": dict(wrote or {}), "ok": bool(wrote)}
    return chip


def _prompt(step: Step, index: int, total: int) -> str:
    if step.kind == "robot_only":
        return f"[로봇만 {index}/{total}] 조용히 계세요 — 로봇 멘트가 나옵니다."
    if step.kind == "user_only":
        return f"[사람만] 삐 소리 뒤 {USER_WINDOW_SEC:g}초 안에 \"{step.say}\" 라고 말하세요 ({step.item_id})."
    if step.kind == "overlap":
        return f"[겹침] 로봇 멘트가 나오는 동안 \"{step.say}\" 라고 말하세요 ({step.item_id})."
    return "[안내] 삐 두 번 — 다음 단계로 넘어갑니다."


def record(out_dir: Path, countdown: float) -> int:
    import soundfile as sf
    import sounddevice as sd
    from src import audio_out

    chip = apply_chip_from_env()                   # 마이크 스트림을 열기 전에(제어 전송 거부 방지)
    clips, missing = load_clips(BAKED_DIR, tuple(dict.fromkeys(ROBOT_CLIPS + OVERLAP_CLIPS)))
    if missing:
        print("멘트 파일이 없습니다:", ", ".join(missing))
        return 1
    clip_by_name = {c.name: c for c in clips}
    if audio_out.output_device() is None:          # 입력을 열기 전에 출력 장치를 먼저 잡아 둔다
        print("reSpeaker 재생 장치를 찾지 못했습니다.")
        return 1
    in_dev = next((i for i, d in enumerate(sd.query_devices())
                   if "respeaker" in d["name"].lower() and d["max_input_channels"] >= CHANNELS), None)
    if in_dev is None:
        print("reSpeaker 6채널 입력 장치를 찾지 못했습니다.")
        return 1

    beep = (0.3 * np.sin(2 * np.pi * BEEP_HZ * np.arange(int(BEEP_SEC * RATE)) / RATE)).astype(np.float32)
    chunks, lock = [], threading.Lock()
    received = [0]

    def cb(indata, frames, t, status):
        with lock:
            chunks.append(indata.copy())
            received[0] += frames

    def now() -> float:
        return received[0] / RATE

    def wait(sec: float) -> None:
        target = now() + sec
        while now() < target:
            time.sleep(0.01)

    def play(wave: np.ndarray, rate: int, **kw) -> tuple:
        start = now()
        audio_out.play(wave, rate, blocking=True, **kw)
        return start, now()

    def beep_once() -> tuple:
        return play(beep, RATE, peak_dbfs=BEEP_PEAK_DBFS)

    steps = build_script()
    total_robot = sum(1 for s in steps if s.kind == "robot_only")
    ments, items = [], []
    started = datetime.now().isoformat(timespec="seconds")
    print(f"세 단계로 녹음합니다(약 {(len(steps) * 5) // 60 + 1}분): 1) 로봇만 — 조용히 · 2) 사람만 — 삐 뒤에 말하기 · 3) 로봇 멘트와 겹쳐 말하기.")
    print(f"{countdown:g}초 뒤 녹음을 시작합니다. 마이크 앞에 서세요.")
    time.sleep(countdown)

    with sd.InputStream(device=in_dev, channels=CHANNELS, samplerate=RATE, dtype="int16", blocksize=BLOCK, callback=cb):
        wait(LEAD_SEC)
        robot_first, robot_last_end, n_robot = None, 0.0, 0
        for step in steps:
            if step.kind == "robot_only":
                n_robot += 1
                print(_prompt(step, n_robot, total_robot))
                clip = clip_by_name[step.clip]
                start, end = play(clip.pcm, clip.rate)
                ments.append({"name": clip.name, "text": clip.text, "start": round(start, 3), "end": round(end, 3)})
                robot_first = start if robot_first is None else robot_first
                robot_last_end = end
                wait(GAP_SEC)
            elif step.kind == "cue":
                if robot_first is not None and not items:
                    wait(max(0.0, ROBOT_TAIL_SEC - GAP_SEC))
                    items.append({"id": "robot_only", "kind": "robot_only", "say": None, "expect": None,
                                  "start": round(robot_first, 3), "end": round(robot_last_end + ROBOT_TAIL_SEC, 3)})
                print(_prompt(step, 0, 0))
                beep_once()
                wait(GAP_SEC)
                beep_once()
                wait(PROMPT_PAUSE_SEC)
            elif step.kind == "user_only":
                print(_prompt(step, 0, 0))
                wait(PROMPT_PAUSE_SEC)
                _, beep_end = beep_once()
                wait(USER_WINDOW_SEC)
                items.append({"id": step.item_id, "kind": "user_only", "say": step.say, "expect": step.expect,
                              "start": round(beep_end, 3), "end": round(beep_end + USER_WINDOW_SEC, 3)})
            elif step.kind == "overlap":
                print(_prompt(step, 0, 0))
                wait(PROMPT_PAUSE_SEC)
                clip = clip_by_name[step.clip]
                start, end = play(clip.pcm, clip.rate)
                ments.append({"name": clip.name, "text": clip.text, "start": round(start, 3), "end": round(end, 3)})
                wait(OVERLAP_TAIL_SEC)
                items.append({"id": step.item_id, "kind": "overlap", "say": step.say, "expect": step.expect,
                              "start": round(start, 3), "end": round(end + OVERLAP_TAIL_SEC, 3)})
        wait(0.3)

    with lock:
        data = np.concatenate(chunks)
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / "set.wav", data, RATE, subtype="PCM_16")
    doc = session_dict(started, "set.wav", chip, ments, items)
    (out_dir / "set.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"녹음 끝: {out_dir} (set.wav {len(data) / RATE:.1f}초 · set.json 항목 {len(items)}개)")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="에코 제거 검증용 6채널 녹음 묶음(로봇만·사람만·겹침)과 시각표 json")
    ap.add_argument("--out", type=Path, default=None, help="저장 폴더(기본 ~/vica_data/aec_sets/<YYYYmmdd-HHMMSS>)")
    ap.add_argument("--countdown", type=float, default=10.0, help="녹음 시작 전 대기 초(기본 10)")
    args = ap.parse_args(argv)
    running = voice_stack_running(_proc_cmdlines())
    if running:
        print("⑫·⑬ 음성 칸이 켜져 있습니다. 끈 뒤 다시 실행하세요:")
        for line in running[:5]:
            print("   ", line[:120])
        return 1
    out_dir = args.out or DEFAULT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    return record(out_dir, args.countdown)


if __name__ == "__main__":
    raise SystemExit(main())

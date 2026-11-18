#!/usr/bin/env python3
"""에코 제거 다시 틀기 — 6채널 녹음을 운영 감시기에 off·on 으로 흘려 "멈춰"·"비카야" 를 견준다(설계 5.2·5.4).

  cd ~/VICA-smarthandle/vica-voice-llm && .venv/bin/python -m tools.aec_replay <set.json> [--env .env] [--out replay.json] [--force]

전제: 음성 노드가 꺼져 있어야 한다(GPU·모델을 같이 쓴다). 끝 코드 0 = 판정 통과, 1 = 아님.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="6채널 녹음을 운영 감시기에 에코 제거 off·on 으로 흘려 견준다")
    ap.add_argument("session", type=Path, help="tools.aec_record 가 쓴 시각표 json")
    ap.add_argument("--env", type=Path, default=ROOT / ".env", help="마이크 값을 읽을 .env(기본 저장소 .env)")
    ap.add_argument("--out", type=Path, default=None, help="결과 json(기본 시각표 옆 replay.json)")
    ap.add_argument("--force", action="store_true", help="음성 칸이 켜져 있어도 돌린다")
    args = ap.parse_args(argv)
    # 감시기 모듈은 가져오는 순간 환경에서 문턱을 읽는다 — .env 를 먼저 올리고 tools·src 를 가져온다.
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if args.env != ROOT / ".env":
        load_dotenv(args.env, override=True)
    from tools.mic_venue_check import _proc_cmdlines, voice_stack_running
    running = voice_stack_running(_proc_cmdlines())
    if running and not args.force:
        print("⑫·⑬ 음성 칸이 켜져 있습니다(GPU·모델을 같이 씁니다). 끈 뒤 다시 실행하거나 --force 를 주세요:")
        for line in running[:5]:
            print("   ", line[:120])
        return 1

    import soundfile as sf

    from src.aec_replay import (format_report, frames_for, judge, load_session, ment_reductions, run_item,
                                split_blocks)
    from src.echo_cancel import echo_config_from_env
    from src.env_file import read_env, values_from
    from tools.mic_venue_check import _load_models

    session = load_session(args.session)
    values = values_from(read_env(args.env))
    config = echo_config_from_env(os.environ)
    x6, rate = sf.read(str(session.wav), dtype="int16", always_2d=True)
    if rate != 16000 or x6.shape[1] != 6:
        print(f"녹음 형식이 맞지 않습니다: {rate} Hz · {x6.shape[1]} 채널 (16000 Hz · 6 채널이어야 합니다)")
        return 1
    blocks = split_blocks(x6)
    try:
        frames = {mode: frames_for(blocks, mode, config) for mode in ("off", "on")}
    except RuntimeError as exc:
        print(str(exc))
        return 1
    models = _load_models(None)
    results = {}
    for mode in ("off", "on"):
        results[mode] = {}
        for item in session.items:
            results[mode][item.id] = run_item(models, values, frames[mode], item, session.ments)
            print(f"  {mode} {item.id}")
    reductions = ment_reductions(session, frames["off"], frames["on"])
    out = args.out or args.session.with_name("replay.json")
    body = {"reductions": reductions, "off": {k: asdict(v) for k, v in results["off"].items()},
            "on": {k: asdict(v) for k, v in results["on"].items()}}
    out.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")   # 판정 전에 먼저 남긴다
    report = judge(session, results["off"], results["on"], reductions)
    out.write_text(json.dumps({"report": report, **body}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(format_report(report))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

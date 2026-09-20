#!/usr/bin/env python3
"""소리→의도 직행 종단 스모크(실제 OpenAI 호출, 수동 실행 전용).

로봇 TTS(supertonic)로 시험 문장을 합성해 parse_intent_audio 에 넣는다 — 마이크가
없어도 도는 종단 점검이다. 사용: .venv/bin/python tools/realtime_intent_smoke.py [문장...]
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.destination_loader import load_destinations  # noqa: E402
from src.langchain_intent_parser import parse_intent_audio  # noqa: E402
from src.realtime_intent import float32_to_pcm16  # noqa: E402
from src.tts import VicaTTS  # noqa: E402

DEFAULT = ["화장실로 안내해줘", "오 분", "한 시간", "그래", "아니요", "여기 몇 층이야", "테스트3으로 가자"]


def synth(tts, text: str) -> bytes:
    audio, rate = tts.synthesize(text)   # src/tts.py: VicaTTS.synthesize -> (파형, 샘플레이트)
    audio = np.asarray(audio, dtype=np.float32)
    if rate != 16000:
        x = np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / rate))
        audio = np.interp(x, np.arange(len(audio)), audio).astype(np.float32)
    return float32_to_pcm16(audio)


def main() -> None:
    dests = load_destinations()
    tts = VicaTTS()
    for text in sys.argv[1:] or DEFAULT:
        pcm = synth(tts, text)
        t0 = time.monotonic()
        try:
            intent, heard, dt = parse_intent_audio(pcm, dests)
            print(f"[{time.monotonic()-t0:5.2f}s rt={dt:4.2f}s] '{text}' → heard='{heard}' "
                  f"intent={intent.intent} dest={intent.matched_destination_id or '-'} reply='{intent.reply[:30]}'")
        except Exception as exc:
            print(f"[실패] '{text}': {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()

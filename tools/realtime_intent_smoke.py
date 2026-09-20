#!/usr/bin/env python3
"""소리→의도 직행 종단 스모크(실제 OpenAI 호출, 수동 실행 전용).

로봇 TTS(supertonic)로 시험 문장을 합성해 parse_intent_audio 에 넣는다 — 마이크가
없어도 도는 종단 점검이다. 사용: .venv/bin/python tools/realtime_intent_smoke.py [문장...]
"""
import sys
import time
from pathlib import Path

import numpy as np
from langchain_core.messages import AIMessage, HumanMessage

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


def run_case(tts, dests, text: str, history=None, label: str = "") -> None:
    tag = label or text
    pcm = synth(tts, text)
    t0 = time.monotonic()
    try:
        intent, heard, dt, info = parse_intent_audio(pcm, dests, history=history)
        print(f"[{time.monotonic()-t0:5.2f}s rt={dt:4.2f}s src={info['src']}] '{tag}' → heard='{heard}' "
              f"intent={intent.intent} dest={intent.matched_destination_id or '-'} nc={intent.need_confirm} "
              f"reply='{intent.reply[:30]}'")
    except Exception as exc:
        print(f"[실패] '{tag}': {type(exc).__name__}: {exc}")


def main() -> None:
    dests = load_destinations()
    tts = VicaTTS()
    for text in sys.argv[1:] or DEFAULT:
        run_case(tts, dests, text)

    # 이력 케이스 (항목 G): 직전에 로봇이 첫 목적지의 confirm_prompt 로 확인
    # 질문을 한 상태에서 "그래" — 확인 대기 확정 경로(navigate/need_confirm=False)가
    # 나와야 한다. 인자 없이 돌려도, 인자로 "그래"만 줘도 항상 함께 찍힌다.
    if dests:
        confirm_prompt = dests[0].confirm_prompt
        history = [HumanMessage("화장실"), AIMessage(confirm_prompt)]
        run_case(tts, dests, "그래", history=history, label="[이력] 그래")
    else:
        print("[이력 케이스 생략] 목적지 목록이 비어 있다")


if __name__ == "__main__":
    main()

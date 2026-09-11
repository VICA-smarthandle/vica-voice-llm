"""VICA 음성/LLM 파이프라인 CLI 프로토타입."""
from __future__ import annotations

import os

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .destination_loader import load_destinations
from .emergency_filter import EMERGENCY_REPLY, detect_emergency
from .langchain_intent_parser import parse_intent
from .replies import USAGE_GUIDE

EXIT_WORDS = {"종료", "그만", "exit", "quit"}
MAX_HISTORY = 8


def run(use_tts: bool = True, use_stt: bool = False) -> None:
    destinations = load_destinations()
    history: list[BaseMessage] = []
    robot_state = None

    tts = None
    if use_tts:
        try:
            from .tts import VicaTTS

            print("TTS 모델 로드 중...")
            tts = VicaTTS()
        except Exception as exc:
            print(f"TTS 비활성화 (로드 실패: {exc})")

    stt = None
    if use_stt:
        try:
            from .stt import VicaSTT

            print("STT 모델 로드 중...")
            stt = VicaSTT()
        except Exception as exc:
            print(f"STT 비활성화 (로드 실패: {exc})")

    mode = "마이크" if stt else "키보드"
    print(f"VICA 음성 파이프라인 프로토타입 [{mode} 입력] (종료하려면 '종료')")
    print(f"목적지 {len(destinations)}개 로드됨\n")
    print(f"VICA > {USAGE_GUIDE}\n")
    if tts:
        tts.speak(USAGE_GUIDE)

    while True:
        try:
            if stt:
                input("나 > [엔터를 누르면 녹음 시작] ")
                text = stt.listen().strip()
                print(f"나(인식) > {text}")
            else:
                text = input("나 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue
        if text in EXIT_WORDS:
            break

        keyword = detect_emergency(text)
        if keyword:
            print(f"VICA > [긴급] '{keyword}' 감지 — 즉시 정지 신호 (safety_flag=emergency)\n")
            if tts:
                tts.speak(EMERGENCY_REPLY)
            history.append(HumanMessage(text))
            history.append(AIMessage(EMERGENCY_REPLY))
            history[:] = history[-MAX_HISTORY:]
            continue

        intent = parse_intent(text, destinations, history=history, robot_state=robot_state)
        print(f"VICA > {intent.reply}")
        print(
            f"        (intent={intent.intent}, "
            f"matched={intent.matched_destination_id}, "
            f"need_confirm={intent.need_confirm}, "
            f"safety={intent.safety_flag})\n"
        )
        if tts:
            tts.speak(intent.reply)

        history.append(HumanMessage(text))
        history.append(AIMessage(intent.reply))
        history[:] = history[-MAX_HISTORY:]


if __name__ == "__main__":
    run(
        use_tts=os.environ.get("VICA_TTS", "1") != "0",
        use_stt=os.environ.get("VICA_STT", "0") == "1",
    )

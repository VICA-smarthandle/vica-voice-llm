"""VICA 음성/LLM 파이프라인 CLI 프로토타입.

마이크/STT 없이 '키보드 텍스트'로 전체 흐름을 검증한다.

  키보드 입력
    -> emergency_filter   (LLM 전 긴급어 차단 = 안전 경로)
    -> parse_intent       (Ollama Cloud, 멀티턴 대화)
    -> VicaIntent 출력

실행 (프로젝트 루트에서):
    .venv/bin/python -m src.main
"""
from __future__ import annotations

import os

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .destination_loader import load_destinations
from .emergency_filter import EMERGENCY_REPLY, detect_emergency
from .langchain_intent_parser import parse_intent
from .replies import USAGE_GUIDE

EXIT_WORDS = {"종료", "그만", "exit", "quit"}
MAX_HISTORY = 8  # 최근 메시지만 유지 (대화가 길어져도 프롬프트가 커지지 않게)


def run(use_tts: bool = True, use_stt: bool = False) -> None:
    destinations = load_destinations()
    history: list[BaseMessage] = []
    # CLI 프로토타입에는 실로봇 상태가 없다. parse_intent 는 robot_state=None 을
    # "현재 위치 알 수 없음" 으로 처리한다. 실제 상태는 ROS2 경로(src/ros_node.py 의
    # /vica/robot_state 구독)에서만 채운다.
    robot_state = None

    tts = None
    if use_tts:
        try:
            from .tts import VicaTTS

            print("TTS 모델 로드 중...")
            tts = VicaTTS()
        except Exception as exc:  # 모델 로드 실패해도 텍스트 모드로 계속.
            print(f"TTS 비활성화 (로드 실패: {exc})")

    stt = None
    if use_stt:
        try:
            from .stt import VicaSTT

            print("STT 모델 로드 중...")
            stt = VicaSTT()
        except Exception as exc:  # 모델 로드 실패해도 키보드 모드로 계속.
            print(f"STT 비활성화 (로드 실패: {exc})")

    mode = "마이크" if stt else "키보드"
    print(f"VICA 음성 파이프라인 프로토타입 [{mode} 입력] (종료하려면 '종료')")
    print(f"목적지 {len(destinations)}개 로드됨\n")
    # 첫 상호작용에서 로봇이 자기 어휘를 가르쳐 준다 — 어휘를 모르면 본능적으로
    # "멈춰"(E-stop)를 쓰게 된다. ROS 경로의 인사 연결은 웨이크워드 작업 뒤 후속.
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

        # 1) 긴급어는 LLM 을 거치지 않고 즉시 처리한다 (안전 경로).
        keyword = detect_emergency(text)
        if keyword:
            print(f"VICA > [긴급] '{keyword}' 감지 — 즉시 정지 신호 (safety_flag=emergency)\n")
            if tts:
                tts.speak(EMERGENCY_REPLY)
            history.append(HumanMessage(text))
            history.append(AIMessage(EMERGENCY_REPLY))
            history[:] = history[-MAX_HISTORY:]
            continue

        # 2) 일반 발화는 LLM intent 파서로 해석한다.
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

        # 3) 대화 히스토리를 갱신한다 (멀티턴용).
        history.append(HumanMessage(text))
        history.append(AIMessage(intent.reply))
        history[:] = history[-MAX_HISTORY:]


if __name__ == "__main__":
    run(
        use_tts=os.environ.get("VICA_TTS", "1") != "0",
        use_stt=os.environ.get("VICA_STT", "0") == "1",
    )

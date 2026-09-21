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


def run_case(tts, dests, text: str, history=None, label: str = "", pcm=None) -> None:
    tag = label or text
    pcm = pcm if pcm is not None else synth(tts, text)
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

    # 모델 전결 케이스 (2026-09-20): 정정·미션 질문의 답·사람 말 아님.
    if len(dests) >= 2:
        a, b = dests[0], dests[1]
        run_case(tts, dests, f"아니 {b.name}로 가자",
                 history=[HumanMessage(a.name), AIMessage(a.confirm_prompt)], label=f"[정정] 아니 {b.name}로 가자")
        arrival = [HumanMessage(a.name), AIMessage(a.confirm_prompt), HumanMessage("그래"),
                   AIMessage(f"{a.name}로 안내를 시작합니다."),
                   AIMessage(f"{a.name} 앞에 도착했습니다. 여기서 대기할까요?"),
                   HumanMessage("응"), AIMessage("몇 분쯤 걸리실까요?")]
        run_case(tts, dests, "오 분", history=arrival, label="[도착 질문] 오 분")
        import numpy as np
        noise = (np.random.default_rng(0).normal(0, 300, 16000 * 2)).astype("<i2").tobytes()
        run_case(tts, dests, "", history=arrival, label="[잡음 2초]", pcm=noise)

    # 대장(P1) 케이스: [지금 상황] 블록만으로 다섯 질문에 답하는지.
    from src.building_directory import DirectoryEntry, format_directory_block
    from src.ledger_view import render_ledger
    from src.schema import RobotState
    st = RobotState(current_floor=4, current_building="로봇관", dialog_state="waiting", place_here="407호 앞",
                    place_here_dist_m=1.2, last_destination="407호", last_arrived_age_sec=660,
                    aborted_destination="화장실", wait_minutes=10, wait_left_sec=300)
    situation = render_ledger(st, awaiting_answer=False, now_text="15:40")
    directory = format_directory_block([DirectoryEntry("세미나실", "로봇관", 3)], "로봇관", 4)
    for q in ("우리 몇 층이야", "지금 어디 있어", "아까 어디 갔었지", "어디 가려고 했더라", "지금 몇 시야",
              "세미나실 갈 수 있어"):
        pcm = synth(tts, q)
        try:
            intent, heard, dt, info = parse_intent_audio(pcm, dests, situation=situation, directory_block=directory)
            print(f"[대장] '{q}' → intent={intent.intent} reply='{intent.reply[:40]}' dt={dt:.2f}s")
        except Exception as exc:
            print(f"[대장 실패] '{q}': {type(exc).__name__}: {exc}")

    if not dests:
        print("[이력 케이스 생략] 목적지 목록이 비어 있다")


if __name__ == "__main__":
    main()

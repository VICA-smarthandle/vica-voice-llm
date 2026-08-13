"""현행 파이프라인(A 기준선) 전 구간 계측 하네스 — docs/handoff-gpt-realtime-test.md 3-0.

## 무엇을 재는가 (한 회차 = jsonl 한 줄)

  stt_sec              발화 종료(녹음 엔터) → STT 텍스트            (--mic 일 때만)
  llm_sec              텍스트 → VicaIntent
  tts_synth_sec        intent → 재생 시작 (합성)
  tts_play_button_sec  재생 시작 → **사람이 Enter 로 찍은 재생 종료** (1차 수치)
  tts_audio_sec        합성 파형 길이 (기계 참조치)
  total_button_sec     발화 종료 → 버튼 종료 (전 구간 체감)
  so_fail              구조화 출력(SO) 실패 — reply 가 LLM_UNAVAILABLE fallback 이면 True

t_start/t_end 는 epoch 초다 — tegrastats 로그(인수인계 2절)와 구간 대조에 쓴다.

## 왜 speak() 을 안 쓰나

VicaTTS.speak() 은 합성과 재생을 한 호출에 묶어 '재생 시작' 경계를 잴 수 없다.
src/ 수정은 금지(인수인계 4절)이므로 speak() 과 같은 재생 코드(sd.play)를 여기로
옮기고 _synthesize() 를 그대로 쓴다.

## 버튼 프로토콜

재생이 시작되면 스크립트가 대기한다. **스피커에서 소리가 끝나는 순간 Enter.**
프로세스 타임스탬프는 오디오 버퍼 잔량을 놓치므로 사람 귀가 1차 수치이고, 반응
오차(~±0.2초)는 반복 중앙값으로 흡수한다 (인수인계 3-0).

## 실행 (프로젝트 루트에서. tegrastats 를 먼저 켜 둔다)

    .venv/bin/python scripts/pipeline_probe.py --set --runs 10        # 텍스트 세트 6종 x 10회
    .venv/bin/python scripts/pipeline_probe.py --set --mic --runs 5   # 음성 세트 6종 x 5회
    .venv/bin/python scripts/pipeline_probe.py --text "화장실로 가줘"
    .venv/bin/python scripts/pipeline_probe.py --mic                  # 자유 발화 1회
    .venv/bin/python scripts/pipeline_probe.py --say "화장실로 안내해드릴까요?"
    # --say: STT·LLM 없이 TTS+버튼만 — B(realtime) 변형의 TTS 구간을 같은
    #        프로토콜로 잴 때 탐침이 출력한 reply 를 그대로 넣는다

## [미검증] 2026-08-13 작성, 실측 전이다. 실기에서 고친 것은 인수인계 7절에 적어라.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 인수인계 3-3 발화 세트. 기대값은 화면 표기용이고 판정은 사람이 7절에 적는다.
# ('cancel' 은 현행 VicaIntentType 에 없다 — A 에서 다른 값이 나오는 것 자체가 기록이다.)
UTTERANCE_SET = [
    ("화장실로 가줘", "navigate/화장실"),
    ("배가 아파요", "navigate(간접)/화장실"),
    ("여기 몇 층이야?", "question"),
    ("저기로 가줘", "clarify"),
    ("취소해줘", "cancel — 현행 enum 에 없음"),
    ("밥 먹을 데 있어?", "navigate 또는 question/식당"),
]


def _load(need_stt: bool, need_tts: bool, need_llm: bool):
    """모델을 한 번만 로드한다. 로드 시간은 콜드스타트 참고치로 화면에 남긴다."""
    destinations = None
    if need_llm:
        from src.destination_loader import load_destinations

        destinations = load_destinations()
        print(f"[로드] 목적지 {len(destinations)}개")

    stt = None
    if need_stt:
        t0 = time.perf_counter()
        from src.stt import VicaSTT

        stt = VicaSTT()
        print(f"[로드] STT {time.perf_counter() - t0:.1f}초")

    tts = None
    if need_tts:
        t0 = time.perf_counter()
        from src.tts import VicaTTS

        tts = VicaTTS()
        print(f"[로드] TTS {time.perf_counter() - t0:.1f}초")

    from src.metrics import sample_system

    sample_system()  # psutil cpu_percent 프라이밍 (첫 호출은 0 이 나온다)
    return destinations, stt, tts


def _speak_with_button(tts, reply: str, rec: dict, t_utter_end: float) -> None:
    """합성 → 재생 → 사람이 Enter 로 재생 종료를 찍는다 (1차 수치)."""
    import sounddevice as sd

    t0 = time.perf_counter()
    wav, sample_rate = tts._synthesize(reply)  # speak() 과 동일 경로 (머리 주석 참고)
    t_play = time.perf_counter()
    rec["tts_synth_sec"] = round(t_play - t0, 3)
    rec["tts_audio_sec"] = round(len(wav) / sample_rate, 3)
    sd.play(wav, samplerate=sample_rate)  # 비차단 — Enter 대기와 동시에 재생
    input("  👉 스피커 소리가 끝나는 순간 Enter: ")
    t_button = time.perf_counter()
    sd.wait()  # 일찍 눌렀어도 재생은 끝까지 (다음 회차 소리와 겹치지 않게)
    rec["tts_play_button_sec"] = round(t_button - t_play, 3)
    rec["total_button_sec"] = round(t_button - t_utter_end, 3)
    print(
        f"  [TTS] 합성 {rec['tts_synth_sec']}초 / 재생[버튼] {rec['tts_play_button_sec']}초"
        f" (파형 {rec['tts_audio_sec']}초) / 전체[버튼] {rec['total_button_sec']}초"
    )


def _measure_once(text, destinations, stt, tts, label: str) -> dict:
    """한 회차: (텍스트 또는 마이크) → 긴급어 필터 → intent → TTS+버튼."""
    from src.emergency_filter import detect_emergency
    from src.langchain_intent_parser import parse_intent
    from src.metrics import sample_system
    from src.replies import LLM_UNAVAILABLE

    rec: dict = {"t_start": round(time.time(), 3), "label": label}

    if stt is not None:
        input(f"🎤 [{label}] 엔터를 누르면 녹음 시작 → 발화 → 엔터로 종료: ")
        audio = stt.record_until_enter()
        t_utter_end = time.perf_counter()  # 발화 종료 = 녹음 종료 엔터
        text = stt.transcribe(audio)
        rec["stt_sec"] = round(time.perf_counter() - t_utter_end, 3)
        print(f"  [STT] {rec['stt_sec']}초: {text!r}")
    else:
        t_utter_end = time.perf_counter()
    rec["utterance"] = text

    if text:
        # 파이프라인 순서 그대로: 긴급어는 LLM 이전 차단 (세트에는 없지만 순서를 지킨다)
        keyword = detect_emergency(text)
        if keyword:
            rec["emergency"] = keyword
            print(f"  [긴급] '{keyword}' — LLM 미호출 (안전 경로)")
        else:
            t0 = time.perf_counter()
            # history=[] 단독 발화 — B 탐침도 단독 발화라 조건을 맞춘다
            intent = parse_intent(text, destinations, history=[], robot_state=None)
            rec["llm_sec"] = round(time.perf_counter() - t0, 3)
            rec["intent"] = intent.intent
            rec["matched"] = intent.matched_destination_id
            rec["need_confirm"] = intent.need_confirm
            rec["confidence"] = intent.confidence
            rec["reply"] = intent.reply
            rec["so_fail"] = intent.reply == LLM_UNAVAILABLE  # 3-2 의 A(SO) 쪽 실패 집계
            print(
                f"  [LLM] {rec['llm_sec']}초: intent={intent.intent}"
                f" matched={intent.matched_destination_id} reply={intent.reply!r}"
            )
            if tts is not None and intent.reply:
                _speak_with_button(tts, intent.reply, rec, t_utter_end)

    rec["sys"] = sample_system()  # 참조용 스냅샷 — 정식 연산량은 tegrastats (3-5)
    rec["t_end"] = round(time.time(), 3)
    return rec


def _summary(records: list[dict]) -> None:
    print(f"\n[요약] n={len(records)}")
    for key in ("stt_sec", "llm_sec", "tts_synth_sec", "tts_play_button_sec", "total_button_sec"):
        vals = [r[key] for r in records if key in r]
        if vals:
            print(f"  {key:22s} 중앙값 {statistics.median(vals):.3f}초 (n={len(vals)})")
    judged = [r for r in records if "so_fail" in r]
    if judged:
        fails = sum(1 for r in judged if r["so_fail"])
        print(f"  so_fail                {fails}/{len(judged)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--text", help="단일 발화를 텍스트로 넣는다")
    parser.add_argument("--set", action="store_true", dest="use_set", help="문서 3-3 발화 세트 6종")
    parser.add_argument("--mic", action="store_true", help="마이크 입력 (STT 경유)")
    parser.add_argument("--say", help="STT·LLM 없이 TTS+버튼만 (B 변형 TTS 구간용)")
    parser.add_argument("--runs", type=int, default=1, help="반복 횟수 (기본 1)")
    parser.add_argument("--no-tts", action="store_true", help="TTS 단계 생략")
    parser.add_argument("--out", default="logs/pipeline_probe.jsonl", help="jsonl 기록 경로")
    args = parser.parse_args()

    if args.say and (args.text or args.use_set or args.mic):
        parser.error("--say 는 단독으로 쓴다")
    if args.text and args.mic:
        parser.error("--mic 은 --set 과 함께 쓰거나 단독(자유 발화)으로 쓴다")
    if not (args.say or args.text or args.use_set or args.mic):
        parser.error("--text / --set / --mic / --say 중 하나는 필요하다")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    need_tts = not args.no_tts
    destinations, stt, tts = _load(
        need_stt=args.mic, need_tts=need_tts, need_llm=not args.say
    )
    if args.say and tts is None:
        print("--say 는 TTS 가 필요하다 (--no-tts 와 함께 쓸 수 없다)", file=sys.stderr)
        return 1

    if args.use_set:
        items = UTTERANCE_SET
    elif args.text:
        items = [(args.text, "")]
    elif args.mic:
        items = [(None, "자유 발화")]
    else:
        items = [(None, "say")]

    records: list[dict] = []
    try:
        for i in range(args.runs):
            for text, expected in items:
                label = f"{i + 1}/{args.runs}" + (f" {expected}" if expected else "")
                if args.mic and text:
                    print(f"\n다음 발화를 말한다: {text!r} (기대: {expected})")
                if args.say:
                    rec = {"t_start": round(time.time(), 3), "label": label, "reply": args.say}
                    _speak_with_button(tts, args.say, rec, time.perf_counter())
                    rec["t_end"] = round(time.time(), 3)
                else:
                    rec = _measure_once(text, destinations, stt, tts, label)
                records.append(rec)
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except KeyboardInterrupt:
        print("\n중단됨 — 여기까지의 회차는 기록됐다")

    _summary(records)
    print(f"\n기록: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

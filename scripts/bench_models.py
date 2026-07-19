#!/usr/bin/env python3
"""LLM 모델별 intent 해석 품질·속도 벤치마크 (실험용, 자동화 테스트 아님).

사용 (저장소 루트에서):
    .venv/bin/python scripts/bench_models.py                    # .env 의 기본 모델
    .venv/bin/python scripts/bench_models.py exaone3.5:2.4b gemma4:e2b

- 로컬/클라우드 선택은 .env 의 OLLAMA_HOST (+OLLAMA_API_KEY) 를 따른다.
  클라우드 모델과 로컬 모델을 한 번에 섞어 비교할 수는 없다 (호스트가 하나).
- 각 발화는 독립 호출(히스토리 없음). 짧은 긍정("네")은 규칙 처리라 제외.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.destination_loader import load_destinations  # noqa: E402
from src.langchain_intent_parser import DEFAULT_MODEL, OLLAMA_HOST, parse_intent  # noqa: E402

# (발화, 기대 intent, 기대 matched_destination_id)  — None 은 "매칭 없어야 함"
CASES = [
    ("407호로 안내해줘", "navigate", "engineering_4f_room_407_prof_yoon_jiyoung_office"),
    ("윤지영 교수님 사무실로 가줘", "navigate", "engineering_4f_room_407_prof_yoon_jiyoung_office"),
    ("407화로 가죠", "navigate", "engineering_4f_room_407_prof_yoon_jiyoung_office"),  # STT 오인식 견고성
    ("화장실로 안내해줘", "navigate", "starlight_1f_restroom"),
    ("안내센터로 가고 싶어요", "navigate", "starlight_1f_information_center"),
    ("커피 마시는 곳으로 가줘", "navigate", "starlight_1f_cafe"),  # alias 매칭
    ("여기가 몇 층이에요?", "question", None),
    ("으로 가주세요", "clarify", None),  # STT 부분 인식
]


def bench(model: str) -> None:
    destinations = load_destinations()
    print(f"\n=== {model} (host: {OLLAMA_HOST}) ===")
    ok = 0
    times: list[float] = []
    for text, want_intent, want_dest in CASES:
        t0 = time.monotonic()
        result = parse_intent(text, destinations, model=model)
        dt = time.monotonic() - t0
        times.append(dt)
        # clarify/unknown 은 서로 바꿔 답해도 실용상 무해로 본다
        intent_ok = (result.intent == want_intent) or (
            want_intent in ("clarify", "unknown") and result.intent in ("clarify", "unknown")
        )
        dest_ok = (result.matched_destination_id or None) == want_dest
        passed = intent_ok and dest_ok
        ok += passed
        mark = "O" if passed else "X"
        print(f" [{mark}] {dt:5.2f}s  '{text}' -> {result.intent}"
              f" / {result.matched_destination_id or '-'}")
    avg = sum(times) / len(times)
    print(f" 점수: {ok}/{len(CASES)}  평균 {avg:.2f}s  최악 {max(times):.2f}s")


def main() -> None:
    models = sys.argv[1:] or [DEFAULT_MODEL]
    for model in models:
        bench(model)


if __name__ == "__main__":
    main()

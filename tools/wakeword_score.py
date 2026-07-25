"""웨이크워드 측정의 채점 로직 (마이크·whisper 없이 돌아가는 순수 계산).

측정 도구가 잘못 채점하면 없느니만 못하다. 틀린 숫자를 믿고 결정하게 되기
때문이다. 그래서 판정·집계·표 출력을 마이크 코드와 분리해 단위 테스트로 검증한다.
실제 녹음과 STT는 wakeword_check.py 가 담당한다.

판정 규칙은 운영 긴급어와 **같은 규칙**을 쓴다. 다르게 채점하면 "실제로 도입했을
때 어떻게 동작하는가"를 재는 의미가 없어진다.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Sequence

# 운영 긴급어와 동일한 낱말 첫머리 판정을 쓴다. 사설 함수를 가져오는 이유는
# 측정 유효성 때문이다 — 채점 규칙이 운영과 다르면 결과를 믿을 수 없다.
from src.emergency_filter import _starts_at_token_boundary

# 깨우는 말 후보. "비카야"만 정확히 볼지, 받아쓰기 흔들림까지 인정할지에 따라
# 인식률과 오인율이 함께 움직인다. 넓힐수록 잘 깨지만 헛깨움도 늘어난다.
DEFAULT_WAKE_WORD = "비카야"
DEFAULT_VARIANTS: tuple[str, ...] = ("비카야",)

# 잠정 판정 기준. 팀이 확정하기 전까지의 참고선이며 절대 기준이 아니다.
GOOD_HIT_RATE = 0.90      # 목표 문구를 이 정도는 잡아야 쓸 만하다
POOR_HIT_RATE = 0.70      # 이 아래면 그대로 쓰기 어렵다
WARN_FALSE_RATE = 0.05    # 함정 문구를 이 이상 잡으면 위험하다


@dataclass(frozen=True)
class Trial:
    """시도 한 번의 기록."""

    phrase: str      # 말하라고 지시한 문구
    should_wake: bool  # 이 문구는 로봇을 깨워야 하는가
    heard: str       # whisper 가 받아쓴 결과 (빈 문자열 가능)
    matched: Optional[str]  # 깨움으로 판정된 후보 (아니면 None)
    rms: float = 0.0        # 녹음 음량
    # 음량이 낮아 STT 를 돌리지 않은 경우. 운영 감시도 같은 게이트를 쓰므로
    # 이건 "목소리가 작아서 아예 못 들었다"는 별개의 실패 방식이다.
    # (조용한 오디오에 whisper 를 돌리면 없는 말을 지어낸다 — 환청)
    gated: bool = False

    @property
    def woke(self) -> bool:
        return self.matched is not None

    @property
    def correct(self) -> bool:
        return self.woke == self.should_wake


def match_wake_word(
    text: str, variants: Sequence[str] = DEFAULT_VARIANTS
) -> Optional[str]:
    """받아쓴 문장에 깨우는 말이 있으면 그 후보를, 없으면 None.

    긴 낱말에 우연히 섞인 경우는 인정하지 않는다(운영 긴급어와 같은 규칙).
    """
    if not text:
        return None
    for variant in variants:
        if not variant:
            continue
        if _starts_at_token_boundary(text, variant):
            return variant
    return None


@dataclass
class PhraseResult:
    """문구 하나에 대한 집계."""

    phrase: str
    should_wake: bool
    total: int = 0
    woke: int = 0
    gated: int = 0  # 음량 미달로 STT 까지 못 간 횟수
    heard_texts: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        """깨어난 비율. 목표 문구면 인식률, 함정 문구면 오인율이다."""
        return self.woke / self.total if self.total else 0.0

    @property
    def label(self) -> str:
        return "인식률" if self.should_wake else "오인율"

    @property
    def verdict(self) -> str:
        if self.total == 0:
            return "측정 없음"
        if self.should_wake:
            if self.rate >= GOOD_HIT_RATE:
                return "좋음"
            return "개선 필요" if self.rate >= POOR_HIT_RATE else "부적합"
        if self.rate == 0.0:
            return "좋음"
        return "주의" if self.rate <= WARN_FALSE_RATE else "부적합"


def summarize(trials: Sequence[Trial]) -> list[PhraseResult]:
    """시도 기록을 문구별로 묶는다. 지시한 순서를 유지한다."""
    results: dict[str, PhraseResult] = {}
    for trial in trials:
        result = results.get(trial.phrase)
        if result is None:
            result = PhraseResult(phrase=trial.phrase, should_wake=trial.should_wake)
            results[trial.phrase] = result
        result.total += 1
        if trial.woke:
            result.woke += 1
        if trial.gated:
            result.gated += 1
        result.heard_texts.append(trial.heard)
    return list(results.values())


def heard_counts(trials: Sequence[Trial], phrase: str) -> list[tuple[str, int]]:
    """특정 문구를 whisper 가 무엇으로 받아썼는지 많은 순으로.

    "비카야"가 실제로 "비카야/비까야/미카야" 중 무엇으로 적히는지 보면, 후보
    목록(DEFAULT_VARIANTS)을 추측이 아니라 실측으로 정할 수 있다.
    """
    counter = Counter(
        (t.heard.strip() or "(빈 인식)") for t in trials if t.phrase == phrase
    )
    return counter.most_common()


def overall_verdict(results: Sequence[PhraseResult]) -> str:
    """전체 판정. 하나라도 부적합이면 부적합이다."""
    if not results:
        return "측정 없음"
    verdicts = [r.verdict for r in results]
    if "부적합" in verdicts:
        return "부적합 — 이대로 도입하기 어렵다"
    if "개선 필요" in verdicts or "주의" in verdicts:
        return "조건부 — 문구나 판정 규칙을 손봐야 한다"
    return "양호 — 다음 단계 검토 가능"


def format_report(trials: Sequence[Trial]) -> str:
    """사람이 읽는 결과표. 팀 회의에 그대로 가져갈 수 있게 만든다."""
    results = summarize(trials)
    if not results:
        return "측정 결과가 없습니다."

    lines = ["", "=" * 62, "웨이크워드 측정 결과", "=" * 62, ""]
    lines.append(f"{'말한 문구':<14}{'구분':<8}{'깨어남':>8}{'비율':>9}{'음량미달':>9}  판정")
    lines.append("-" * 62)
    for r in results:
        kind = "목표" if r.should_wake else "함정"
        lines.append(
            f"{r.phrase:<14}{kind:<8}{r.woke:>4}/{r.total:<3}{r.rate:>8.0%}"
            f"{r.gated:>8}회  {r.verdict}"
        )
    lines.append("-" * 62)
    lines.append("목표 = 깨어나야 하는 말(인식률, 높을수록 좋음)")
    lines.append("함정 = 깨어나면 안 되는 말(오인율, 낮을수록 좋음)")
    lines.append("음량미달 = 소리가 작아 STT 까지 가지도 못한 횟수(운영 감시와 같은 게이트).")
    lines.append("           목표 문구에서 이 값이 크면 '말이 작아서 못 깨우는' 문제다.")

    lines.append("")
    lines.append("whisper 가 실제로 받아쓴 내용")
    lines.append("-" * 62)
    for r in results:
        lines.append(f"[{r.phrase}]")
        for text, count in heard_counts(trials, r.phrase):
            lines.append(f"    {count:>3}회  {text}")

    lines.append("")
    lines.append("=" * 62)
    lines.append(f"종합: {overall_verdict(results)}")
    lines.append("=" * 62)
    lines.append("")
    lines.append("※ 판정 기준(인식률 90%/70%, 오인율 5%)은 잠정값이며 팀이 확정한다.")
    return "\n".join(lines)

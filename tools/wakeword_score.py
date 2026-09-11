"""웨이크워드 측정의 채점 로직 (마이크·whisper 없이 돌아가는 순수 계산)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Sequence

from src.emergency_filter import _starts_at_token_boundary

DEFAULT_WAKE_WORD = "비카야"
DEFAULT_VARIANTS: tuple[str, ...] = ("비카야",)

GOOD_HIT_RATE = 0.90
POOR_HIT_RATE = 0.70
WARN_FALSE_RATE = 0.05


@dataclass(frozen=True)
class Trial:
    """시도 한 번의 기록."""

    phrase: str
    should_wake: bool
    heard: str
    matched: Optional[str]
    rms: float = 0.0
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
    """받아쓴 문장에 깨우는 말이 있으면 그 후보를, 없으면 None."""
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
    gated: int = 0
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
    """특정 문구를 whisper 가 무엇으로 받아썼는지 많은 순으로."""
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

"""마이크 현장 점검기 — 값·측정·합격 기준·판정(순수 함수, 마이크·모델 없음).

설계: docs/superpowers/specs/2026-10-12-mic-venue-check-design.md 6·7절. 숫자는 모두 초기값이다(사용자 10-12:
"수치만 보고서는 모른다, 계속 돌려봐야 안다"). 기준을 다듬을 때는 이 파일 위쪽 상수만 고친다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

# ---------------------------------------------------------------- 합격 기준 (초기값)
NOISE_PASS_RATIO = 2.0         # 수음 최소 ÷ 잡음 바닥. 10-06 규칙 "수음 최소 = 잡음 × 2" 와 같은 여유
NOISE_WARN_RATIO = 1.3         # 이 아래면 사람이 말하지 않아도 잡음이 말로 잡힐 수 있다
NOISE_SPREAD_WARN = 2.0        # 2단계 세 측정의 최대 ÷ 최소 — 10-06 의 들쭉날쭉을 드러낸다
WAKE_MARGIN = 0.10             # 가장 낮은 호출 점수가 호출 문턱 + 이것 아래면 아슬아슬
SHORT_MARGIN = 1.5             # 가장 작은 짧은 답 소리가 짧은 답 최소 × 이것 아래면 아슬아슬
STOP_PASS, STOP_WARN = 5, 4    # "멈춰" 5번 중 확정 수
ROBOT_LEVEL_WARN = 1.5         # 로봇 목소리가 지난 점검의 이 배를 넘으면 볼륨이 올라가 있다
SHORT_OVER_LISTEN = 1.5        # 저장 검사: 짧은 답 최소 ≥ 수음 최소 × 이것 (10-06 규칙)
NOISE_PERCENTILE = 0.9         # 잡음 바닥 = 조각 rms 의 상위 10 %

CALL_REPS, ANSWER_REPS, STOP_REPS = 3, 4, 5
RUN_REPS = 2                   # 3단계 끝의 '이어서 부르기' — 참고 줄(2026-10-13 사용자: 실제로는 쉬지 않고 말하기도 한다)
PLACE_WORD = "화장실"           # 3단계 문장의 목적지 — 호출 뒤 받은 말에 이것이 있어야 목적지가 간 것이다

# ---------------------------------------------------------------- 값 고치기 허용 범위 (설계 7절)
RANGES = {   # 필드: (최소, 최대, 눈금)
    "listen_min_rms": (0.005, 0.015, 0.0005),
    "short_answer_min_rms": (0.012, 0.025, 0.001),
    "wake_threshold": (0.50, 0.90, 0.01),
    "emergency_single": (0.40, 0.70, 0.01),
}
NAMES = {
    "listen_min_rms": "수음 최소", "short_answer_min_rms": "짧은 답 최소",
    "wake_threshold": "호출 문턱", "emergency_single": "“멈춰” 한 조각 문턱",
    "agc": "AGC 목표", "echo": "에코 지우기",
}
AGC_CHOICES = (0.005, 0.007, 0.010)                                          # 이 로봇에서 써 본 값만
ECHO_CHOICES = {"strong": (1.5, 1), "mid": (1.0, 1), "factory": (1.0, 0)}   # 이름: (GAMMA_ETAIL, NLATTENONOFF)
ECHO_LABELS = {"strong": "센 (1.5·켜짐)", "mid": "중간 (1.0·켜짐)", "factory": "공장값 (1.0·꺼짐)"}

PASS, WARN, FAIL, REDO, WAIT = "pass", "warn", "fail", "redo", "wait"
_RANK = {PASS: 0, WARN: 1, WAIT: 2, REDO: 3, FAIL: 4}
TITLES = {PASS: "시연 가능", WARN: "시연 가능 · 확인 필요", WAIT: "점검 중",
          REDO: "다시 녹음 필요", FAIL: "고친 뒤 다시"}


@dataclass(frozen=True)
class MicValues:
    """점검기가 다루는 값 일곱 개 — `.env` 와 1:1 (src/env_file.KEYS)."""
    listen_min_rms: float
    short_answer_min_rms: float
    wake_threshold: float
    emergency_single: float
    agc: float
    echo_etail: float
    echo_nlatten: int

    def chip(self) -> tuple:
        """칩에 쓰는 셋 — 이것이 바뀌면 녹음이 무효다(칩이 소리 자체를 바꾼다)."""
        return (round(float(self.agc), 6), round(float(self.echo_etail), 6), int(self.echo_nlatten))

    def thresholds(self) -> tuple:
        """6단계(실시간으로 잰 자기 반응)에 영향을 주는 문턱 둘."""
        return (round(float(self.wake_threshold), 6), round(float(self.emergency_single), 6))


def echo_name(etail: float, nlatten: int) -> Optional[str]:
    """에코 단계 이름. 사다리 밖 값이면 None."""
    for name, (e, n) in ECHO_CHOICES.items():
        if abs(e - float(etail)) < 1e-6 and int(n) == int(nlatten):
            return name
    return None


def check_values(v: MicValues) -> list:
    """저장을 막을 문제(빈 목록이면 저장 가능): 허용 범위 밖, 짧은 답 최소 < 수음 최소 × 1.5."""
    problems = []
    for key, (lo, hi, _step) in RANGES.items():
        x = float(getattr(v, key))
        if not lo - 1e-9 <= x <= hi + 1e-9:
            problems.append(f"{NAMES[key]} {x:g} 은(는) 허용 범위 {lo:g} ~ {hi:g} 밖입니다")
    if v.short_answer_min_rms < v.listen_min_rms * SHORT_OVER_LISTEN - 1e-9:
        problems.append(f"짧은 답 최소는 수음 최소의 {SHORT_OVER_LISTEN:g}배 이상이어야 합니다 "
                        f"(지금 {v.short_answer_min_rms / v.listen_min_rms:.2f}배)")
    return problems


def noise_floor(values: Sequence[float]) -> Optional[float]:
    """조각 rms 의 상위 10 % 값. 비어 있으면 None."""
    xs = sorted(float(x) for x in values)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(len(xs) * NOISE_PERCENTILE))]


# ---------------------------------------------------------------- 측정
@dataclass(frozen=True)
class CallTry:
    """'비카야 … 화장실로 가자' 한 번의 재채점 결과(쉬고 부르기·이어서 부르기 같은 모양)."""
    woke: bool
    captured: bool            # 청취 창이 이어진 말을 받았나(받아쓰기가 나왔나)
    peak: float               # 호출 점수 최고
    text: str = ""
    emergency: bool = False   # 이 말에 멈춰가 확정됐나(오발동)


@dataclass(frozen=True)
class AnswerTry:
    expected: str             # "yes" | "no"
    accepted: bool
    correct: bool             # 긍정·부정이 맞나
    clip_rms: Optional[float]
    text: str = ""
    emergency: bool = False


@dataclass(frozen=True)
class StopTry:
    confirmed: bool
    gate: bool                # 관문(한 조각 또는 두 조각)을 넘었나 — 넘었으면 확인 STT 를 탔다
    peak: float               # 멈춰 점수 최고
    text: str = ""            # 확인 STT 글자(관문을 넘었을 때)
    woke: bool = False        # 이 말에 호출이 확정됐나(오발동)


@dataclass(frozen=True)
class RobotRun:
    self_reactions: int
    level: float              # 재생 중 조각 rms 평균(에코 지우기 뒤 소리)
    played_ok: bool = True    # 스피커 재생이 끝까지 됐나 — 실패하면 level 이 조용한 값이라 합격으로 칠 수 없다


@dataclass
class Measurements:
    noise: list = field(default_factory=list)       # 2단계 세 측정
    calls: list = field(default_factory=list)       # CallTry — 쉬고 부르기
    runs: list = field(default_factory=list)        # CallTry — 이어서 부르기(참고)
    answers: list = field(default_factory=list)     # AnswerTry
    stops: list = field(default_factory=list)       # StopTry
    robot: Optional[RobotRun] = None
    chips: dict = field(default_factory=dict)       # 줄 키 → 녹음할 때 MicValues.chip()
    robot_thresholds: Optional[tuple] = None        # 6단계 때 MicValues.thresholds()


@dataclass(frozen=True)
class Row:
    key: str
    name: str
    grade: str
    detail: str
    todo: str = ""


@dataclass(frozen=True)
class Verdict:
    rows: tuple
    overall: str

    @property
    def title(self) -> str:
        return TITLES[self.overall]


# ---------------------------------------------------------------- 판정
def judge(m: Measurements, v: MicValues, previous_robot_level: Optional[float] = None) -> Verdict:
    """설계 6절 표 그대로. 가장 나쁜 줄이 종합이다(불합격 > 다시 녹음 > 점검 중 > 주의 > 합격)."""
    rows = [
        _noise_row(m, v),
        _call_row(m, v),
        _run_row(m, v),
        _answer_row(m, v),
        _stop_row(m, v),
        _robot_row(m, v, previous_robot_level),
        _cross_row(m, v),
    ]
    overall = max((r.grade for r in rows), key=_RANK.__getitem__)
    return Verdict(tuple(rows), overall)


_AGC_REDO = "AGC 가 바뀌어 다시 녹음해야 합니다"


def _agc_stale(m: Measurements, key: str, v: MicValues) -> bool:
    chip = m.chips.get(key)
    return chip is not None and chip[0] != v.chip()[0]


def _noise_row(m: Measurements, v: MicValues) -> Row:
    name = "잡음 바닥"
    if not m.noise:
        return Row("noise", name, WAIT, "2단계 전")
    if _agc_stale(m, "noise", v):
        return Row("noise", name, REDO, _AGC_REDO, "2단계를 다시 하세요")
    floor, low = max(m.noise), min(m.noise)
    ratio = v.listen_min_rms / floor if floor > 0 else math.inf
    spread = floor / low if low > 0 else math.inf
    detail = f"{floor:.4f} · 수음 최소까지 {ratio:.1f}배"
    if spread > NOISE_SPREAD_WARN:
        detail += f" · 세 번 차이 {spread:.1f}배"
    if ratio < NOISE_WARN_RATIO:
        return Row("noise", name, FAIL, detail,
                   "에어컨·환풍기 같은 소음원을 찾거나 로봇 자리를 옮기세요. 또는 수음 최소를 올리세요.")
    if ratio < NOISE_PASS_RATIO:
        return Row("noise", name, WARN, detail,
                   "잡음이 수음 최소에 가깝습니다. 더 시끄러워지면 말하지 않아도 말로 잡힐 수 있습니다.")
    if spread > NOISE_SPREAD_WARN:
        return Row("noise", name, WARN, detail, "잡음이 일정하지 않습니다. 조용할 때 2단계를 다시 해 보세요.")
    return Row("noise", name, PASS, detail)


def call_delivered(t: CallTry) -> bool:
    """호출이 걸리고 목적지까지 건너갔나 — 받은 말에 PLACE_WORD 가 있어야 한다(띄어쓰기·구두점 무시)."""
    heard = "".join(ch for ch in t.text if ch.isalnum())
    return t.woke and t.captured and PLACE_WORD in heard


def _call_miss_reason(missed: Sequence[CallTry]) -> str:
    if any(not t.woke for t in missed):
        return "호출 놓침"
    if any(not t.captured for t in missed):
        return "호출 뒤 말을 못 받음"
    return f"목적지를 못 받음 '{missed[0].text}'"


def _call_row(m: Measurements, v: MicValues) -> Row:
    name = "“비카야” 쉬고 “화장실로 가자” ×3"
    tries = m.calls
    if tries and _agc_stale(m, "call", v):
        return Row("call", name, REDO, _AGC_REDO, "3단계를 다시 하세요")
    if len(tries) < CALL_REPS:
        return Row("call", name, WAIT, f"{len(tries)}/{CALL_REPS}")
    ok = sum(call_delivered(t) for t in tries)
    low = min(t.peak for t in tries)
    detail = f"{ok}/{CALL_REPS} · 가장 낮은 점수 {low:.2f}"
    if ok < CALL_REPS:
        why = _call_miss_reason([t for t in tries if not call_delivered(t)])
        return Row("call", name, FAIL, f"{detail} · {why}",
                   "손잡이 쪽을 향해 또렷하게 다시 해 보세요. 그래도 놓치면 호출 문턱을 보세요.")
    if low < v.wake_threshold + WAKE_MARGIN - 1e-9:
        return Row("call", name, WARN, f"{detail} (문턱 + {WAKE_MARGIN:.1f} 아래)", "호출이 아슬아슬하게 걸립니다.")
    return Row("call", name, PASS, detail)


def _run_row(m: Measurements, v: MicValues) -> Row:
    """이어서 부르기 — 지금 귀는 호출을 확인한 뒤에 창을 열어 이어 말한 목적지 머리를 놓친다(10-13 분석).
    값을 고쳐서 나아지는 문제가 아니라서 '주의'까지만 낸다: 시연 때 말하는 법을 알려 주는 줄이다."""
    name = "“비카야화장실로가자” 이어서 ×2 (참고)"
    tries = m.runs
    if tries and _agc_stale(m, "call", v):
        return Row("run", name, REDO, _AGC_REDO, "3단계를 다시 하세요")
    if len(tries) < RUN_REPS:
        return Row("run", name, WAIT, f"{len(tries)}/{RUN_REPS}")
    ok = sum(call_delivered(t) for t in tries)
    detail = f"{ok}/{RUN_REPS} · 가장 낮은 점수 {min(t.peak for t in tries):.2f}"
    if ok < RUN_REPS:
        why = _call_miss_reason([t for t in tries if not call_delivered(t)])
        return Row("run", name, WARN, f"{detail} · {why}",
                   "쉬지 않고 말하면 지금 귀는 목적지를 놓칩니다. 시연 때는 “비카야” 뒤 “네?”를 듣고 말하세요.")
    return Row("run", name, PASS, detail)


def _answer_row(m: Measurements, v: MicValues) -> Row:
    name = "“그래”·“아니” ×4"
    tries = m.answers
    if tries and _agc_stale(m, "answer", v):
        return Row("answer", name, REDO, _AGC_REDO, "4단계를 다시 하세요")
    if len(tries) < ANSWER_REPS:
        return Row("answer", name, WAIT, f"{len(tries)}/{ANSWER_REPS}")
    ok = sum(t.accepted and t.correct for t in tries)
    sizes = [t.clip_rms for t in tries if t.clip_rms is not None]
    low = min(sizes) if sizes else 0.0
    detail = f"{ok}/{ANSWER_REPS} · 가장 작은 소리 {low:.3f} (짧은 답 최소의 {low / v.short_answer_min_rms:.1f}배)"
    if ok < ANSWER_REPS:
        wrong = [t for t in tries if t.accepted and not t.correct]
        if wrong:
            return Row("answer", name, FAIL, f"{detail} · 받아쓰기 '{wrong[0].text}'",
                       "긍정·부정을 다르게 알아들었습니다. 받아쓴 글자를 보고 다시 해 보세요.")
        return Row("answer", name, FAIL, detail, "소리가 작았으면 짧은 답 최소를 낮추세요.")
    if low < v.short_answer_min_rms * SHORT_MARGIN - 1e-9:
        return Row("answer", name, WARN, detail, "짧은 답이 아슬아슬하게 들어갑니다.")
    return Row("answer", name, PASS, detail)


def _stop_row(m: Measurements, v: MicValues) -> Row:
    name = "“멈춰” ×5"
    tries = m.stops
    if tries and _agc_stale(m, "stop", v):
        return Row("stop", name, REDO, _AGC_REDO, "5단계를 다시 하세요")
    if len(tries) < STOP_REPS:
        return Row("stop", name, WAIT, f"{len(tries)}/{STOP_REPS}")
    ok = sum(t.confirmed for t in tries)
    misses = [t for t in tries if not t.confirmed]
    parts = [f"확인 기각 '{t.text}'" if t.gate else f"점수 {t.peak:.2f}" for t in misses]
    detail = f"{ok}/{STOP_REPS}" + (" · 놓침: " + ", ".join(parts) if parts else "")
    if ok >= STOP_PASS:
        return Row("stop", name, PASS, detail)
    todo = []
    if any(not t.gate for t in misses):
        todo.append("점수가 낮았습니다. 잡음을 확인하세요. 멈춰 문턱을 낮출 수 있지만 오발동이 늘어납니다.")
    if any(t.gate for t in misses):
        todo.append("받아쓴 글자가 '멈춰'가 아니었습니다. 말투를 확인하세요.")
    return Row("stop", name, WARN if ok >= STOP_WARN else FAIL, detail, " ".join(todo))


def _robot_row(m: Measurements, v: MicValues, previous: Optional[float]) -> Row:
    name = "로봇 혼자 말하기"
    r = m.robot
    if r is None:
        return Row("robot", name, WAIT, "6단계 전")
    chip = m.chips.get("robot")
    if (chip is not None and chip != v.chip()) or (
            m.robot_thresholds is not None and tuple(m.robot_thresholds) != v.thresholds()):
        return Row("robot", name, REDO, "AGC·에코·문턱이 바뀌어 다시 재야 합니다", "6단계를 다시 하세요")
    if not r.played_ok:
        return Row("robot", name, FAIL, "재생 실패 — 로봇 목소리를 틀지 못했습니다",
                   "스피커 연결·볼륨을 확인하고 6단계를 다시 하세요.")
    detail = f"자기 반응 {r.self_reactions}회 · 로봇 목소리 {r.level:.3f}"
    if previous:
        detail += f" (지난 점검 {previous:.3f})"
    if r.self_reactions > 0:
        return Row("robot", name, FAIL, detail, "스피커 볼륨을 낮추고 6단계만 다시 하세요.")
    if previous and r.level > previous * ROBOT_LEVEL_WARN:
        return Row("robot", name, WARN, detail, "로봇 목소리가 지난 점검보다 큽니다. 스피커 볼륨을 확인하세요.")
    return Row("robot", name, PASS, detail)


def _cross_row(m: Measurements, v: MicValues) -> Row:
    name = "오발동"
    if any(_agc_stale(m, k, v) for k in ("call", "answer", "stop")) and (m.calls or m.answers or m.stops):
        return Row("cross", name, REDO, _AGC_REDO, "3~5단계를 다시 하세요")
    n = (sum(t.emergency for t in m.calls) + sum(t.emergency for t in m.runs)
         + sum(t.emergency for t in m.answers) + sum(t.woke for t in m.stops))
    if n:
        return Row("cross", name, FAIL, f"{n}회",
                   "다른 말에 멈춰나 비카야가 잘못 걸렸습니다. 문턱을 내렸다면 되돌리세요.")
    done = (len(m.calls) >= CALL_REPS and len(m.runs) >= RUN_REPS and len(m.answers) >= ANSWER_REPS
            and len(m.stops) >= STOP_REPS)
    return Row("cross", name, PASS if done else WAIT, "0회")

"""녹음을 ⑫ 와 같은 귀(WakewordMonitor)에 다시 흘려 채점한다 — 마이크 현장 점검기(2026-10-12).

10-06 src/venue_replay.py(feat/midm-fallback) 를 지금 귀에 맞게 옮겼다: 대화 STT·긴급 확인 STT 가 나뉘었고
(10-12), "멈춰" 한 조각 규칙이 생겼다. 녹음 하나마다 새 귀를 만든다 — 같은 녹음·같은 값이면 늘 같은 결과다.
칩 사람 말 표시(vad)는 노드처럼 청취 중일 때만 넣는다(WakewordMonitor.run 과 같은 규칙).
설계: docs/superpowers/specs/2026-10-12-mic-venue-check-design.md 5절
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .handle_mode import AFFIRMATIVES, NEGATIVES, SOFT_AFFIRMATIVES, normalize_short_reply
from .mic_check_rules import AnswerTry, CallTry, MicValues, StopTry
from .wakeword_monitor import FRAME, SAMPLE_RATE, WakewordMonitor

DT = FRAME / SAMPLE_RATE      # 0.08 s
FLUSH_FRAMES = 25             # 호출 모델의 직전 2초 기억을 비우는 조용한 조각
PAD_FRAMES = 50               # 녹음 뒤 조용한 조각(4 s) — 청취 창과 멈춰 확인이 닫힐 시간


@dataclass(frozen=True)
class Recording:
    frames: tuple             # int16 1280 조각들
    vads: tuple               # 조각마다 칩 사람 말 표시(Optional[bool])

    @property
    def seconds(self) -> float:
        return len(self.frames) * DT


@dataclass(frozen=True)
class Models:
    """한 번 올린 모델을 나눠 쓴다(재채점마다 다시 올리지 않는다)."""
    predict: Callable[[np.ndarray], dict]
    transcribe: Callable[[np.ndarray], str]                            # 긴급 확인(small)
    transcribe_listen: Optional[Callable[[np.ndarray], str]] = None    # 대화(turbo, 필터판)
    transcribe_confirm: Optional[Callable[[np.ndarray], str]] = None   # 질문 창(귀띔판)
    listen_hint: Optional[str] = None

    @classmethod
    def from_loaded(cls, monitor: WakewordMonitor) -> "Models":
        """_load_real() 을 마친 귀의 모델을 꺼낸다."""
        return cls(monitor._predict, monitor._transcribe, monitor._transcribe_listen,
                   monitor._transcribe_confirm, monitor._listen_hint)


class _Run:
    """재채점 한 번의 귀와 그 결과."""

    def __init__(self, models: Models, values: MicValues):
        self.texts: list = []
        self.emergencies: list = []
        self.counting = False
        self.peak_a = 0.0
        self.peak_b = 0.0
        self._predict = models.predict
        self.monitor = WakewordMonitor(
            on_emergency=self.emergencies.append,
            on_user_text=self.texts.append,
            predict=self._tracked,
            transcribe=models.transcribe,
            transcribe_listen=models.transcribe_listen,
            transcribe_confirm=models.transcribe_confirm,
            listen_hint=models.listen_hint,
            voice_barge_in=False,
            gate_a=values.wake_threshold,
            single_b=values.emergency_single,
            listen_min_rms=values.listen_min_rms,
            short_answer_min_rms=values.short_answer_min_rms,
        )

    def _tracked(self, frame: np.ndarray) -> dict:
        scores = self._predict(frame)
        if self.counting:
            self.peak_a = max(self.peak_a, float(scores["a"]))
            self.peak_b = max(self.peak_b, float(scores["b"]))
        return scores

    def feed(self, frames: Sequence, vads: Sequence, t: float) -> tuple:
        out = []
        for frame, vad in zip(frames, vads):
            out.append(self.monitor.process_frame(
                frame, now=t, vad=(vad if self.monitor.state == "listen" else None)))
            t += DT
        return t, out


def _cycle(quiet: Optional[Recording], n: int) -> list:
    src = list(quiet.frames) if quiet is not None and quiet.frames else [np.zeros(FRAME, dtype=np.int16)]
    return [src[i % len(src)] for i in range(n)]


def _play(models: Models, values: MicValues, rec: Recording, quiet: Optional[Recording],
          followup: bool) -> tuple:
    """조용한 조각 → (질문 창 열기) → 녹음 → 조용한 조각. (_Run, 녹음 이후 결과들) 을 돌려준다."""
    run = _Run(models, values)
    t, _ = run.feed(_cycle(quiet, FLUSH_FRAMES), [False] * FLUSH_FRAMES, 0.0)
    if followup:
        run.monitor.arm_followup(now=t)          # 노드처럼: 질문을 말했다는 예약
        run.monitor.set_speaking(False, now=t)   # → 질문이 끝난 순간 질문 창이 열린다
    run.counting = True
    t, out = run.feed(list(rec.frames), list(rec.vads), t)
    run.counting = False
    _, tail = run.feed(_cycle(quiet, PAD_FRAMES), [False] * PAD_FRAMES, t)
    return run, out + tail


def is_affirmative(text: str) -> bool:
    return normalize_short_reply(text) in (AFFIRMATIVES | SOFT_AFFIRMATIVES)


def is_negative(text: str) -> bool:
    return normalize_short_reply(text) in NEGATIVES


def replay_call(models: Models, rec: Recording, quiet: Optional[Recording], values: MicValues) -> CallTry:
    """'비카야, 화장실로 가자' — 대기 상태에서 흘려 호출과 자유 창 받아쓰기를 본다."""
    run, out = _play(models, values, rec, quiet, followup=False)
    return CallTry(woke="wake" in out, captured=bool(run.texts), peak=run.peak_a,
                   text=run.texts[0] if run.texts else "", emergency=bool(run.emergencies))


def replay_answer(models: Models, rec: Recording, quiet: Optional[Recording], values: MicValues,
                  expected: str) -> AnswerTry:
    """'네'·'아니요' — 질문 창을 열고 흘려 받아들여지는지와 긍정·부정을 본다."""
    run, _ = _play(models, values, rec, quiet, followup=True)
    text = run.texts[0] if run.texts else ""
    stats = run.monitor.last_listen_stats
    correct = bool(text) and (is_affirmative(text) if expected == "yes" else is_negative(text))
    return AnswerTry(expected=expected, accepted=bool(run.texts), correct=correct,
                     clip_rms=float(stats["rms"]) if stats else None, text=text,
                     emergency=bool(run.emergencies))


def replay_stop(models: Models, rec: Recording, quiet: Optional[Recording], values: MicValues) -> StopTry:
    """'멈춰' — 관문(한 조각·두 조각) → 확인 STT → 정확 매칭까지 ⑫ 와 같게."""
    run, out = _play(models, values, rec, quiet, followup=False)
    check = run.monitor.last_emergency_check
    return StopTry(confirmed=bool(run.emergencies), gate=check is not None, peak=run.peak_b,
                   text=(check or {}).get("text", "") or "", woke="wake" in out)

"""세션 주인 목소리 따라가기 — 등록·판정·세션 경계(설계 docs/superpowers/specs/2026-11-11-speaker-follow-design.md).

세션 주인 = 처음 목적지를 말한 사람 1명(D1). 등록은 세션이 시작되게 한 발화 하나(호출 창이면 그 창의 "비카야"
소리 포함, D14). 안내 중에는 주인 목소리 또는 손잡이 방향에서 온 긴 말만 받고(4.2), 짧은 말은 거르지 않는다(D13).
대기 중 위험 명령은 목소리만으로 판정해 미션에 넘긴다(B1) — 여기서 버리지 않는다.
임베딩 함수는 밖에서 받는다(src/spk_embed.load_embedder) — 이 모듈은 모델 없이 시험이 돈다.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Optional, Tuple

import numpy as np

from .speech_trim import trim_speech
from .spk_embed import cosine

MODES = ("off", "log", "on")
SESSION_STATES = ("confirming", "navigating")          # 세션 밖에서 이 상태가 되면 시작
END_STATES = ("idle", "returning", "estopped")          # 세션 안에서 이 상태가 되면 끝(D10)
WAIT_STATES = ("waiting", "waiting_release")
ENROLL_MAX_AGE_SEC = 10.0      # 세션 시작 직전 이만큼 안의 발화로 등록
WAKE_MAX_AGE_SEC = 30.0        # 호출 창 판정에 붙일 "비카야" 소리의 유효 시간
JOIN_MAX_AGE_SEC = 20.0        # 되묻기용 이어 붙인 점수 — 직전 발화의 유효 시간


@dataclass
class SpkConfig:
    mode: str = "off"
    model_path: str = ""
    th_wake: float = 0.55
    th: float = 0.50
    short_sec: float = 1.0
    doa_center: Optional[float] = None
    doa_width: float = 60.0


def _float(env: Mapping[str, str], key: str, default: Optional[float]) -> Optional[float]:
    raw = (env.get(key) or "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def spk_config_from_env(env: Mapping[str, str]) -> SpkConfig:
    mode = (env.get("VICA_SPK_FOLLOW") or "off").strip().lower()
    return SpkConfig(
        mode=mode if mode in MODES else "off",
        model_path=(env.get("VICA_SPK_MODEL") or "").strip(),
        th_wake=_float(env, "VICA_SPK_TH_WAKE", 0.55),
        th=_float(env, "VICA_SPK_TH", 0.50),
        short_sec=_float(env, "VICA_SPK_SHORT_SEC", 1.0),
        doa_center=_float(env, "VICA_USER_DOA_CENTER", None),
        doa_width=_float(env, "VICA_USER_DOA_WIDTH", 60.0),
    )


@dataclass
class Verdict:
    verdict: str          # pass · fail · short · no_target
    score: float          # 없으면 −1
    joined: float         # 직전 발화와 이어 붙인 점수, 없으면 −1
    unit: str             # wake+utt · utt
    speech_sec: float
    doa: Optional[float]
    drop: bool
    reason: str
    text: str = ""

    def to_json(self) -> str:
        """의도 노드로 보내는 판정(버릴지는 호출 노드 안의 일이라 뺀다)."""
        return json.dumps({k: v for k, v in asdict(self).items() if k != "drop"}, ensure_ascii=False)


def _in_sector(doa: Optional[float], cfg: SpkConfig) -> bool:
    if doa is None or cfg.doa_center is None:
        return False
    return abs((doa - cfg.doa_center + 180.0) % 360.0 - 180.0) <= cfg.doa_width


def judge(score: Optional[float], has_wake: bool, speech_sec: float, doa: Optional[float], cfg: SpkConfig,
          *, has_target: bool, guiding: bool, waiting: bool) -> Tuple[str, bool, str]:
    """(판정, 버릴지, 사유). 규칙 순서는 설계 4.2·B1·B3·D13."""
    if not has_target:
        return "no_target", False, "등록 없음"
    th = cfg.th_wake if has_wake else cfg.th
    if waiting:
        if score is not None and score >= th:
            return "pass", False, "목소리"
        return "fail", False, "목소리 미달(대기)"
    if speech_sec < cfg.short_sec:
        return "short", False, "짧은 말"
    if score is not None and score >= th:
        verdict, reason = "pass", "목소리"
    elif _in_sector(doa, cfg):
        verdict, reason = "pass", "손잡이 방향"
    elif score is None:
        return "short", False, "점수 없음"
    else:
        verdict, reason = "fail", "목소리 미달·방향 밖"
    return verdict, cfg.mode == "on" and guiding and verdict == "fail", reason


class SessionTracker:
    """미션 dialog_state 로 세션 시작·끝을 가른다. 같은 상태가 1 Hz 로 반복돼도 사건은 한 번."""

    def __init__(self):
        self.active = False
        self.guiding = False
        self.waiting = False

    def on_dialog_state(self, state: str) -> Optional[str]:
        event = None
        if not self.active and state in SESSION_STATES:
            self.active, event = True, "start"
        elif self.active and state in END_STATES:
            self.active, self.guiding, event = False, False, "end"
        if self.active and state == "navigating":
            self.guiding = True
        self.waiting = self.active and state in WAIT_STATES
        return event


class SpeakerFollow:
    def __init__(self, cfg: SpkConfig, embed: Callable[[np.ndarray], Optional[np.ndarray]],
                 log: Callable[[str], None] = print):
        self.cfg = cfg
        self._embed = embed
        self._log = log
        self.session = SessionTracker()
        self._target: Optional[np.ndarray] = None
        self._wake: Optional[Tuple[float, np.ndarray]] = None          # (시각, 자른 소리)
        self._last: Optional[Tuple[float, Optional[np.ndarray], np.ndarray]] = None   # (시각, 벡터, 발화 소리)
        # 감시 스레드(check·note_wake_audio)와 ROS 콜백 스레드(on_dialog_state)가 같이 부른다.
        self._lock = threading.Lock()

    @property
    def has_target(self) -> bool:
        return self._target is not None

    def note_wake_audio(self, audio_i16: np.ndarray, now: float) -> None:
        if self.cfg.mode == "off":
            return
        with self._lock:
            self._wake = (now, trim_speech(np.asarray(audio_i16, dtype=np.float32) / 32768.0))

    def on_dialog_state(self, state: str, now: float) -> None:
        with self._lock:
            self._on_dialog_state(state, now)

    def _on_dialog_state(self, state: str, now: float) -> None:
        event = self.session.on_dialog_state(state)
        if event == "start":
            last = self._last
            if last is not None and now - last[0] <= ENROLL_MAX_AGE_SEC and last[1] is not None:
                self._target = last[1]
                self._log(f"[화자] 등록 — 세션 시작({state}), 소리 {len(last[2]) / 16000:.1f} s")
            else:
                self._target = None
                self._log(f"[화자] 등록 없음 — 세션 시작({state}) 직전 {ENROLL_MAX_AGE_SEC:.0f} s 안 발화 없음")
        elif event == "end":
            self._target, self._wake, self._last = None, None, None
            self._log(f"[화자] 세션 끝({state}) — 등록 지움")

    def _vec(self, audio: np.ndarray) -> Optional[np.ndarray]:
        return self._embed(audio) if len(audio) else None

    def check(self, audio_i16: np.ndarray, is_wake_window: bool, doa: Optional[float], speech_sec: float,
              now: float, text: str = "") -> Verdict:
        if self.cfg.mode == "off":
            return Verdict("no_target", -1.0, -1.0, "utt", speech_sec, doa, False, "꺼짐", text)
        with self._lock:
            return self._check(audio_i16, is_wake_window, doa, speech_sec, now, text)

    def _check(self, audio_i16: np.ndarray, is_wake_window: bool, doa: Optional[float], speech_sec: float,
               now: float, text: str) -> Verdict:
        utt = trim_speech(np.asarray(audio_i16, dtype=np.float32) / 32768.0)
        unit, audio = "utt", utt
        if is_wake_window and self._wake is not None and now - self._wake[0] <= WAKE_MAX_AGE_SEC:
            unit, audio = "wake+utt", np.concatenate([self._wake[1], utt])
            self._wake = None
        vec = self._vec(audio)
        score = cosine(self._target, vec) if self._target is not None and vec is not None else None
        joined = -1.0
        last = self._last
        # 이어 붙인 점수는 대기 중 되묻기에만 쓴다 — 안내 중에는 임베딩 1 회로 감시 스레드 지연을 줄인다(검토 I4).
        if (self.session.waiting and self._target is not None and last is not None
                and now - last[0] <= JOIN_MAX_AGE_SEC and len(utt)):
            jv = self._vec(np.concatenate([last[2], utt]))
            if jv is not None:
                joined = cosine(self._target, jv)
        verdict, drop, reason = judge(score, unit == "wake+utt", speech_sec, doa, self.cfg,
                                      has_target=self.has_target, guiding=self.session.guiding,
                                      waiting=self.session.waiting)
        self._last = (now, vec, utt)
        return Verdict(verdict, -1.0 if score is None else round(score, 4), round(joined, 4), unit,
                       round(speech_sec, 2), doa, drop, reason, text)

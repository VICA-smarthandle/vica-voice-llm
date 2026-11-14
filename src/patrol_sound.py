"""경비(순찰) 큰 소리 판정 — reSpeaker 원음(ch1~4) 크기가 기준을 0.5 s 넘으면 /vica/patrol/event 에 sound 를 낸다.

계약은 ~/vica_data/patrol_plan/SPEC.md 2·3.1·4.C(2026-11-14), 수치 근거는 11-11 실측
(~/vica_data/patrol_tests/2026-11-11_sound/README.txt — 주행 617 s 동안 67 dB·0.5 s 는 0건).

  크기(dB) = 20*log10(rms/32768) + VICA_PATROL_DB_OFFSET(기본 107.5 — 폰 소음계 60 dB 에 맞춘 값)
  기준 65 dB(주행·지점 정지 같은 값) 이상이 0.5 s 이어지면 sound 1번, 그 뒤 3 s 쉼 · 그 밖 상태는 판정 안 함
  로봇이 말하는 동안은 판정하지 않고 이어진 시간도 0

2026-11-14 실기 뒤 사용자 결정: 종전 "주행 67 dB · 정지 1.0 s 뒤 53 dB · 0.5 s 지속" → "65 dB · 0.5 s 지속".
주행·정지 기준을 따로 두는 틀(VICA_PATROL_DB_MOVING·VICA_PATROL_DB_STOPPED, 정지는 1.0 s 뒤부터)은 그대로 두고
기본값만 같게 했다. 지속을 0.3 s 로 줄이는 안은 버렸다 — 11-11 실측 주행 617 s 를 65 dB · 0.3 s 로 다시 돌리면
sound 가 5건(81.0 · 67.2 · 68.6 · 69.2 · 68.4 dB — 로봇 주행 소음), 0.5 s 면 같은 자료에서 0건이다.

세 군데에서 불린다. 판정 상태(patrolling·patrol_dwell)가 아니면 armed 가 False 라 ①②는 불리지 않는다.
  ① 오디오 콜백(wakeword_monitor.run 의 cb) → push: 원음 채널 제곱 평균 한 번
  ② 감시 루프(같은 run) → poll: 판정, sound 발행. 미션 상태가 3 s 끊기면 스스로 끈다
  ③ ROS 콜백(ros_wakeword_node) → set_state·set_speaking·set_watch: 값이 바뀔 때만 일한다

관리자가 앱에서 '큰 소리'를 꺼 두면(/vica/patrol/watch, 2026-11-14) 순찰 중에도 armed 가 False 다 — 재지도 않는다.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Mapping, NamedTuple, Optional

import numpy as np

STATE_MOVING = "patrolling"
STATE_DWELL = "patrol_dwell"
JUDGED_STATES = (STATE_MOVING, STATE_DWELL)   # 순찰 상태 다섯 중 판정하는 둘(경보·일시정지는 듣지 않는다)

BLOCK_SEC = 0.08                    # 마이크 블록 길이 — wakeword_monitor 의 FRAME 1280 / 16 kHz
RAW_CHANNELS = slice(1, 5)          # reSpeaker 6채널 중 원음(0 = 칩 처리음, 5 = 재생 참조)
FULL_SCALE = 32768.0
MIN_RMS = 1e-3                      # 무음의 log(0) 막기 — 시험 도구 sound_level.py 와 같은 바닥
GAP_RESET_SEC = 0.5                 # 블록이 이만큼 끊기면 이어진 것으로 치지 않는다
BACKLOG_MAX = 250                 # 감시 루프가 막혀도 쌓아 둘 블록 수(20 s)
STATE_STALE_SEC = 3.0               # 미션 상태(1 Hz)가 이만큼 끊기면 순찰이 아닌 것으로 본다


@dataclass(frozen=True)
class PatrolSoundConfig:
    db_offset: float = 107.5        # VICA_PATROL_DB_OFFSET
    db_moving: float = 65.0         # VICA_PATROL_DB_MOVING — patrolling (11-14 사용자 결정 65, 종전 67)
    db_stopped: float = 65.0        # VICA_PATROL_DB_STOPPED — patrol_dwell 1.0 s 뒤 (11-14 사용자 결정 65, 종전 53)
    settle_sec: float = 1.0         # patrol_dwell 이 이만큼 이어진 뒤 정지 기준(지금은 두 기준이 같다)
    hold_sec: float = 0.5           # 기준 이상이 이만큼 이어지면 사건 (0.3 s 는 주행 소음에 5건 — 11-14 사용자 결정 0.5 유지)
    rest_sec: float = 3.0           # 사건 뒤 쉬는 시간


def config_from_env(env: Optional[Mapping[str, str]] = None) -> PatrolSoundConfig:
    """.env 의 세 값. 줄이 없거나 숫자가 아니면 코드 기본값."""
    env = os.environ if env is None else env
    base = PatrolSoundConfig()

    def one(name: str, default: float) -> float:
        try:
            value = float(str(env.get(name, "")).strip())
        except ValueError:
            return default
        return value if math.isfinite(value) else default

    return PatrolSoundConfig(
        db_offset=one("VICA_PATROL_DB_OFFSET", base.db_offset),
        db_moving=one("VICA_PATROL_DB_MOVING", base.db_moving),
        db_stopped=one("VICA_PATROL_DB_STOPPED", base.db_stopped),
    )


def sound_watched(text: str) -> Optional[bool]:
    """/vica/patrol/watch 의 JSON({"off": ["sound"]}) → 큰 소리를 보는가. 못 읽으면 None(지금 값을 그대로 둔다)."""
    try:
        off = json.loads(text).get("off")
    except (AttributeError, TypeError, ValueError):
        return None
    return "sound" not in off if isinstance(off, list) else None


def level_db(mean_square: float, offset: float) -> float:
    """원음 제곱 평균(int16 눈금) → 맞춘 dB."""
    rms = max(math.sqrt(max(float(mean_square), 0.0)), MIN_RMS)
    return 20.0 * math.log10(rms / FULL_SCALE) + offset


class SoundEvent(NamedTuple):
    db: float           # 이어진 구간의 최대 크기
    threshold: float    # 판정 순간의 기준


class PatrolSoundDetector:
    """순수 판정 — 블록 하나의 크기와 시각을 받아 사건이면 SoundEvent 를 돌려준다. 시계·ROS·마이크 없음.

    set_state·set_speaking 은 ROS 스레드, feed 는 감시 스레드가 부른다. 상태는 (값, 시작 시각) 한 쌍을
    통째로 바꿔 끼워 잠금 없이 읽는다.
    """

    def __init__(self, cfg: Optional[PatrolSoundConfig] = None, block_sec: float = BLOCK_SEC):
        self.cfg = cfg or PatrolSoundConfig()
        # 0.5 s 를 채우는 블록 수 — 80 ms 블록이면 7개(0.56 s), 실측 자료의 0.1 s 블록이면 5개.
        self.hold_blocks = max(1, math.ceil(self.cfg.hold_sec / block_sec - 1e-9))
        self._ctx: tuple[str, float] = ("", 0.0)
        self._speaking = False
        self._run = 0
        self._peak = 0.0
        self._last_at: Optional[float] = None
        self._rest_until = -math.inf

    @property
    def state(self) -> str:
        return self._ctx[0]

    def set_state(self, state: str, now: float) -> None:
        """미션 dialog_state. 값이 바뀐 시각을 그 상태의 시작으로 둔다(같은 값 반복은 무시)."""
        if state != self._ctx[0]:
            self._ctx = (state, now)

    def set_speaking(self, speaking: bool) -> None:
        self._speaking = bool(speaking)

    def threshold(self, now: float) -> Optional[float]:
        """지금 적용할 기준(dB). 판정하지 않는 상태면 None."""
        state, since = self._ctx
        if state == STATE_MOVING:
            return self.cfg.db_moving
        if state == STATE_DWELL:
            return self.cfg.db_stopped if now - since >= self.cfg.settle_sec else self.cfg.db_moving
        return None

    def feed(self, mean_square: float, now: float) -> Optional[SoundEvent]:
        return self.feed_db(level_db(mean_square, self.cfg.db_offset), now)

    def feed_db(self, db: float, now: float) -> Optional[SoundEvent]:
        """블록 하나. 기준 이상이 hold_blocks 개 이어지면 사건 1번, 그 뒤 rest_sec 는 쉰다."""
        if self._last_at is not None and now - self._last_at > GAP_RESET_SEC:
            self._run = 0
        self._last_at = now
        thr = None if self._speaking else self.threshold(now)
        if thr is None or now < self._rest_until or db < thr:
            self._run = 0
            return None
        self._peak = db if self._run == 0 else max(self._peak, db)
        self._run += 1
        if self._run < self.hold_blocks:
            return None
        self._run = 0
        self._rest_until = now + self.cfg.rest_sec
        return SoundEvent(db=self._peak, threshold=thr)


class PatrolSoundTap:
    """마이크 콜백·감시 루프·ROS 콜백을 판정기에 잇는 연결. emit 은 JSON 문자열 하나를 받는다."""

    def __init__(self, emit: Callable[[str], None], cfg: Optional[PatrolSoundConfig] = None,
                 log: Optional[Callable[[str], None]] = None, block_sec: float = BLOCK_SEC,
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time):
        self.cfg = cfg or config_from_env()
        self.armed = False           # 판정 상태(patrolling·patrol_dwell)이고 큰 소리를 보는가 — False 면 콜백·감시 루프가 부르지 않는다
        self._watch = True           # 관리자가 '큰 소리'를 켜 뒀나. 미션 알림을 못 받았으면 켬이다
        self._det = PatrolSoundDetector(self.cfg, block_sec)
        self._emit = emit
        self._log = log or (lambda line: None)
        self._clock = clock
        self._wall = wall
        self._values: deque[tuple[float, float]] = deque(maxlen=BACKLOG_MAX)   # (받은 시각, 제곱 평균)
        self._heard_at = 0.0         # 미션 상태를 마지막으로 들은 시각
        self._warned = False

    # ---- ③ ROS 콜백 ----------------------------------------------------------------
    def set_state(self, state: str) -> None:
        """/vica/robot_state 의 dialog_state(1 Hz). 값이 바뀔 때만 일한다."""
        state = state or ""
        now = self._clock()
        self._heard_at = now
        if state == self._det.state:
            return
        self._det.set_state(state, now)
        self._arm()

    def set_watch(self, watched: bool) -> None:
        """/vica/patrol/watch — 관리자가 '큰 소리'를 꺼 뒀으면 순찰 중에도 재지 않는다. 값이 바뀔 때만 일한다."""
        watched = bool(watched)
        if watched == self._watch:
            return
        self._watch = watched
        self._arm()

    def _arm(self) -> None:
        armed = self._watch and self._det.state in JUDGED_STATES
        if armed and not self.armed:
            self._values.clear()
            self._log(f"[경비] 소리 판정 켬 — 주행 {self.cfg.db_moving:g} dB·정지 {self.cfg.db_stopped:g} dB"
                      f"({self.cfg.settle_sec:g} s 뒤)·{self.cfg.hold_sec:g} s 지속·더할 값 {self.cfg.db_offset:g}")
        elif self.armed and not armed:
            self._log("[경비] 소리 판정 끔")
        self.armed = armed

    def set_speaking(self, speaking: bool) -> None:
        """/vica/tts_state — 로봇이 말하는 동안은 판정하지 않는다."""
        self._det.set_speaking(speaking)

    # ---- ① 오디오 콜백 --------------------------------------------------------------
    def push(self, block: np.ndarray) -> None:
        """6채널 int16 블록 하나 — 원음 채널 제곱 평균 한 번. 예외를 밖으로 내지 않는다."""
        try:
            raw = block[:, RAW_CHANNELS].astype(np.float64)
            self._values.append((self._clock(), float(np.mean(raw * raw))))
        except Exception:  # noqa: BLE001 — 콜백에서 예외가 나가면 마이크 스트림이 멈춘다
            pass

    # ---- ② 감시 루프 ----------------------------------------------------------------
    def poll(self) -> None:
        """쌓인 블록을 판정해 sound 를 낸다. 실패해도 감시 루프는 계속 간다."""
        try:
            while True:
                try:
                    at, mean_square = self._values.popleft()
                except IndexError:
                    break
                event = self._det.feed(mean_square, at)
                if event is not None:
                    self._emit(json.dumps({"src": "audio", "kind": "sound", "stamp": round(self._wall(), 3),
                                           "db": round(event.db, 1), "threshold": event.threshold}))
                    self._log(f"[경비] 큰 소리 {event.db:.1f} dB (기준 {event.threshold:g} dB, {self._det.state})")
            now = self._clock()
            if now - self._heard_at > STATE_STALE_SEC:
                self._log("[경비] 미션 상태가 끊겨 소리 판정을 끕니다")
                self.set_state("")   # 미션이 순찰 상태에서 멈춰도 계산·발행을 이어 가지 않는다
                self._heard_at = now
        except Exception as exc:  # noqa: BLE001 — 발행 실패로 호출어·긴급어 감시를 멈추지 않는다
            if not self._warned:
                self._warned = True
                self._log(f"[경비] 소리 판정 오류(이후 생략): {exc!r}")

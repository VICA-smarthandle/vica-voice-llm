"""소프트웨어 에코 제거 줄임량 기록 — 로봇이 말한 덩어리마다 한 줄, 듣지 않으면 경고(설계 4.4)."""
from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

AEC_LOG_DIR = "~/vica_data/aec_log"
REF_ACTIVE_RMS = 50.0
HANG_SEC = 0.5
MIN_SEC = 1.0
WEAK_DB, WEAK_MIN_SEC, WEAK_RUN = -3.0, 3.0, 5
WEAK_WARNING = "에코 제거가 듣지 않음 — 최근 5번 연속 줄임 3 dB 미만(스피커 지연이 바뀌었을 수 있음)"
_EPS = 1e-9
_FULL_SCALE = 32768.0


def _ms(x: np.ndarray) -> float:
    v = x.astype(np.float64)
    return float(np.mean(v * v))


class EchoMeter:
    def __init__(self, mode: str, log_dir: Optional[Path], warn: Callable[[str], None],
                 block_sec: float = 0.08, wall: Callable[[], float] = time.time) -> None:
        self._mode = mode
        self._log_dir = Path(log_dir).expanduser() if log_dir is not None else None
        self._warn = warn
        self._block_sec = block_sec
        self._wall = wall
        self._blocks = 0
        self._quiet = 0
        self._sum_in = 0.0
        self._sum_out = 0.0
        self._weak_run = 0
        self._weak_warned = False
        self._write_warned = False

    def update(self, mic: np.ndarray, out: np.ndarray, ref: np.ndarray) -> Optional[dict]:
        if math.sqrt(_ms(ref)) > REF_ACTIVE_RMS:
            self._blocks += 1
            self._quiet = 0
            self._sum_in += _ms(mic)
            self._sum_out += _ms(out)
            return None
        if self._blocks == 0:
            return None
        self._quiet += 1
        if self._quiet * self._block_sec < HANG_SEC - _EPS:
            return None
        return self._finish()

    def _reset(self) -> None:
        self._blocks = 0
        self._quiet = 0
        self._sum_in = 0.0
        self._sum_out = 0.0

    def _finish(self) -> Optional[dict]:
        n, sum_in, sum_out = self._blocks, self._sum_in, self._sum_out
        self._reset()
        sec = n * self._block_sec
        if sec < MIN_SEC - _EPS:
            return None
        rms_in = math.sqrt(sum_in / n)
        rms_out = math.sqrt(sum_out / n)
        reduce_db = 20.0 * math.log10(max(rms_out, _EPS) / max(rms_in, _EPS))
        now = self._wall()
        row = {
            "time": datetime.fromtimestamp(now).isoformat(timespec="milliseconds"),
            "sec": round(sec, 2),
            "rms_in": round(rms_in / _FULL_SCALE, 6),
            "rms_out": round(rms_out / _FULL_SCALE, 6),
            "reduce_db": round(reduce_db, 2),
            "mode": self._mode,
        }
        self._write(row, now)
        self._check_weak(sec, reduce_db)
        return row

    def _check_weak(self, sec: float, reduce_db: float) -> None:
        if sec < WEAK_MIN_SEC - _EPS:
            return
        if reduce_db > WEAK_DB:
            self._weak_run += 1
            if self._weak_run >= WEAK_RUN and not self._weak_warned:
                self._weak_warned = True
                self._warn(WEAK_WARNING)
        else:
            self._weak_run = 0
            self._weak_warned = False

    def _write(self, row: dict, now: float) -> None:
        if self._log_dir is None:
            return
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            name = datetime.fromtimestamp(now).strftime("%Y%m%d") + ".jsonl"
            with open(self._log_dir / name, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as exc:
            if not self._write_warned:
                self._write_warned = True
                self._warn(f"에코 제거 기록 저장 실패(이후 경고 생략): {exc}")

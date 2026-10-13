"""`.env` 의 마이크 값 읽기·바꾸기·백업·되돌리기 — 마이크 현장 점검기(2026-10-12) 전용.

설계 7절. 바꾸기 전에 늘 백업하고, 바꾼 줄 끝에 언제·어디서·전 값을 남긴다. 관계없는 줄·주석·순서는 건드리지
않는다. 같은 키가 주석 아닌 줄로 두 번 이상 있으면 어느 줄이 쓰일지 헷갈리므로 쓰지 않는다.
읽는 규칙은 ⑫ 노드와 같다(wakeword_monitor·dsp_state 의 해석 함수와 같은 결과).
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional

from .dsp_state import agc_desired_from_env, echo_tuning_from_env
from .mic_check_rules import MicValues
from .wakeword_monitor import (EMERGENCY_SINGLE_FRAME_DEFAULT, LISTEN_MIN_RMS, SHORT_ANSWER_MIN_RMS,
                               WAKE_THRESHOLD_DEFAULT)

KEYS = {   # MicValues 필드 → .env 키
    "listen_min_rms": "VICA_LISTEN_MIN_RMS",
    "short_answer_min_rms": "VICA_SHORT_ANSWER_MIN_RMS",
    "wake_threshold": "VICA_WAKE_THRESHOLD",
    "emergency_single": "VICA_EMERGENCY_SINGLE_FRAME",
    "agc": "VICA_MIC_AGC_DESIRED",
    "echo_etail": "VICA_MIC_GAMMA_ETAIL",
    "echo_nlatten": "VICA_MIC_NLATTEN",
}
APPEND_HEADER = "# 마이크 점검기에서 추가 (2026-10-12 설계)"
_BACKUP_RE = re.compile(r"^\.env\.\d{8}-\d{6}(-\d+)?$")
_OFF = ("off", "none", "false", "no")


class EnvEditError(Exception):
    """쓰면 안 되는 상태 — 화면에 그대로 보여 줄 문장을 담는다."""


@dataclass(frozen=True)
class Change:
    field: str
    key: str
    old: str    # 바꾸기 전 ⑫ 가 쓰던 값(줄이 없으면 코드 기본값)
    new: str


def _line_key(line: str) -> Optional[str]:
    s = line.strip()
    if not s or s.startswith("#") or "=" not in s:
        return None
    left = s.split("=", 1)[0].strip()
    if left.startswith("export "):
        left = left[len("export "):].strip()
    return left or None


def _line_value(line: str) -> str:
    value = line.split("=", 1)[1].strip()
    if value[:1] in ("'", '"') and value.count(value[0]) >= 2:
        return value[1:value.index(value[0], 1)]
    return value.split(" #", 1)[0].split("\t#", 1)[0].strip()


def read_env(path: Path) -> dict:
    """주석 아닌 KEY=VALUE 를 읽는다(같은 키가 여럿이면 마지막 — python-dotenv 와 같다)."""
    out: dict = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            key = _line_key(line)
            if key:
                out[key] = _line_value(line)
    return out


def _float(mapping: Mapping, key: str, default: float) -> float:
    raw = str(mapping.get(key, "")).strip()
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if 0.0 < value < 1.0 else default


def values_from(mapping: Mapping) -> MicValues:
    """⑫ 노드가 같은 환경에서 쓰게 될 값. 노드와 같은 해석(빈 값·이상한 값은 기본값)."""
    if str(mapping.get(KEYS["emergency_single"], "")).strip().lower() in _OFF:
        raise EnvEditError("VICA_EMERGENCY_SINGLE_FRAME 이 off 입니다. 점검기는 한 조각 규칙이 켜진 상태를 "
                           "전제로 합니다 — 줄을 지우거나 숫자로 바꾼 뒤 다시 실행하세요.")
    agc = agc_desired_from_env(str(mapping.get(KEYS["agc"], "0.010")))   # ⑫ 의 기본도 "0.010"
    if agc is None:
        raise EnvEditError("VICA_MIC_AGC_DESIRED 가 꺼져 있습니다(off·0). 점검기는 AGC 목표가 있는 상태를 전제로 합니다.")
    etail, nlatten = echo_tuning_from_env(str(mapping.get(KEYS["echo_etail"], "")),
                                          str(mapping.get(KEYS["echo_nlatten"], "")))
    return MicValues(
        listen_min_rms=_float(mapping, KEYS["listen_min_rms"], LISTEN_MIN_RMS),
        short_answer_min_rms=_float(mapping, KEYS["short_answer_min_rms"], SHORT_ANSWER_MIN_RMS),
        wake_threshold=_float(mapping, KEYS["wake_threshold"], WAKE_THRESHOLD_DEFAULT),
        emergency_single=_float(mapping, KEYS["emergency_single"], EMERGENCY_SINGLE_FRAME_DEFAULT),
        agc=agc,
        # 줄이 없으면 ⑫ 는 칩에 쓰지 않는다 — 칩 공장값(1.0·꺼짐)으로 본다.
        echo_etail=1.0 if etail is None else etail,
        echo_nlatten=0 if nlatten is None else nlatten,
    )


def format_value(field: str, value) -> str:
    if field in ("listen_min_rms", "short_answer_min_rms"):
        text = f"{float(value):.4f}".rstrip("0")
        return text + "0" if text.endswith(".") else text
    if field in ("wake_threshold", "emergency_single"):
        return f"{float(value):.2f}"
    if field == "agc":
        return f"{float(value):.3f}"
    if field == "echo_etail":
        return f"{float(value):.1f}"
    return str(int(value))


def plan_changes(before: MicValues, after: MicValues) -> list:
    """before(지금 ⑫ 가 쓰는 값) → after 에서 달라진 값마다 Change. KEYS 순서."""
    changes = []
    for field, key in KEYS.items():
        old, new = format_value(field, getattr(before, field)), format_value(field, getattr(after, field))
        if old != new:
            changes.append(Change(field, key, old, new))
    return changes


def change_line(change: Change, note: str) -> str:
    return f"{change.key}={change.new}  # {note} (전 {change.old})"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """전원이 끊겨도 .env 가 반쯤 쓰인 채 남지 않게 임시 파일에 쓰고 바꿔치기한다."""
    path = Path(path)
    mode = os.stat(path).st_mode & 0o7777 if path.exists() else None
    fd, tmp = tempfile.mkstemp(prefix=".env.tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def backup_file(path: Path, backup_dir: Path, now: Optional[datetime] = None, suffix: str = "") -> Path:
    path, backup_dir = Path(path), Path(backup_dir)
    if not path.exists():
        raise EnvEditError(f"{path} 가 없습니다.")
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    target = backup_dir / f".env.{stamp}{suffix}"
    n = 1
    while target.exists():
        target = backup_dir / f".env.{stamp}-{n}{suffix}"
        n += 1
    shutil.copy2(path, target)
    return target


def write_changes(path: Path, changes: list, note: str, backup_dir: Path,
                  now: Optional[datetime] = None) -> Path:
    """백업한 뒤 바꾼다. 백업 경로를 돌려준다. 바꿀 것이 없거나 같은 키가 두 줄 이상이면 쓰지 않는다."""
    path = Path(path)
    if not changes:
        raise EnvEditError(".env 와 다른 값이 없습니다.")
    if not path.exists():
        raise EnvEditError(f"{path} 가 없습니다.")
    lines = path.read_text(encoding="utf-8").splitlines()
    for c in changes:
        hits = [i for i, line in enumerate(lines) if _line_key(line) == c.key]
        if len(hits) > 1:
            raise EnvEditError(f"{c.key} 가 .env 에 {len(hits)}줄 있어 어느 줄이 쓰일지 알 수 없습니다. "
                               "하나만 남긴 뒤 다시 저장하세요.")
    backup = backup_file(path, backup_dir, now)
    appended = []
    for c in changes:
        hits = [i for i, line in enumerate(lines) if _line_key(line) == c.key]
        if hits:
            lines[hits[0]] = change_line(c, note)
        else:
            appended.append(change_line(c, note))
    if appended:
        if APPEND_HEADER not in lines:
            if lines and lines[-1].strip():
                lines.append("")
            lines.append(APPEND_HEADER)
        lines.extend(appended)
    _atomic_write_bytes(path, ("\n".join(lines) + "\n").encode("utf-8"))
    return backup


def latest_backup(backup_dir: Path) -> Optional[Path]:
    backup_dir = Path(backup_dir)
    if not backup_dir.is_dir():
        return None
    found = sorted(p for p in backup_dir.iterdir() if _BACKUP_RE.match(p.name))
    return found[-1] if found else None


def restore_latest(path: Path, backup_dir: Path, now: Optional[datetime] = None) -> Path:
    """마이크 값 7개(KEYS)만 가장 최근 백업(마지막 저장 직전)의 값으로 되돌린다. 다른 줄은 그대로 둔다.

    되돌리기 전 지금 파일 전체를 .before-restore 로 남긴다. 백업에 있던 키는 그 값으로 쓰고(제자리 교체, 없으면
    끝에 추가), 백업에 없던 키는 지금 파일에서 줄을 지운다 — ⑫ 가 기본값으로 돌아가게(백업 때와 같게).
    """
    path = Path(path)
    latest = latest_backup(backup_dir)
    if latest is None:
        raise EnvEditError(f"되돌릴 백업이 없습니다 ({backup_dir}).")
    if not path.exists():
        raise EnvEditError(f"{path} 가 없습니다.")
    saved = read_env(latest)
    lines = path.read_text(encoding="utf-8").splitlines()
    for key in KEYS.values():
        hits = [i for i, line in enumerate(lines) if _line_key(line) == key]
        if len(hits) > 1:
            raise EnvEditError(f"{key} 가 .env 에 {len(hits)}줄 있어 어느 줄이 쓰일지 알 수 없습니다. "
                               "하나만 남긴 뒤 다시 하세요.")
    backup_file(path, backup_dir, now, suffix=".before-restore")
    note = f"{(now or datetime.now()):%m-%d} 원래대로 (백업 {latest.name})"
    appended = []
    drop = set()
    for key in KEYS.values():
        hits = [i for i, line in enumerate(lines) if _line_key(line) == key]
        if key in saved:
            text = f"{key}={saved[key]}  # {note}"
            if hits:
                lines[hits[0]] = text
            else:
                appended.append(text)
        else:
            drop.update(hits)
    lines = [line for i, line in enumerate(lines) if i not in drop]
    if appended:
        if APPEND_HEADER not in lines:
            if lines and lines[-1].strip():
                lines.append("")
            lines.append(APPEND_HEADER)
        lines.extend(appended)
    _atomic_write_bytes(path, ("\n".join(lines) + "\n").encode("utf-8"))
    return latest

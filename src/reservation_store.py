"""예약 저장(설계 2026-11-18 5.4) — 순수 모듈, ROS 없음.

    ~/vica_data/reservations/facilities.yaml     시설(이름·별칭·기본 시간·운영 시각)
    ~/vica_data/reservations/reservations.json   예약 배열

쓰기는 같은 폴더의 임시 파일에 쓴 뒤 이름을 바꾼다. 읽히지 않는 파일은 덮어쓰지 않는다(OSError) — 잘린 파일을
빈 목록으로 보고 쓰면 그 전 예약이 전부 사라진다.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

DEFAULT_DIR = "~/vica_data/reservations"
REPO_FACILITIES = Path(__file__).resolve().parents[1] / "config" / "reservations" / "facilities.yaml"


@dataclass(frozen=True)
class Facility:
    name: str
    aliases: tuple[str, ...]
    default_hours: int
    open: str       # "HH:MM"
    close: str      # "HH:MM"


@dataclass(frozen=True)
class Reservation:
    id: str
    facility: str
    date: str       # YYYY-MM-DD
    start: str      # HH:MM
    end: str        # HH:MM
    name: str
    source: str
    created_at: str


class ConflictError(Exception):
    """같은 시설·같은 날에 시간이 겹치는 예약이 있다."""


def reservation_dir() -> Path:
    return Path(os.environ.get("VICA_RESERVATION_DIR", DEFAULT_DIR)).expanduser()


def load_facilities(path: Path) -> list[Facility]:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError):
        return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        out.append(Facility(
            name=str(item["name"]), aliases=tuple(str(a) for a in item.get("aliases") or ()),
            default_hours=int(item.get("default_hours", 2)),
            open=str(item.get("open", "09:00")), close=str(item.get("close", "18:00"))))
    return out


def default_facilities() -> list[Facility]:
    """운영 폴더의 시설 파일. 없으면 저장소에 든 시연용 파일."""
    own = reservation_dir() / "facilities.yaml"
    return load_facilities(own if own.exists() else REPO_FACILITIES)


class ReservationStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def _read(self) -> list[Reservation]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError("배열이 아니다")
            return [Reservation(**item) for item in data]
        except (ValueError, TypeError) as exc:
            raise OSError(f"예약 파일을 읽을 수 없다({self.path}): {exc}") from exc

    def _write(self, items: list[Reservation]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(items, key=lambda r: (r.date, r.start, r.facility))
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".reservations-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump([asdict(r) for r in ordered], f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def list(self) -> list[Reservation]:
        return sorted(self._read(), key=lambda r: (r.date, r.start, r.facility))

    def conflicts(self, facility: str, date: str, start: str, end: str) -> list[Reservation]:
        """같은 시설·같은 날에 [start, end) 와 겹치는 예약. 맞닿는 것은 겹침이 아니다."""
        return [r for r in self._read()
                if r.facility == facility and r.date == date and r.start < end and start < r.end]

    def add(self, facility: str, date: str, start: str, end: str, name: str, now: datetime,
            source: str = "voice") -> Reservation:
        items = self._read()        # 쓰기 직전에 다시 읽어 겹침을 본다
        if any(r.facility == facility and r.date == date and r.start < end and start < r.end for r in items):
            raise ConflictError(f"{facility} {date} {start}~{end}")
        taken = {r.id for r in items}
        base = f"r{now:%Y%m%d-%H%M%S}"
        rid, n = base, 1
        while rid in taken:
            n += 1
            rid = f"{base}-{n}"
        res = Reservation(id=rid, facility=facility, date=date, start=start, end=end, name=name,
                          source=source, created_at=now.isoformat(timespec="seconds"))
        self._write([*items, res])
        return res

    def cancel(self, reservation_id: str) -> Optional[Reservation]:
        items = self._read()
        hit = next((r for r in items if r.id == reservation_id), None)
        if hit is None:
            return None
        self._write([r for r in items if r.id != reservation_id])
        return hit

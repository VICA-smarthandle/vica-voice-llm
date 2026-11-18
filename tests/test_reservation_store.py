"""예약 저장(설계 2026-11-18 5.4)."""
import json
from datetime import datetime

import pytest

from src.reservation_store import ConflictError, Facility, ReservationStore, load_facilities

NOW = datetime(2026, 11, 18, 14, 3, 21)


def test_add_then_list_sorted(tmp_path):
    s = ReservationStore(tmp_path / "reservations.json")
    late = s.add("세미나실", "2026-11-19", "14:00", "16:00", "김토니", NOW)
    early = s.add("세미나실", "2026-11-19", "10:00", "11:00", "이서연", NOW)
    assert [r.id for r in s.list()] == [early.id, late.id]
    assert early.id and early.id != late.id and late.source == "voice"
    on_disk = json.loads((tmp_path / "reservations.json").read_text(encoding="utf-8"))
    assert on_disk[0]["name"] == "이서연" and on_disk[1]["end"] == "16:00"


def test_overlap_is_same_facility_same_day_only(tmp_path):
    s = ReservationStore(tmp_path / "reservations.json")
    s.add("세미나실", "2026-11-19", "14:00", "16:00", "김토니", NOW)
    assert s.conflicts("세미나실", "2026-11-19", "15:00", "17:00")
    assert not s.conflicts("세미나실", "2026-11-19", "16:00", "18:00")   # 맞닿는 것은 겹침이 아니다
    assert not s.conflicts("상담실", "2026-11-19", "14:00", "16:00")
    assert not s.conflicts("세미나실", "2026-11-20", "14:00", "16:00")
    with pytest.raises(ConflictError):
        s.add("세미나실", "2026-11-19", "15:00", "17:00", "박준호", NOW)
    assert len(s.list()) == 1


def test_missing_file_is_empty_and_add_creates_the_folder(tmp_path):
    s = ReservationStore(tmp_path / "new" / "reservations.json")
    assert s.list() == []
    s.add("상담실", "2026-11-19", "09:00", "11:00", "박준호", NOW)
    assert (tmp_path / "new" / "reservations.json").exists()


def test_broken_file_is_not_overwritten(tmp_path):
    path = tmp_path / "reservations.json"
    path.write_text('[{"id": "r1", "facil', encoding="utf-8")
    s = ReservationStore(path)
    with pytest.raises(OSError):
        s.add("세미나실", "2026-11-19", "14:00", "16:00", "김토니", NOW)
    with pytest.raises(OSError):
        s.list()
    assert path.read_text(encoding="utf-8") == '[{"id": "r1", "facil'


def test_cancel_removes_one_and_returns_it(tmp_path):
    s = ReservationStore(tmp_path / "reservations.json")
    a = s.add("세미나실", "2026-11-19", "14:00", "16:00", "김토니", NOW)
    assert s.cancel(a.id) == a and s.list() == [] and s.cancel(a.id) is None


def test_load_facilities(tmp_path):
    path = tmp_path / "facilities.yaml"
    path.write_text("- name: 세미나실\n  aliases: [세미나룸]\n  default_hours: 2\n  open: '09:00'\n  close: '18:00'\n", encoding="utf-8")
    assert load_facilities(path) == [Facility("세미나실", ("세미나룸",), 2, "09:00", "18:00")]
    assert load_facilities(tmp_path / "none.yaml") == []

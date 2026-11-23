"""예약 점검 — 실기에서 같은 시간을 다시 쓰려고 지울 때 쓴다(설계 2026-11-18 5.4)."""
from __future__ import annotations

import argparse

from src.reservation_store import ReservationStore, reservation_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("cancel").add_argument("id")
    args = ap.parse_args()
    store = ReservationStore(reservation_dir() / "reservations.json")
    if args.cmd == "list":
        rows = store.list()
        for r in rows:
            print(f"{r.id}  {r.date} {r.start}~{r.end}  {r.facility}  {r.name}  ({r.source}, {r.created_at})")
        print(f"{len(rows)}건 · {store.path}")
    else:
        hit = store.cancel(args.id)
        print(f"지움: {hit.date} {hit.start}~{hit.end} {hit.facility} {hit.name}" if hit else "그런 예약이 없습니다")


if __name__ == "__main__":
    main()

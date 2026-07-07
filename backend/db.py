"""목적지 SQLite 저장소 (Phase 5 테스트용 관리자 백엔드).

목적지 전체를 JSON 한 덩어리로 저장하는 단순한 구조다.
(스키마가 자주 바뀌는 개발 단계에서는 컬럼 분해보다 관리가 쉽다)

첫 실행 시 DB 가 비어 있으면 config/destinations.yaml 을 시드로 넣는다.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from src.destination_loader import load_destinations
from src.schema import DestinationData, DestinationPose

# DB 파일은 백엔드 폴더에 둔다 (git 에는 올리지 않음).
DEFAULT_DB_PATH = Path(__file__).resolve().parent / "destinations.db"


class DestinationDB:
    def __init__(self, path: Path | str = DEFAULT_DB_PATH):
        # check_same_thread=False: FastAPI 는 요청을 여러 스레드에서 처리할 수 있다.
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS destinations (id TEXT PRIMARY KEY, data TEXT NOT NULL)"
        )
        self._seed_if_empty()

    def _seed_if_empty(self) -> None:
        """DB 가 비어 있으면 YAML 목적지를 초기 데이터로 넣는다."""
        count = self._conn.execute("SELECT COUNT(*) FROM destinations").fetchone()[0]
        if count == 0:
            for dest in load_destinations():
                self.upsert(dest)

    def list_all(self) -> list[DestinationData]:
        rows = self._conn.execute("SELECT data FROM destinations ORDER BY id").fetchall()
        return [DestinationData(**json.loads(r[0])) for r in rows]

    def get(self, dest_id: str) -> Optional[DestinationData]:
        row = self._conn.execute(
            "SELECT data FROM destinations WHERE id = ?", (dest_id,)
        ).fetchone()
        return DestinationData(**json.loads(row[0])) if row else None

    def upsert(self, dest: DestinationData) -> None:
        self._conn.execute(
            "INSERT INTO destinations (id, data) VALUES (?, ?) "
            "ON CONFLICT(id) DO UPDATE SET data = excluded.data",
            (dest.id, dest.model_dump_json()),
        )
        self._conn.commit()

    def update_pose(self, dest_id: str, pose: DestinationPose) -> Optional[DestinationData]:
        """목적지의 pose 만 바꾼다 (calibration 용). 없으면 None."""
        dest = self.get(dest_id)
        if dest is None:
            return None
        dest.pose = pose
        self.upsert(dest)
        return dest

    def search(self, query: str) -> list[DestinationData]:
        """name / aliases / room 에 대한 단순 부분 일치 검색 (공백 무시)."""
        norm_query = query.replace(" ", "")
        if not norm_query:
            return []
        results = []
        for dest in self.list_all():
            haystacks = [dest.name, *dest.aliases]
            if dest.room:
                haystacks.append(dest.room)
            if any(norm_query in h.replace(" ", "") for h in haystacks):
                results.append(dest)
        return results

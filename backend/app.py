"""목적지 관리 FastAPI 백엔드 (CLAUDE.md Phase 5).

테스트/관리자용이다. LangChain destination tool 은 이 API 를 '조회만' 한다.
목적지 추가와 pose 수정은 관리자 앱 / calibration tool 이 한다.

실행:
    .venv/bin/uvicorn backend.app:app --host 0.0.0.0 --port 8000

API:
    GET    /destinations                  전체 목록
    GET    /destinations/search?query=..  검색 (name/aliases/room 부분 일치)
    POST   /destinations                  추가/수정 (upsert)
    PATCH  /destinations/{id}/pose        pose 만 수정 (calibration)
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from src.schema import DestinationData, DestinationPose

from .db import DestinationDB

app = FastAPI(title="VICA Destinations API")
db = DestinationDB()


@app.get("/destinations")
def list_destinations() -> list[DestinationData]:
    return db.list_all()


@app.get("/destinations/search")
def search_destinations(query: str) -> list[DestinationData]:
    return db.search(query)


@app.post("/destinations")
def upsert_destination(dest: DestinationData) -> DestinationData:
    db.upsert(dest)
    return dest


@app.patch("/destinations/{dest_id}/pose")
def update_pose(dest_id: str, pose: DestinationPose) -> DestinationData:
    dest = db.update_pose(dest_id, pose)
    if dest is None:
        raise HTTPException(status_code=404, detail=f"목적지 없음: {dest_id}")
    return dest

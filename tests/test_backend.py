"""목적지 백엔드(Phase 5) API 검증. 실제 서버/네트워크 없이 TestClient 로 돈다."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import backend.app as app_module
from backend.db import DestinationDB


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """테스트마다 임시 DB 를 쓰는 클라이언트 (YAML 시드 포함)."""
    monkeypatch.setattr(app_module, "db", DestinationDB(tmp_path / "test.db"))
    return TestClient(app_module.app)


def test_list_returns_seeded_destinations(client):
    resp = client.get("/destinations")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) >= 1
    assert any(d["id"] == "starlight_1f_restroom" for d in items)


def test_search_by_room_number(client):
    resp = client.get("/destinations/search", params={"query": "407호"})
    assert resp.status_code == 200
    ids = [d["id"] for d in resp.json()]
    assert "engineering_4f_room_407_prof_yoon_jiyoung_office" in ids


def test_search_ignores_spaces(client):
    resp = client.get("/destinations/search", params={"query": "윤지영교수님"})
    assert len(resp.json()) >= 1


def test_search_no_match(client):
    resp = client.get("/destinations/search", params={"query": "존재하지않는곳"})
    assert resp.json() == []


def test_post_then_search(client):
    new_dest = {"id": "test_cafe", "name": "테스트 카페", "aliases": ["카페"]}
    resp = client.post("/destinations", json=new_dest)
    assert resp.status_code == 200
    resp = client.get("/destinations/search", params={"query": "카페"})
    assert [d["id"] for d in resp.json()] == ["test_cafe"]


def test_patch_pose(client):
    pose = {"frame_id": "map", "x": 1.5, "y": -2.0, "yaw": 3.14}
    resp = client.patch("/destinations/starlight_1f_restroom/pose", json=pose)
    assert resp.status_code == 200
    assert resp.json()["pose"]["x"] == 1.5
    # 다시 조회해도 유지되는지 (DB 반영 확인)
    resp = client.get("/destinations")
    dest = next(d for d in resp.json() if d["id"] == "starlight_1f_restroom")
    assert dest["pose"]["y"] == -2.0


def test_patch_pose_unknown_id_404(client):
    resp = client.patch("/destinations/no_such_id/pose", json={"x": 0.0})
    assert resp.status_code == 404

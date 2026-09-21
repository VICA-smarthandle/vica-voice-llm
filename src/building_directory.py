"""건물 디렉터리 (스펙 3.1): 이 지도(층) 밖의 장소를 이름·건물·층으로만 안다.

    ~/vica_data/destinations/directory.yaml
    - name: 세미나실
      building: 로봇관
      floor: 3

갈 수는 없다(지도 = 한 층). 모델은 "3층에 있다, 층 이동은 못 한다"고 말하고 등록 목적지
'엘리베이터'가 있으면 그리로 안내를 제안한다(지시문 규칙). 파일이 없으면 빈 목록 — 그러면
블록도 없고 모델은 지금처럼 "모른다"고 답한다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

DEFAULT_DIRECTORY_YAML = "~/vica_data/destinations/directory.yaml"


@dataclass(frozen=True)
class DirectoryEntry:
    name: str
    building: str
    floor: int


def directory_path() -> str:
    return os.environ.get("VICA_DIRECTORY_YAML", DEFAULT_DIRECTORY_YAML)


def load_directory(path: str) -> list[DirectoryEntry]:
    try:
        data = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []
    if not isinstance(data, list):
        return []
    entries: list[DirectoryEntry] = []
    for row in data:
        if not isinstance(row, dict) or not row.get("name"):
            continue
        try:
            floor = int(row.get("floor"))
        except (TypeError, ValueError):
            continue
        entries.append(DirectoryEntry(name=str(row["name"]).strip(),
                                      building=str(row.get("building") or "").strip(), floor=floor))
    return entries


def format_directory_block(entries: list[DirectoryEntry], current_building: str,
                           current_floor: Optional[int]) -> str:
    others = [e for e in entries
              if not (e.building == current_building and current_floor is not None and e.floor == current_floor)]
    if not others:
        return ""
    lines = [f"- {e.name}: {e.building} {e.floor}층".replace(":  ", ": ") for e in others]
    return "\n[다른 층 장소] (이 지도에 없어 직접 갈 수 없다)\n" + "\n".join(lines) + "\n"

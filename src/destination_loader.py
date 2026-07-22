"""목적지 목록을 읽어 DestinationData 목록으로 바꾼다.

목적지 원본은 config/destinations.yaml 파일이다. (목적지와 pose 는 관리자 앱/
calibration tool 이 관리하고, 여기서는 읽기만 한다.)

confirm_prompt / arrival_message 가 비어 있으면 name 으로 자동 생성한다.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .schema import DestinationData

# 기본 목적지 파일 경로 (프로젝트 루트 기준).
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "destinations.yaml"


def _josa_euro(word: str) -> str:
    """단어 뒤에 붙는 조사 '으로 / 로' 를 받침에 맞게 돌려준다.

    받침 없음 또는 ㄹ 받침이면 '로', 그 외 받침이면 '으로'.
    예) 화장실 -> 로, 안내센터 -> 로, 식당 -> 으로
    """
    if not word:
        return "로"
    last = word[-1]
    if not ("가" <= last <= "힣"):  # 한글이 아니면 '로'로 둔다
        return "로"
    jongseong = (ord(last) - 0xAC00) % 28  # 0=받침없음, 8=ㄹ
    return "로" if jongseong in (0, 8) else "으로"


def _fill_defaults(dest: DestinationData) -> DestinationData:
    """confirm_prompt / arrival_message 가 비어 있으면 name 으로 채운다."""
    if not dest.confirm_prompt:
        dest.confirm_prompt = f"{dest.name}{_josa_euro(dest.name)} 안내해드릴까요?"
    if not dest.arrival_message and dest.is_approachable:
        dest.arrival_message = f"{dest.name} 앞에 도착했습니다."
    return dest


def _load_from_yaml(path: Path | str) -> list[DestinationData]:
    """YAML 파일을 읽어 DestinationData 목록을 돌려준다."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    items = raw.get("destinations", []) if raw else []
    return [_fill_defaults(DestinationData(**item)) for item in items]


def load_destinations(path: Path | str = DEFAULT_PATH) -> list[DestinationData]:
    """config/destinations.yaml 에서 목적지 목록을 읽어 돌려준다."""
    return _load_from_yaml(path)

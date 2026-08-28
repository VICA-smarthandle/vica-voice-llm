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
    """config/destinations.yaml 에서 목적지 목록을 읽어, 공개 목적지만 돌려준다.

    목적지 원본은 destinations.yaml 단일 소스다(CLAUDE.md Phase 5). 비공개
    (authorization != "public") 목적지는 음성 안내 대상에서 제외한다.
    """
    return [
        destination
        for destination in _load_from_yaml(path)
        if destination.authorization == "public"
    ]


def build_place_hint(destinations, max_chars: int = 200) -> str:
    """장소 이름·별칭을 STT 귀띔(whisper initial_prompt)용 한 줄로 만든다.

    자유 명령 창에서 목적지 오전사('휴게실'→'조계실', '안내소'→'음내소')를
    줄인다 (2026-08-28 실측). 귀띔은 기울이기일 뿐이라 길수록 부작용이
    커지므로 max_chars 에서 자른다 — 앞선 목적지가 우선 생존한다.
    """
    # 이름을 전부 먼저, 별칭은 남는 자리에 — 별칭 많은 목적지가 상한을
    # 독식해 다른 목적지가 통째로 빠지는 일을 막는다 (실데이터에서 '입구'
    # 소실 실측).
    seen: set[str] = set()
    words: list[str] = []
    for group in ([d.name] for d in destinations), \
                 (getattr(d, "aliases", []) for d in destinations):
        for items in group:
            for word in items:
                word = word.strip()
                if word and word not in seen:
                    seen.add(word)
                    words.append(word)
    hint = ""
    for word in words:
        candidate = f"{hint}, {word}" if hint else word
        if len(candidate) > max_chars:
            break
        hint = candidate
    return hint

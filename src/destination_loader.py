"""목적지 목록을 읽어 DestinationData 목록으로 바꾼다."""
from __future__ import annotations

from pathlib import Path

import yaml

from .mission_phrases import ARRIVED_FALLBACK
from .schema import DestinationData

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "destinations.yaml"


def _josa_euro(word: str) -> str:
    """단어 뒤에 붙는 조사 '으로 / 로' 를 받침에 맞게 돌려준다."""
    if not word:
        return "로"
    last = word[-1]
    if not ("가" <= last <= "힣"):
        return "로"
    jongseong = (ord(last) - 0xAC00) % 28
    return "로" if jongseong in (0, 8) else "으로"


def _fill_defaults(dest: DestinationData) -> DestinationData:
    """confirm_prompt / arrival_message 가 비어 있으면 name 으로 채운다."""
    if not dest.confirm_prompt:
        dest.confirm_prompt = f"{dest.name}{_josa_euro(dest.name)} 안내해드릴까요?"
    if not dest.arrival_message and dest.is_approachable:
        dest.arrival_message = ARRIVED_FALLBACK.format(name=dest.name)
    return dest


def _load_from_yaml(path: Path | str) -> list[DestinationData]:
    """YAML 파일을 읽어 DestinationData 목록을 돌려준다."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    items = raw.get("destinations", []) if raw else []
    return [_fill_defaults(DestinationData(**item)) for item in items]


def load_destinations(path: Path | str = DEFAULT_PATH) -> list[DestinationData]:
    """config/destinations.yaml 에서 목적지 목록을 읽어, 공개 목적지만 돌려준다."""
    return [
        destination
        for destination in _load_from_yaml(path)
        if destination.authorization == "public"
    ]


def load_voice_destinations(path: Path | str = DEFAULT_PATH) -> list[DestinationData]:
    """LLM 이 볼 목적지 — 공개 목적지 + 비공개 목적지를 '접근 불가'로 (2026-10-10 사용자 결정 A)."""
    return [
        destination if destination.authorization == "public"
        else destination.model_copy(update={"is_approachable": False})
        for destination in _load_from_yaml(path)
    ]


def build_place_hint(destinations, max_chars: int = 200) -> str:
    """장소 이름·별칭을 STT 귀띔(whisper initial_prompt)용 한 줄로 만든다."""
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

"""사용자가 말한 '목적지 표현'을 실제 목적지(DestinationData)로 매칭한다."""
from __future__ import annotations

import re
from typing import Optional

from .ko_numbers import korean_numbers_to_digits
from .schema import DestinationData


def _normalize(text: str) -> str:
    """비교를 위해 공백을 제거하고 소문자로 바꾼다. 한글 숫자 읽기('사백칠호')는 숫자('407호')로 —"""
    return korean_numbers_to_digits(re.sub(r"\s+", "", text)).strip().lower()


def _score(query_norm: str, target_norm: str) -> int:
    """query 와 target(name 또는 alias) 의 매칭 점수. 0이면 매칭 아님."""
    if not query_norm or not target_norm:
        return 0
    if query_norm == target_norm:
        return 3
    if target_norm in query_norm:
        return 2
    if query_norm in target_norm:
        return 1
    return 0


def match_destination(
    query: Optional[str],
    destinations: list[DestinationData],
) -> Optional[DestinationData]:
    """query(목적지 표현)에 가장 잘 맞는 목적지를 돌려준다. 없으면 None."""
    if not query:
        return None
    query_norm = _normalize(query)

    best: Optional[DestinationData] = None
    best_score = 0
    for dest in destinations:
        candidates = [dest.name, *dest.aliases]
        score = max(_score(query_norm, _normalize(c)) for c in candidates)
        if score > best_score:
            best_score = score
            best = dest
    return best


def tied_destinations(
    query: Optional[str],
    destinations: list[DestinationData],
) -> list[DestinationData]:
    """query 에 똑같이 가장 잘 맞는 갈 수 있는 목적지들(목록 순서). 둘 이상이면 어느 곳인지 모르는 말이다."""
    if not query:
        return []
    query_norm = _normalize(query)
    scored = [(max(_score(query_norm, _normalize(c)) for c in [d.name, *d.aliases]), d)
              for d in destinations if d.is_approachable]
    best = max((sc for sc, _ in scored), default=0)
    return [d for sc, d in scored if best > 0 and sc == best]

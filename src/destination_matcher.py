"""사용자가 말한 '목적지 표현'을 실제 목적지(DestinationData)로 매칭한다."""
from __future__ import annotations

import re
from typing import Optional

from .schema import DestinationData


def _normalize(text: str) -> str:
    """비교를 위해 공백을 제거하고 소문자로 바꾼다."""
    return re.sub(r"\s+", "", text).strip().lower()


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

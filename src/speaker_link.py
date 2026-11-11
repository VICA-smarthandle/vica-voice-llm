"""의도 노드 쪽 화자 판정 보관함 — 호출 노드가 `/vica/user_speaker`(JSON)로 보낸 판정을 같은 글자의 의도에 싣는다.

판정과 글자(`/vica/user_text`)는 다른 토픽이라 도착 순서가 보장되지 않는다. 의도 노드는 LLM 해석이 끝난 뒤
의도를 발행할 때 글자로 판정을 찾는다(설계 6.1, 계획 작업 6). 순수 모듈.
"""
from __future__ import annotations

import json
from typing import Dict, Optional, Tuple

TTL_SEC = 15.0   # 호출 창 판정은 미션 응답을 최대 6 s 기다렸다 나간다 — 넉넉히


class SpeakerLink:
    def __init__(self, ttl_sec: float = TTL_SEC):
        self.ttl_sec = ttl_sec
        self._by_text: Dict[str, Tuple[float, dict]] = {}

    def put(self, json_text: str, now: float) -> None:
        try:
            data = json.loads(json_text)
        except (ValueError, TypeError):
            return
        if not isinstance(data, dict) or "text" not in data:
            return
        self._prune(now)
        self._by_text[str(data["text"]).strip()] = (now, data)

    def take(self, text: str, now: float) -> Optional[dict]:
        self._prune(now)
        hit = self._by_text.pop((text or "").strip(), None)
        return None if hit is None else hit[1]

    def _prune(self, now: float) -> None:
        for key in [k for k, (t, _) in self._by_text.items() if now - t > self.ttl_sec]:
            del self._by_text[key]

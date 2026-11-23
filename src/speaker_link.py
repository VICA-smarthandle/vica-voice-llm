"""의도 노드 쪽 화자 판정 보관함 — 호출 노드가 `/vica/user_speaker`(JSON)로 보낸 판정을 같은 글자의 의도에 싣는다."""
from __future__ import annotations

import json
from typing import Dict, Optional, Tuple

TTL_SEC = 15.0


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


LABEL_PREFIX = "spk:"


def verdict_label(json_text: str) -> str:
    """판정 JSON → /vica/user_audio layout.dim 라벨. 소리 의도 경로는 소리와 판정을 한 메시지로 짝짓는다(검토 C2)."""
    return LABEL_PREFIX + json_text


def verdict_from_labels(labels) -> Optional[dict]:  # noqa: ANN001
    """layout.dim 라벨들에서 판정을 꺼낸다. 없거나 깨졌으면 None."""
    for label in labels:
        if isinstance(label, str) and label.startswith(LABEL_PREFIX):
            try:
                data = json.loads(label[len(LABEL_PREFIX):])
            except ValueError:
                return None
            return data if isinstance(data, dict) else None
    return None

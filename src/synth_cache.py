"""합성 결과 캐시 — 같은 문장을 두 번 합성하지 않는다.

확인 질문("~로 안내해드릴까요?")·도착 멘트·접수 멘트는 고정 문장인데
매번 합성해 응답이 늦었다 (2026-08-28 실측). 문장 단위로 (wav, rate) 를
저장하고 용량을 넘으면 가장 오래 안 쓴 것부터 버린다(LRU). 기동 시
미리 데우는 것(prewarm)은 ros_tts_node 몫이다.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import numpy as np


class SynthCache:
    def __init__(self, capacity: int = 64):
        self._capacity = capacity
        self._store: OrderedDict[str, tuple[np.ndarray, int]] = OrderedDict()

    def get(self, text: str) -> Optional[tuple[np.ndarray, int]]:
        hit = self._store.get(text)
        if hit is not None:
            self._store.move_to_end(text)
        return hit

    def put(self, text: str, wav: np.ndarray, rate: int) -> None:
        self._store[text] = (wav, rate)
        self._store.move_to_end(text)
        while len(self._store) > self._capacity:
            self._store.popitem(last=False)

    def __len__(self) -> int:
        return len(self._store)

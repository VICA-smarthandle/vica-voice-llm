"""합성 결과 캐시 — 같은 문장을 두 번 합성하지 않는다."""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import numpy as np


class SynthCache:
    def __init__(self, capacity: int = 64):
        self._capacity = capacity
        self._store: OrderedDict[str, tuple[np.ndarray, int]] = OrderedDict()
        self._pinned: dict[str, tuple[np.ndarray, int]] = {}

    def get(self, text: str) -> Optional[tuple[np.ndarray, int]]:
        hit = self._pinned.get(text)
        if hit is not None:
            return hit
        hit = self._store.get(text)
        if hit is not None:
            self._store.move_to_end(text)
        return hit

    def put(self, text: str, wav: np.ndarray, rate: int, pinned: bool = False) -> None:
        if pinned:
            self._pinned[text] = (wav, rate)
            self._store.pop(text, None)
            return
        if text in self._pinned:
            return
        self._store[text] = (wav, rate)
        self._store.move_to_end(text)
        while len(self._store) > self._capacity:
            self._store.popitem(last=False)

    def __len__(self) -> int:
        return len(self._store) + len(self._pinned)

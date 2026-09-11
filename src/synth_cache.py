"""합성 결과 캐시 — 같은 문장을 두 번 합성하지 않는다."""
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

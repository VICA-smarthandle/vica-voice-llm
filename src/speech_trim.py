"""말소리 자르기 — 앞뒤 조용한 부분과 말 뒤 작은 잡음 덩어리를 버린다."""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
FRAME_SEC = 0.02
REL_DB = 30.0
FLOOR = 0.003
PAD_SEC = 0.10
MERGE_GAP_SEC = 0.30
KEEP_REL = 0.2


def trim_speech(x: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """앞뒤 조용한 부분을 잘라 말소리 + 앞뒤 PAD_SEC 만 남긴다. 말이 없으면 빈 배열."""
    frame = int(sr * FRAME_SEC)
    n = len(x) // frame
    if n == 0:
        return x[:0]
    rms = np.sqrt(np.mean(x[: n * frame].reshape(n, frame).astype(np.float64) ** 2, axis=1))
    peak = float(rms.max())
    if peak < FLOOR:
        return x[:0]
    voiced = np.flatnonzero(rms >= max(FLOOR, peak * 10 ** (-REL_DB / 20)))
    max_gap = int(round(MERGE_GAP_SEC / FRAME_SEC))
    groups = [[int(voiced[0]), int(voiced[0])]]
    for i in voiced[1:]:
        if int(i) - groups[-1][1] - 1 <= max_gap:
            groups[-1][1] = int(i)
        else:
            groups.append([int(i), int(i)])
    loud = [g for g in groups if float(rms[g[0]:g[1] + 1].max()) >= KEEP_REL * peak]
    pad = int(sr * PAD_SEC)
    start = max(0, loud[0][0] * frame - pad)
    end = min(len(x), (loud[-1][1] + 1) * frame + pad)
    return x[start:end]

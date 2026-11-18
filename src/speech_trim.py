"""말소리 자르기 — 앞뒤 조용한 부분과 말 뒤 작은 잡음 덩어리를 버린다.

화자 판정(src/speaker_follow.py)과 오프라인 평가기(tools/spk_verify_eval.py)가 같이 쓴다. 상수의 근거는
11-10 녹음 실측(커밋 563191d). 순수 모듈.
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
FRAME_SEC = 0.02          # 말소리 자르기 창 20 ms
REL_DB = 30.0             # 가장 큰 창보다 30 dB 넘게 작은 창은 말이 아니다
FLOOR = 0.003             # 이것보다 작은 창은 크기와 상관없이 말이 아니다(칩 처리음 바닥 잡음 수준)
PAD_SEC = 0.10            # 자른 말 앞뒤로 남길 여유
MERGE_GAP_SEC = 0.30      # 이보다 짧은 쉼은 한 덩어리로 본다
KEEP_REL = 0.2            # 가장 큰 덩어리보다 −14 dB 넘게 작은 덩어리는 버린다(숨·키보드 소리. 11-10 녹음 실측)


def trim_speech(x: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """앞뒤 조용한 부분을 잘라 말소리 + 앞뒤 PAD_SEC 만 남긴다. 말이 없으면 빈 배열.

    소리 나는 창을 MERGE_GAP_SEC 보다 짧은 쉼으로 이어 덩어리로 묶고, 가장 큰 덩어리보다 KEEP_REL 배 넘게
    작은 덩어리(말 뒤 숨·키보드 소리)는 버린다 — 첫 덩어리부터 끝 덩어리까지를 남긴다.
    """
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

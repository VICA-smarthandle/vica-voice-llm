#!/usr/bin/env python3
"""청각 안내에 쓸 음성 파일을 미리 합성한다 (TTS 가 있는 기기에서 한 번 실행).

호출 응답은 빠를수록 좋다 — 사용자가 "들었나?" 하고 기다리는 순간이다. TTS 큐를
거치면 대기와 합성 시간이 붙으므로, 자주 쓰는 짧은 말은 파일로 만들어 둔다.

실행:
    .venv/bin/python scripts/make_cue_wavs.py

만들어진 파일은 assets/ 에 저장되고 ros_wakeword_node 가 자동으로 찾는다.
파일이 없으면 노드는 TTS 로 대체하므로 동작은 하되 조금 늦다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ASSETS = Path(__file__).resolve().parents[1] / "assets"

# 파일 이름 -> 문구. 정본은 src/ment_cache.py 의 CACHED_MENTS 하나다 —
# ros_tts_node 가 같은 표로 재생하므로 여기서 따로 정의하면 캐시가 빗나간다.


def main() -> int:
    import soundfile as sf

    from src.ment_cache import CACHED_MENTS
    from src.tts import VicaTTS

    ASSETS.mkdir(exist_ok=True)
    tts = VicaTTS()

    for filename, text in CACHED_MENTS.items():
        wav, sample_rate = tts._synthesize(text)
        out = ASSETS / filename
        sf.write(str(out), wav, sample_rate)
        print(f"만듦: {out}  ({text!r}, {len(wav) / sample_rate:.2f}초)")

    print("\n노드를 다시 띄우면 이 파일을 씁니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

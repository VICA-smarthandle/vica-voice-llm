#!/usr/bin/env python3
"""청각 안내에 쓸 음성 파일을 미리 합성한다 (TTS 가 있는 기기에서 한 번 실행)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ASSETS = Path(__file__).resolve().parents[1] / "assets"


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

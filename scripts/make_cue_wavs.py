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

# 파일 이름 -> 합성할 문구. replies.py 의 상수를 그대로 쓴다(문구가 갈라지지 않게).
TARGETS = {"wake_greeting.wav": "WAKE_GREETING"}


def main() -> int:
    import soundfile as sf

    from src import replies
    from src.tts import VicaTTS

    ASSETS.mkdir(exist_ok=True)
    tts = VicaTTS()

    for filename, const_name in TARGETS.items():
        text = getattr(replies, const_name)
        wav, sample_rate = tts._synthesize(text)
        out = ASSETS / filename
        sf.write(str(out), wav, sample_rate)
        print(f"만듦: {out}  ({text!r}, {len(wav) / sample_rate:.2f}초)")

    print("\n노드를 다시 띄우면 이 파일을 씁니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

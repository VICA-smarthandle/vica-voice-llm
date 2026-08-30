"""고정 멘트 녹음 캐시 — 합성 대신 구워 둔 wav 를 즉시 재생하기 위한 대응표.

고정 멘트는 매번 합성(GPU 0.7~0.8초)할 이유가 없다 (2026-08-25 사용자 결정).
ros_tts_node 는 요청 문장이 이 표에 있으면 합성을 건너뛰고 녹음을 재생한다 —
문장이 조금이라도 다르면 캐시가 빗나가고 합성으로 자연 폴백하므로 안전하다.

정본은 CACHED_MENTS 하나다. scripts/make_cue_wavs.py 가 같은 표를 굽고,
tests/test_ment_cache.py 가 시나리오 멘트의 등록 누락을 잡는다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import replies

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

# 파일 이름 -> 문구. 문구를 바꾸면 make_cue_wavs.py 로 다시 굽는다.
CACHED_MENTS: dict[str, str] = {
    "wake_greeting.wav": replies.WAKE_GREETING,
    "approach_question.wav": replies.APPROACH_QUESTION,
    "approach_onboarding.wav": replies.APPROACH_ONBOARDING,
    "approach_turn_notice.wav": replies.APPROACH_TURN_NOTICE,
    "approach_turn_done.wav": replies.APPROACH_TURN_DONE,
    "approach_farewell.wav": replies.APPROACH_FAREWELL,
}


class MentCache:
    """assets 의 녹음을 읽어 문장으로 찾는다. 없는 파일은 건너뛴다(합성 폴백)."""

    def __init__(self, assets_dir: Path | str = ASSETS_DIR) -> None:
        self._by_text: dict[str, tuple] = {}
        self.missing: list[str] = []
        assets = Path(assets_dir)
        for filename, text in CACHED_MENTS.items():
            path = assets / filename
            if not path.exists():
                self.missing.append(filename)
                continue
            try:
                import soundfile as sf

                wav, rate = sf.read(str(path), dtype="float32")
                self._by_text[text.strip()] = (wav, int(rate))
            except Exception:
                # 깨진 파일 하나가 노드 기동을 막으면 안 된다 — 합성 폴백.
                self.missing.append(filename)
        # 구운 멘트 일괄 적재 (assets/baked/manifest.json — 2026-08-30 결정:
        # 전 멘트를 CosyVoice F2 클로닝으로 굽는다). 같은 문구면 구운 판이
        # 이긴다(목소리 통일). manifest 가 없으면 조용히 넘어간다.
        baked = assets / "baked"
        manifest = baked / "manifest.json"
        if manifest.exists():
            try:
                import json

                import soundfile as sf
                for filename, text in json.load(open(manifest)).items():
                    path = baked / filename
                    if not path.exists():
                        self.missing.append(f"baked/{filename}")
                        continue
                    try:
                        wav, rate = sf.read(str(path), dtype="float32")
                        self._by_text[text.strip()] = (wav, int(rate))
                    except Exception:
                        self.missing.append(f"baked/{filename}")
            except Exception:
                pass   # manifest 가 깨져도 기동은 계속

    def lookup(self, text: str) -> Optional[tuple]:
        """문장이 구워져 있으면 (wav, sample_rate), 아니면 None."""
        if not text:
            return None
        return self._by_text.get(text.strip())

    def __len__(self) -> int:
        return len(self._by_text)

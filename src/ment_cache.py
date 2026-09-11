"""고정 멘트 녹음 캐시 — 합성 대신 구워 둔 wav 를 즉시 재생하기 위한 대응표."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import replies

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

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
                self.missing.append(filename)
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
                pass

    def lookup(self, text: str) -> Optional[tuple]:
        """문장이 구워져 있으면 (wav, sample_rate), 아니면 None."""
        if not text:
            return None
        return self._by_text.get(text.strip())

    def __len__(self) -> int:
        return len(self._by_text)

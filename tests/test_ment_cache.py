"""고정 멘트 녹음 캐시 시험 (소리 장치 없이).

고정 멘트는 매번 합성(0.7~0.8초)하지 않고 구워 둔 녹음을 즉시 재생한다
(2026-08-25 사용자 결정). 정본 대응표는 ment_cache.CACHED_MENTS 하나이며
scripts/make_cue_wavs.py 도 같은 표를 굽는다 — 표가 갈라지면 캐시가 빗나간다.
"""
import numpy as np
import pytest
import soundfile as sf

from src import replies
from src.ment_cache import ASSETS_DIR, CACHED_MENTS, MentCache


class TestRegistry:
    def test_mapping_is_filename_to_nonempty_text(self):
        assert CACHED_MENTS, "등록된 멘트가 하나는 있어야 한다"
        for filename, text in CACHED_MENTS.items():
            assert filename.endswith(".wav")
            assert isinstance(text, str) and text.strip()

    @pytest.mark.parametrize("const", [
        "WAKE_GREETING",
        "APPROACH_QUESTION",
        "APPROACH_ONBOARDING",
        "APPROACH_TURN_NOTICE",
        "APPROACH_TURN_DONE",
        "APPROACH_FAREWELL",
    ])
    def test_scenario_ments_are_registered(self, const):
        text = getattr(replies, const)
        assert text in CACHED_MENTS.values()


class TestLookup:
    @pytest.fixture
    def cache(self, tmp_path, monkeypatch):
        """임시 assets 에 등록 멘트 중 하나만 구워 둔 캐시."""
        filename = "approach_turn_done.wav"
        assert filename in CACHED_MENTS
        sf.write(str(tmp_path / filename),
                 np.zeros(1600, dtype=np.float32), 16000, subtype="PCM_16")
        return MentCache(tmp_path)

    def test_hit_returns_audio(self, cache):
        got = cache.lookup(replies.APPROACH_TURN_DONE)
        assert got is not None
        wav, rate = got
        assert rate == 16000 and len(wav) == 1600

    def test_strip_tolerance(self, cache):
        assert cache.lookup("  " + replies.APPROACH_TURN_DONE + " \n") is not None

    def test_miss_returns_none(self, cache):
        assert cache.lookup("등록되지 않은 문장입니다") is None
        assert cache.lookup("") is None

    def test_missing_files_are_skipped_not_fatal(self, cache):
        # 등록 멘트 대부분의 파일이 없는 임시 폴더 — 로드는 성공해야 한다.
        assert cache.lookup(replies.APPROACH_QUESTION) is None
        assert len(cache.missing) >= 1  # 없는 파일 목록을 노출한다 (기동 로그용)


class TestRealAssets:
    def test_committed_greeting_loads_from_real_assets(self):
        """저장소에 커밋된 wav 는 실제로 읽혀야 한다 (배선 스모크)."""
        cache = MentCache(ASSETS_DIR)
        assert cache.lookup(replies.WAKE_GREETING) is not None

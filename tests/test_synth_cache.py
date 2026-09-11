"""합성 캐시 시험 — 같은 문장을 두 번 합성하지 않는다."""
import numpy as np

from src.synth_cache import SynthCache


def _wav(n=16):
    return np.zeros(n, dtype=np.float32), 24000


class TestSynthCache:
    def test_miss_then_hit(self):
        c = SynthCache()
        assert c.get("안내소로 안내해드릴까요?") is None
        c.put("안내소로 안내해드릴까요?", *_wav())
        got = c.get("안내소로 안내해드릴까요?")
        assert got is not None and got[1] == 24000

    def test_capacity_evicts_least_recently_used(self):
        c = SynthCache(capacity=2)
        c.put("가", *_wav())
        c.put("나", *_wav())
        c.get("가")
        c.put("다", *_wav())
        assert c.get("가") is not None
        assert c.get("나") is None
        assert c.get("다") is not None

    def test_put_same_text_updates_not_duplicates(self):
        c = SynthCache(capacity=2)
        c.put("가", *_wav(8))
        c.put("가", *_wav(32))
        assert len(c) == 1
        assert len(c.get("가")[0]) == 32

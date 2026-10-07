"""합성 캐시 시험 — 같은 문장을 두 번 합성하지 않는다.

확인 질문("~로 안내해드릴까요?")은 목적지별 고정 문장인데 매번 합성해
응답이 늦었다 (2026-08-28 실측: 대기~재생까지 0.9초대). 합성 결과를
문장 단위로 저장하고, 기동 시 미리 데워 첫 사용부터 0초로 만든다.
"""
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
        c.get("가")                       # '가'를 최근 사용으로 갱신
        c.put("다", *_wav())              # 용량 초과 -> '나'가 밀린다
        assert c.get("가") is not None
        assert c.get("나") is None
        assert c.get("다") is not None

    def test_put_same_text_updates_not_duplicates(self):
        c = SynthCache(capacity=2)
        c.put("가", *_wav(8))
        c.put("가", *_wav(32))
        assert len(c) == 1
        assert len(c.get("가")[0]) == 32


class TestPinned:
    """미리 합성한 문장은 고정 — 주행 중 LLM 대답들에 밀려 첫 도착 전에 지워지지 않는다
    (2026-10-07, 도착 멘트·M1·질문이 늘어 64문장을 넘는다)."""

    def test_pinned_survives_eviction_and_does_not_count_toward_capacity(self):
        c = SynthCache(capacity=2)
        c.put("화장실에 도착했습니다.", *_wav(), pinned=True)
        c.put("가", *_wav())
        c.put("나", *_wav())
        c.put("다", *_wav())                       # '가'가 밀린다 — 고정 문장은 아니다
        assert c.get("화장실에 도착했습니다.") is not None
        assert c.get("가") is None
        assert c.get("나") is not None and c.get("다") is not None
        assert len(c) == 3

    def test_unpinned_put_does_not_replace_a_pinned_sentence(self):
        c = SynthCache(capacity=2)
        c.put("가", *_wav(8), pinned=True)
        c.put("가", *_wav(32))
        assert len(c.get("가")[0]) == 8
        assert len(c) == 1

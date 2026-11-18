"""화자 임베딩 감싸개 — 모델이 쓸 수 없는 벡터를 내면 None 으로 바꾸는지(실모델 없이 가짜 추출기로).

11-11 실측: sherpa ERes2Net 에 0.05 s 무음을 넣으면 is_ready 가 참인데 NaN 192개를 돌려줬다.
"""
from __future__ import annotations

import numpy as np

from src.spk_embed import cosine, mean_embedding, wrap_extractor


class FakeStream:
    def accept_waveform(self, sample_rate, waveform):
        self.n = len(waveform)

    def input_finished(self):
        pass


class FakeExtractor:
    def __init__(self, out, ready=True):
        self.out, self.ready = out, ready

    def create_stream(self):
        return FakeStream()

    def is_ready(self, stream):
        return self.ready

    def compute(self, stream):
        return list(self.out)


def test_nan_vector_becomes_none():
    embed = wrap_extractor(FakeExtractor([float("nan")] * 4))
    assert embed(np.zeros(800, np.float32)) is None


def test_zero_vector_becomes_none():
    assert wrap_extractor(FakeExtractor([0.0] * 4))(np.zeros(16000, np.float32)) is None


def test_not_ready_becomes_none():
    assert wrap_extractor(FakeExtractor([1.0, 0.0], ready=False))(np.zeros(16000, np.float32)) is None


def test_good_vector_passes_through():
    v = wrap_extractor(FakeExtractor([3.0, 4.0]))(np.zeros(16000, np.float32))
    assert v.tolist() == [3.0, 4.0]


def test_cosine_and_mean():
    assert abs(cosine(np.array([1.0, 0.0]), np.array([1.0, 1.0])) - 0.70711) < 1e-4
    assert mean_embedding([np.array([2.0, 0.0]), np.array([0.0, 1.0])]).round(4).tolist() == [0.7071, 0.7071]

"""화자 임베딩 — sherpa-onnx 화자 모델로 목소리 벡터를 뽑는다(CPU)."""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

SAMPLE_RATE = 16000


def load_embedder(model_path: str, threads: int = 2) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    """float32(−1~1) 모노 16 kHz 소리 → 벡터. 너무 짧아 모델이 준비되지 않으면 None."""
    import sherpa_onnx

    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model_path), num_threads=threads,
                                                      debug=False, provider="cpu")
    if not cfg.validate():
        raise ValueError(f"화자 모델 설정이 맞지 않습니다: {model_path}")
    return wrap_extractor(sherpa_onnx.SpeakerEmbeddingExtractor(cfg))


def wrap_extractor(extractor) -> Callable[[np.ndarray], Optional[np.ndarray]]:  # noqa: ANN001
    """sherpa 추출기 → 판정 함수. 모델이 준비되지 않았거나 쓸 수 없는 벡터(NaN·0)면 None."""
    def embed(audio: np.ndarray) -> Optional[np.ndarray]:
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=np.ascontiguousarray(audio, dtype=np.float32))
        stream.input_finished()
        if not extractor.is_ready(stream):
            return None
        v = np.asarray(extractor.compute(stream), dtype=np.float64)
        if v.size == 0 or not np.all(np.isfinite(v)) or float(np.linalg.norm(v)) == 0.0:
            return None
        return v

    return embed


def mean_embedding(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """벡터마다 단위 길이로 맞춘 뒤 평균하고, 평균도 단위 길이로."""
    units = [np.asarray(v, dtype=np.float64) / np.linalg.norm(v) for v in vectors]
    m = np.mean(units, axis=0)
    return m / np.linalg.norm(m)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

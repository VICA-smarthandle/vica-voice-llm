import ctypes
import numpy as np
import pytest
from src.echo_cancel import BLOCK, EchoCancelUnavailable, EchoCanceller

SR = 16000

def _has_lib() -> bool:
    try:
        ctypes.CDLL("libspeexdsp.so.1")
        return True
    except OSError:
        return False

needs_lib = pytest.mark.skipif(not _has_lib(), reason="libspeexdsp.so.1 없음")

def _noise(sec, seed, amp=6000):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(sec * SR)) * amp).clip(-30000, 30000).astype(np.int16)

def _echo_of(ref, delay_ms=118, gain=0.3):          # 118 ms 늦은 짧은 방 응답(32 ms)
    h = np.zeros(512); h[0], h[40], h[200] = gain, gain * 0.5, gain * 0.25
    y = np.convolve(np.concatenate([np.zeros(delay_ms * 16), ref.astype(np.float64)]), h)[:len(ref)]
    return y.astype(np.int16)

def _run(ec, mic, ref, block=BLOCK):
    out = np.zeros_like(mic)
    for i in range(0, len(mic) - block + 1, block):
        out[i:i + block] = ec.process(mic[i:i + block], ref[i:i + block])
    return out

def _db(a, b):
    r = lambda v: np.sqrt(np.mean(v.astype(np.float64) ** 2)) + 1e-9
    return 20 * np.log10(r(a) / r(b))

@needs_lib
def test_removes_echo_delayed_118ms():
    ref = _noise(8, 1); mic = _echo_of(ref)
    out = _run(EchoCanceller(), mic, ref)
    assert _db(out[5 * SR:], mic[5 * SR:]) <= -6.0           # 설계 5.1 — 5 s 뒤 6 dB 이상

@needs_lib
def test_reference_delay_is_applied():                       # 64 ms 필터는 기준을 100 ms 늦춰야 118 ms 에코에 닿는다
    ref = _noise(8, 5); mic = _echo_of(ref)
    late = _run(EchoCanceller(delay_ms=100, tail_ms=64), mic, ref)
    raw = _run(EchoCanceller(delay_ms=0, tail_ms=64), mic, ref)
    assert _db(late[5 * SR:], mic[5 * SR:]) <= -6.0
    assert _db(raw[5 * SR:], mic[5 * SR:]) > -1.0

@needs_lib
def test_quiet_reference_keeps_input_within_1db_for_20s():   # Review Focus 2
    mic = _noise(20, 2, amp=3000)
    out = _run(EchoCanceller(), mic, np.zeros_like(mic))
    assert abs(_db(out, mic)) <= 1.0
    assert abs(_db(out[-SR:], mic[-SR:])) <= 1.0

@needs_lib
def test_block_size_does_not_change_result():               # 지연 버퍼가 블록 경계를 넘어 이어진다
    ref = _noise(2, 3); mic = _echo_of(ref)
    assert np.array_equal(_run(EchoCanceller(), mic, ref, 1280), _run(EchoCanceller(), mic, ref, 160))

@needs_lib
def test_full_scale_and_non_contiguous_input():             # Review Focus 3
    pair = np.stack([np.full(BLOCK, 32767, np.int16), np.full(BLOCK, -32768, np.int16)], axis=1)
    out = EchoCanceller().process(pair[:, 0], pair[:, 1])
    assert out.dtype == np.int16 and out.shape == (BLOCK,)

@needs_lib
def test_rejects_bad_lengths():
    ec = EchoCanceller()
    with pytest.raises(ValueError):
        ec.process(np.zeros(1000, np.int16), np.zeros(1000, np.int16))
    with pytest.raises(ValueError):
        ec.process(np.zeros(1280, np.int16), np.zeros(160, np.int16))

def test_missing_library_raises_unavailable():
    with pytest.raises(EchoCancelUnavailable):
        EchoCanceller(lib_name="libnope-speexdsp.so.9")

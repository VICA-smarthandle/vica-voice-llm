import numpy as np
import pytest
from src.echo_cancel import (BLOCK, EchoCancelUnavailable, EchoConfig, EchoStage, build_echo_stage,
                             clip_info_with_aec, echo_config_from_env)

class FakeEc:
    made = 0
    def __init__(self, delay_ms=100, tail_ms=256, **_):
        FakeEc.made += 1; self.calls = 0; self.boom = False; self.args = (delay_ms, tail_ms)
    def process(self, mic, ref):
        self.calls += 1
        if self.boom: raise RuntimeError("x")
        return (mic // 2).astype(np.int16)
    def close(self): pass

@pytest.fixture(autouse=True)
def _reset(): FakeEc.made = 0

CH0 = np.full(BLOCK, 1000, np.int16); CH5 = np.full(BLOCK, 3000, np.int16)

@pytest.mark.parametrize("raw,mode,warned", [(None, "off", False), ("", "off", False), ("on", "on", False),
    ("ON", "on", False), (" log ", "log", False), ("1", "off", True), ("true", "off", True)])
def test_mode_from_env(raw, mode, warned):                 # Review Focus 4
    cfg = echo_config_from_env({} if raw is None else {"VICA_SW_AEC": raw})
    assert cfg.mode == mode and bool(cfg.warnings) == warned

def test_numbers_from_env():
    cfg = echo_config_from_env({"VICA_SW_AEC": "on", "VICA_SW_AEC_DELAY_MS": "80", "VICA_SW_AEC_TAIL_MS": "128"})
    assert (cfg.delay_ms, cfg.tail_ms, cfg.warnings) == (80, 128, ())
    bad = echo_config_from_env({"VICA_SW_AEC": "on", "VICA_SW_AEC_DELAY_MS": "abc", "VICA_SW_AEC_TAIL_MS": "9999"})
    assert (bad.delay_ms, bad.tail_ms) == (100, 256) and len(bad.warnings) == 2

def test_on_returns_processed_and_log_returns_same_object():
    on = EchoStage(EchoConfig("on"), lambda s: None, factory=FakeEc)
    assert np.array_equal(on.feed(CH0, CH5), CH0 // 2)
    log = EchoStage(EchoConfig("log"), lambda s: None, factory=FakeEc)
    assert log.feed(CH0, CH5) is CH0 and log.mode == "log"

def test_exception_mid_speech_falls_back_for_good():      # Review Focus 5
    warns = []
    st = EchoStage(EchoConfig("on"), warns.append, factory=FakeEc)
    st.feed(CH0, CH5); ec = st._ec; ec.boom = True
    assert st.feed(CH0, CH5) is CH0 and st.mode == "off" and len(warns) == 1
    ec.boom = False
    assert st.feed(CH0, CH5) is CH0 and ec.calls == 2 and len(warns) == 1   # 다시 부르지 않는다

def test_short_block_passes_through_and_rebuilds_state():
    st = EchoStage(EchoConfig("on"), lambda s: None, factory=FakeEc)
    short = np.zeros(640, np.int16)
    assert st.feed(short, short) is short and FakeEc.made == 2 and st.mode == "on"
    assert np.array_equal(st.feed(CH0, CH5), CH0 // 2)

def test_unavailable_library_means_no_stage():
    def nope(**_): raise EchoCancelUnavailable("없음")
    infos, warns = [], []
    assert build_echo_stage({"VICA_SW_AEC": "on"}, infos.append, warns.append, log_dir=None, factory=nope) is None
    assert infos == [] and len(warns) == 1

def test_build_off_never_touches_the_library(tmp_path):
    assert build_echo_stage({}, lambda s: None, lambda s: None, log_dir=tmp_path, factory=FakeEc) is None
    assert FakeEc.made == 0

def test_build_log_announces_and_passes_numbers(tmp_path):
    infos = []
    st = build_echo_stage({"VICA_SW_AEC": "log", "VICA_SW_AEC_DELAY_MS": "90"}, infos.append, lambda s: None,
                          log_dir=tmp_path, factory=FakeEc)
    assert st.mode == "log" and st._ec.args == (90, 256)
    assert infos == ["소프트웨어 에코 제거: log (기준 늦춤 90 ms · 필터 256 ms)"]

def test_clip_info():
    info = {"result": "wake"}
    assert clip_info_with_aec(info, "off") is info
    assert clip_info_with_aec(info, "on") == {"result": "wake", "aec": "on"} and info == {"result": "wake"}

def test_off_config_never_calls_factory():
    st = EchoStage(EchoConfig("off"), lambda s: None, factory=FakeEc)
    assert FakeEc.made == 0 and st.mode == "off" and st.feed(CH0, CH5) is CH0

def test_meter_failure_falls_back_too():
    class BadMeter:
        def update(self, mic, out, ref): raise RuntimeError("meter")
    warns = []
    st = EchoStage(EchoConfig("on"), warns.append, meter=BadMeter(), factory=FakeEc)
    ec = st._ec
    assert st.feed(CH0, CH5) is CH0 and st.mode == "off" and len(warns) == 1
    assert st.feed(CH0, CH5) is CH0 and ec.calls == 1 and len(warns) == 1

def test_raising_warn_callback_never_escapes():
    def bad_warn(s): raise RuntimeError("warn")
    st = EchoStage(EchoConfig("on"), bad_warn, factory=FakeEc)
    st._ec.boom = True
    assert st.feed(CH0, CH5) is CH0 and st.mode == "off"

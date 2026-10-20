"""마이크 주변 대화 시험기(tools/mic_bystander_check.py)의 순수 로직 — 칩 설정 묶음·요약 계산."""
import math

from tools.mic_bystander_check import PRESETS, db, resolve, summarize


def _preset(key):
    return next(p for p in PRESETS if p.key == key)


def test_base_preset_changes_nothing():
    assert resolve(_preset("base"), {"GAMMAVAD_SR": 15.0}) == {}


def test_vad_preset_scales_the_baseline():
    out = resolve(_preset("vad3"), {"GAMMAVAD_SR": 15.0})
    assert math.isclose(out["GAMMAVAD_SR"], 45.0)


def test_agc_off_sets_fixed_gain():
    out = resolve(_preset("agcoff"), {})
    assert out["AGCONOFF"] == 0 and out["AGCGAIN"] > 0


def test_db():
    assert math.isclose(db(1.0), 0.0)
    assert math.isclose(db(0.1), -20.0)
    assert db(0.0) < -150


def _rec(preset, role, rms, phrase="", wake=0.0, stop=0.0):
    return {"preset": preset, "role": role, "phrase": phrase, "rms": rms, "wake_a": wake, "stop_b": stop}


def test_separable_preset_has_positive_margin_and_a_threshold():
    recs = [_rec("gain2", "user", 0.05), _rec("gain2", "user", 0.04),
            _rec("gain2", "near1", 0.01), _rec("gain2", "near2", 0.005)]
    s = summarize(recs, listen_min_rms=0.008, wake_th=0.7, stop_th=0.5)["presets"][0]
    assert s["key"] == "gain2"
    assert s["margin_db"] == round(db(0.04) - db(0.01), 1)
    assert 0.01 < s["suggest_rms"] < 0.04
    assert s["bystander_pass_now"] == "1/2"
    assert s["user_drop_now"] == "0/2"


def test_overlapping_preset_has_negative_margin_and_no_threshold():
    recs = [_rec("base", "user", 0.02), _rec("base", "near2", 0.03)]
    s = summarize(recs, listen_min_rms=0.008, wake_th=0.7, stop_th=0.5)["presets"][0]
    assert s["margin_db"] < 0 and s["suggest_rms"] is None


def test_wake_and_stop_hits_count_only_their_phrases():
    recs = [_rec("base", "user", 0.05, "비카야", wake=0.9), _rec("base", "user", 0.05, "비카야", wake=0.3),
            _rec("base", "user", 0.05, "멈춰", stop=0.6), _rec("base", "user", 0.05, "네", wake=0.95, stop=0.9)]
    s = summarize(recs, listen_min_rms=0.008, wake_th=0.7, stop_th=0.5)["presets"][0]
    assert s["wake"] == "1/2" and s["stop"] == "1/1"


def test_presets_listed_in_try_order_and_missing_ones_skipped():
    recs = [_rec("vad3", "user", 0.05), _rec("base", "user", 0.05)]
    keys = [p["key"] for p in summarize(recs, 0.008, 0.7, 0.5)["presets"]]
    assert keys == ["base", "vad3"]


def test_far_shouts_are_counted_apart_from_bystander_chat():
    """멀리서 외친 '멈춰' 는 버릴 잡담이 아니라 잡혀야 하는 소리 — 여유 계산에 넣지 않는다."""
    recs = [_rec("gain2", "user", 0.05), _rec("gain2", "near2", 0.005),
            _rec("gain2", "shout3", 0.03, "멈춰", stop=0.8), _rec("gain2", "shout2", 0.04, "멈춰", stop=0.2)]
    s = summarize(recs, 0.008, 0.7, 0.5)["presets"][0]
    assert s["margin_db"] == round(db(0.05) - db(0.005), 1)
    assert s["far_stop"] == "1/2"
    assert set(s["by_dist"]) == {"near2"}


def test_far_users_must_be_kept_too():
    """접근 단계 사용자는 1 m 넘게 떨어져 대답한다 — 여유는 손잡이·앞쪽 사용자 중 작은 쪽 기준(10-15 사용자)."""
    recs = [_rec("gain2", "user", 0.05), _rec("gain2", "far1", 0.012, "그래"), _rec("gain2", "far15", 0.009, "아니"),
            _rec("gain2", "near1", 0.006)]
    s = summarize(recs, 0.01, 0.7, 0.5)["presets"][0]
    assert s["margin_db"] == round(db(0.009) - db(0.006), 1)
    assert s["far"]["n"] == 2 and s["far"]["min_db"] == round(db(0.009), 1)
    assert s["far_drop_now"] == "1/2"
    assert s["user_drop_now"] == "0/1"


def test_wake_sentence_and_two_metre_shouts_are_reported():
    recs = [_rec("gain4", "userwake", 0.05, "비카야, 화장실 가자", wake=0.8),
            _rec("gain4", "userwake", 0.05, "비카야, 화장실 가자", wake=0.4),
            _rec("gain4", "shout2", 0.03, "멈춰", stop=0.6), _rec("gain4", "shout2", 0.03, "멈춰", stop=0.35),
            _rec("gain4", "shout3", 0.02, "멈춰", stop=0.1)]
    s = summarize(recs, 0.012, 0.7, 0.5)["presets"][0]
    assert s["key"] == "gain4"
    assert s["wake_cont"] == "1/2" and s["wake"] == "0/0"
    assert s["shout2"] == "1/2" and s["shout2_low"] == "2/2"
    assert s["user"]["n"] == 2

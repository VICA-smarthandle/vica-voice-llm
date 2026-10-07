"""사람 접근 차임 (2026-10-07 사용자 결정: 첫 인사 + 2초마다 '딩—동↗', 차임 1번).

시각장애인에게 다가가는 동안(미션 dialog_state == "approaching")만 울린다. 후진음은
'피하라'로 들려 쓰지 않는다. 소리 장치 없이 파형과 울릴 때를 판정하는 규칙만 본다.
"""
import numpy as np
import pytest

from src import audio_cue, audio_out
from src.approach_chime import ApproachChime


# ---- 소리 ---------------------------------------------------------------------
def _peak_freq(segment: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
    freqs = np.fft.rfftfreq(len(segment), 1.0 / audio_cue.SAMPLE_RATE)
    return float(freqs[int(np.argmax(spectrum))])


def test_chime_is_two_rising_bell_notes():
    wave = audio_cue.approach_chime()
    sr = audio_cue.SAMPLE_RATE
    assert wave.dtype == np.float32
    assert len(wave) == pytest.approx((audio_cue.APPROACH_CHIME_GAP_SEC
                                       + audio_cue.APPROACH_CHIME_NOTE_SEC) * sr, abs=2)
    first = wave[: int(0.2 * sr)]                                   # 첫 음만 울리는 구간
    second = wave[int(0.30 * sr): int(0.50 * sr)]                   # 둘째 음이 막 친 구간
    assert _peak_freq(first) == pytest.approx(880.0, abs=15)
    assert _peak_freq(second) == pytest.approx(1174.66, abs=15)


def test_chime_rings_out_instead_of_cutting_off():
    wave = audio_cue.approach_chime()
    sr = audio_cue.SAMPLE_RATE
    head = np.max(np.abs(wave[: int(0.05 * sr)]))
    tail = np.max(np.abs(wave[-int(0.05 * sr):]))
    assert head > 0.3
    assert tail < 0.05 * head                                       # 여운으로 잦아든다
    assert abs(float(wave[0])) < 0.05                               # 딸깍 없이 시작


def test_chime_avoids_the_thinking_notes():
    """'생각 중' 운율(도·미·솔)과 헷갈리지 않게 그 음을 쓰지 않는다."""
    for f in audio_cue.APPROACH_CHIME_NOTES_HZ:
        for t in audio_cue.THINKING_NOTES_HZ:
            assert abs(f - t) > 30


# ---- 크기: 차임은 말보다 작게 ----------------------------------------------------
def test_prepare_can_use_a_quieter_peak():
    wave = np.full(100, 0.5, dtype=np.float32)
    out = audio_out.prepare(wave, 16000, None, 1, peak_dbfs=-12.0)
    assert np.max(np.abs(out)) == pytest.approx(10 ** (-12 / 20), rel=1e-4)


def test_prepare_default_peak_is_unchanged():
    wave = np.full(100, 0.5, dtype=np.float32)
    out = audio_out.prepare(wave, 16000, None, 1)
    assert np.max(np.abs(out)) == pytest.approx(10 ** (audio_out.target_peak_dbfs() / 20), rel=1e-4)


# ---- 울릴 때 ------------------------------------------------------------------
def chime(**kw):
    return ApproachChime(period_sec=2.0, stale_sec=3.0, after_speech_sec=1.0, **kw)


def test_rings_only_while_approaching():
    c = chime()
    assert not c.due(0.0)
    c.on_robot_state("idle", 0.0)
    assert not c.due(0.5)
    c.on_robot_state("approaching", 1.0)
    assert c.due(1.0)
    c.on_robot_state("awaiting_user", 1.5)        # 곁에 도착 — 질문 차례
    assert not c.due(1.6)


def test_rings_every_period():
    c = chime()
    c.on_robot_state("approaching", 0.0)
    assert c.due(0.0)
    c.played(0.0)
    assert not c.due(1.9)
    c.on_robot_state("approaching", 1.0)          # 1초마다 오는 상태 알림은 시계를 안 건드린다
    assert not c.due(1.9)
    assert c.due(2.0)


def test_waits_a_breath_after_speech():
    """첫 인사 "동행로봇 비카가 다가가고 있어요"가 끝나자마자 겹쳐 울리지 않는다."""
    c = chime()
    c.on_robot_state("approaching", 0.0)
    c.on_speech_end(2.5)
    c.on_robot_state("approaching", 3.0)          # 상태 알림은 1초마다 계속 온다
    assert not c.due(3.0)
    assert c.due(3.5)


def test_stops_when_state_reports_go_stale():
    """미션이 죽어 상태 알림이 끊기면 저절로 멈춘다."""
    c = chime()
    c.on_robot_state("approaching", 0.0)
    c.played(0.0)
    assert c.due(2.0)
    assert not c.due(3.1)


def test_new_approach_starts_fresh():
    c = chime()
    c.on_robot_state("approaching", 0.0)
    c.played(0.0)
    c.on_robot_state("idle", 0.5)
    c.on_robot_state("approaching", 0.7)          # 다른 사람에게 새로 다가간다
    assert c.due(0.7)


def test_switch_off_never_rings():
    c = chime(enabled=False)
    c.on_robot_state("approaching", 0.0)
    assert not c.due(0.0)

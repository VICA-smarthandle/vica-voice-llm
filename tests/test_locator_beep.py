"""위치 비프 (2026-10-23 사용자 결정: C3M21, 잡기 요청 멘트 뒤·비상정지 멘트 뒤에만).

시각장애인이 로봇 위치를 소리로 찾게 4초 한 번 울린다. 잡음 판정(진동 반복이 멈추는
순간)·해제로 대화 단계가 바뀌면 끊는다. 소리 장치 없이 파형과 울릴 때 규칙만 본다.
"""
import numpy as np
import pytest

from src import audio_cue
from src.locator_beep import LocatorBeep

SR = audio_cue.SAMPLE_RATE


# ---- 소리 ---------------------------------------------------------------------
def _onsets(wave: np.ndarray) -> list[float]:
    frame = int(0.005 * SR)
    env = np.array([np.max(np.abs(wave[i:i + frame])) for i in range(0, len(wave), frame)])
    on = env > 0.05
    return [i * frame / SR for i in range(1, len(on)) if on[i] and not on[i - 1]] + (
        [0.0] if on[0] else [])


def _peak_freq(segment: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
    freqs = np.fft.rfftfreq(len(segment), 1.0 / SR)
    return float(freqs[int(np.argmax(spectrum))])


def test_beep_is_eleven_pulses_every_375ms_in_four_seconds():
    wave = audio_cue.locator_beep()
    assert wave.dtype == np.float32
    assert len(wave) == int(4.0 * SR)
    assert np.max(np.abs(wave)) == pytest.approx(1.0, abs=1e-6)
    starts = sorted(_onsets(wave))
    assert len(starts) == 11
    for i, s in enumerate(starts):
        assert s == pytest.approx(i * 0.375, abs=0.006)


def test_each_pulse_rises_from_low_to_high():
    """0.25→0.75 kHz 로 올라간다 — 1차의 1 kHz 비프는 멘트보다 높게 들려 내렸다."""
    pulse = audio_cue.locator_beep()[: int(0.21 * SR)]
    early = pulse[int(0.01 * SR): int(0.05 * SR)]
    late = pulse[int(0.15 * SR): int(0.19 * SR)]
    assert 250 <= _peak_freq(early) <= 360
    assert 520 <= _peak_freq(late) <= 760


def test_pulses_sound_for_210ms_then_rest():
    wave = audio_cue.locator_beep()
    assert np.max(np.abs(wave[int(0.10 * SR): int(0.20 * SR)])) > 0.3
    assert np.max(np.abs(wave[int(0.215 * SR): int(0.37 * SR)])) < 0.01


def test_energy_stays_in_the_low_band():
    """기음 ≤0.75 kHz·3배음 ≤2.25 kHz — 딸깍을 빼면 그 위는 거의 없다."""
    wave = audio_cue.locator_beep().astype(np.float64)
    power = np.abs(np.fft.rfft(wave)) ** 2
    freqs = np.fft.rfftfreq(len(wave), 1.0 / SR)
    assert power[freqs <= 2500].sum() / power.sum() > 0.95


def test_level_is_close_to_speech_not_louder():
    """최고점 -15 dBFS 에서 RMS 가 멘트(약 -20 dBFS)와 비슷하다 — 같은 최고점이면 13~16 dB 컸다."""
    wave = audio_cue.locator_beep().astype(np.float64) * 10 ** (audio_cue.LOCATOR_DBFS / 20)
    rms_dbfs = 20 * np.log10(np.sqrt(np.mean(wave ** 2)))
    assert -22.0 <= rms_dbfs <= -18.0


# ---- 울릴 때: 손잡이 잡기 요청 ------------------------------------------------------
def beep():
    return LocatorBeep(after_speech_sec=0.5, stale_sec=3.0)


def alive(b, state, *times):
    """미션이 1초마다 보내는 같은 단계 알림."""
    for t in times:
        b.on_robot_state(state, t)


def test_grip_wait_rings_once_after_the_hint_ends():
    b = beep()
    b.on_robot_state("grip_wait", 0.0)          # 회전 끝 — 손잡이 멘트 시작
    assert not b.due(1.0)                        # 멘트 중
    b.on_speech_end("response", 2.75)            # 멘트 2.35 s + 꼬리 0.4 s
    alive(b, "grip_wait", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    assert not b.due(3.2)
    assert b.due(3.25)
    b.played(3.25)
    assert b.still_wanted(5.0)
    assert not b.due(8.0)                        # 한 번만


def test_grip_stops_beep_when_mission_judges_grabbed():
    """잡음 판정 = 진동 반복이 멈추는 tick 에 미션이 단계를 idle 로 즉시 알린다."""
    b = beep()
    b.on_robot_state("grip_wait", 0.0)
    b.on_speech_end("response", 2.75)
    b.played(3.25)
    b.on_robot_state("idle", 5.0)
    assert not b.still_wanted(5.0)
    assert not b.due(5.5)


def test_grabbed_during_the_hint_never_rings():
    b = beep()
    b.on_robot_state("grip_wait", 0.0)
    b.on_robot_state("idle", 1.8)                # 멘트 중에 잡음 판정
    b.on_speech_end("response", 2.75)
    assert not b.due(3.25)
    assert not b.due(10.0)


def test_no_ring_without_a_speech_end_in_the_section():
    b = beep()
    b.on_robot_state("grip_wait", 0.0)
    for t in (1.0, 2.0, 3.0):
        b.on_robot_state("grip_wait", t)        # 1초마다 오는 같은 단계 알림
    assert not b.due(3.5)


def test_cut_beep_does_not_ring_again_in_the_same_section():
    b = beep()
    b.on_robot_state("grip_wait", 0.0)
    b.on_speech_end("response", 2.75)
    b.played(3.25)                               # 사용자가 말해 LLM 답에 끊겼다
    b.on_speech_end("response", 6.0)
    assert not b.due(7.0)


def test_next_approach_rings_again():
    b = beep()
    b.on_robot_state("grip_wait", 0.0)
    b.on_speech_end("response", 2.75)
    b.played(3.25)
    b.on_robot_state("idle", 5.0)
    b.on_robot_state("grip_wait", 60.0)         # 다른 사람에게 다시 손잡이를 내준다
    b.on_speech_end("response", 62.75)
    alive(b, "grip_wait", 61.0, 62.0, 63.0)
    assert b.due(63.25)


# ---- 울릴 때: 비상정지 ------------------------------------------------------------
def test_announced_estop_rings_once_after_the_emergency_ment():
    """주행 중 걸림 — "안전을 위해 멈추겠습니다. 관리자를 호출했습니다."(3.40 s) 뒤."""
    b = beep()
    b.on_robot_state("navigating", 0.0)
    b.on_robot_state("estopped", 0.6)            # 비상정지 상태는 1 Hz 로 온다
    b.on_speech_end("emergency", 3.80)
    alive(b, "estopped", 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6)
    assert not b.due(4.2)
    assert b.due(4.30)
    b.played(4.30)
    assert b.still_wanted(8.0)


def test_silent_estop_never_rings():
    """서 있을 때 조용히 걸린 비상정지(버튼·앱·통신·부팅 래치) — 미션도 말하지 않는다."""
    b = beep()
    b.on_robot_state("idle", 0.0)
    b.on_robot_state("estopped", 1.0)
    for t in (2.0, 3.0, 4.0):
        b.on_robot_state("estopped", t)
    assert not b.due(4.5)
    b.on_speech_end("response", 5.0)             # "비카야"에 "지금은 비상 멈춤 상태입니다."
    assert not b.due(6.0)


def test_estop_release_stops_the_beep_and_release_ment_does_not_ring():
    b = beep()
    b.on_robot_state("estopped", 0.0)
    b.on_speech_end("emergency", 3.80)
    b.played(4.30)
    b.on_robot_state("idle", 5.0)                # 관리자가 앱에서 해제
    assert not b.still_wanted(5.0)
    b.on_speech_end("emergency", 7.6)            # "비상멈춤이 해제되었습니다." 도 emergency
    assert not b.due(8.1)


def test_tipped_repeat_does_not_ring_twice():
    b = beep()
    b.on_robot_state("estopped", 0.0)
    b.on_speech_end("emergency", 4.93)           # 넘어짐 멘트 4.53 s + 꼬리
    b.played(5.43)
    b.on_speech_end("emergency", 14.93)          # 10 s 뒤 되풀이
    assert not b.due(15.43)


def test_speech_after_arming_pushes_the_beep_later():
    b = beep()
    b.on_robot_state("estopped", 0.0)
    b.on_speech_end("emergency", 2.32)
    b.on_speech_end("response", 4.0)             # 이어진 말이 있으면 그 뒤로 민다
    alive(b, "estopped", 1.0, 2.0, 3.0, 4.0)
    assert not b.due(4.2)
    assert b.due(4.5)


# ---- 울리지 않는 때 ---------------------------------------------------------------
@pytest.mark.parametrize("state", ["arrived", "waiting", "approaching", "navigating",
                                   "paused_handle", "idle"])
def test_other_stages_never_ring(state):
    """목적지 도착·대기 중 호출 위치 진동·손 놓침 등에는 울리지 않는다."""
    b = beep()
    b.on_robot_state(state, 0.0)
    b.on_speech_end("emergency", 1.0)
    b.on_speech_end("response", 2.0)
    assert not b.due(3.0)


def test_stale_state_reports_stop_and_block_the_beep():
    """미션이 죽어 상태 알림이 끊기면 울리지 않고, 울리던 것도 끊는다."""
    b = beep()
    b.on_robot_state("grip_wait", 0.0)
    b.on_speech_end("response", 2.75)
    assert not b.due(3.1 + 0.5)                  # 마지막 알림 3 s 뒤
    b.on_robot_state("grip_wait", 3.0)
    b.played(3.25)
    assert b.still_wanted(5.9)
    assert not b.still_wanted(6.1)

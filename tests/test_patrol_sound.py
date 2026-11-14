"""경비(순찰) 큰 소리 판정 — src/patrol_sound.py (계약: ~/vica_data/patrol_plan/SPEC.md 4.C, 2026-11-14).

마이크·ROS 없이 돈다. 네 묶음이다.
  1. 판정 규칙(PatrolSoundDetector) — 67/53 dB, 0.5 s 지속, 정지 1.0 s 뒤, 3 s 쉼, 로봇이 말하는 동안
  2. 연결(PatrolSoundTap) — 원음 채널만, sound JSON 하나만 발행, 판정 상태가 아니면 꺼짐
  3. 감시 루프 배선(WakewordMonitor.run) — 순찰이 아니면 콜백·루프가 이 모듈을 부르지 않는다
  4. 11-11 실측 자료 회귀 — 주행 구간에 67 dB·0.5 s 규칙을 적용하면 사건 0건(자료가 없으면 건너뜀)
"""
from __future__ import annotations

import bisect
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from src import mission_phrases as mp
from src import patrol_sound as ps
from src.ment_cache import ASSETS_DIR, MentCache
from src.patrol_sound import (PatrolSoundConfig, PatrolSoundDetector, PatrolSoundTap, config_from_env,
                              level_db, sound_watched)
from src.wakeword_monitor import FRAME, SAMPLE_RATE, WakewordMonitor

ROOT = Path(__file__).resolve().parents[1]
DT = 0.08
OTHER_STATES = ("", "idle", "navigating", "waiting", "approaching", "estopped")


def feed(det, db, n, t0=0.0, dt=DT):
    """같은 크기의 블록 n 개를 넣고 (사건 목록, 다음 블록 시각)을 돌려준다."""
    events = []
    for i in range(n):
        event = det.feed_db(db, t0 + i * dt)
        if event is not None:
            events.append(event)
    return events, t0 + n * dt


def moving():
    det = PatrolSoundDetector()
    det.set_state("patrolling", 0.0)
    return det


# ---- 1. 판정 규칙 -------------------------------------------------------------------------

def test_block_length_is_the_microphone_frame():
    assert ps.BLOCK_SEC == FRAME / SAMPLE_RATE


def test_level_matches_the_phone_calibration():
    # 11-11 맞춤: 원음 −47.5 dBFS ↔ 폰 소음계 60 dB → 더할 값 107.5
    rms = 32768 * 10 ** (-47.5 / 20)
    assert level_db(rms * rms, 107.5) == pytest.approx(60.0, abs=1e-6)
    assert level_db(32768.0 ** 2, 107.5) == pytest.approx(107.5)
    assert level_db(0.0, 107.5) == pytest.approx(-42.8, abs=0.1)   # 무음도 숫자가 나온다


def test_half_second_is_seven_blocks_of_80ms():
    assert PatrolSoundDetector().hold_blocks == 7                 # 6개 = 0.48 s 는 모자란다
    assert PatrolSoundDetector(block_sec=0.1).hold_blocks == 5    # 실측 자료(0.1 s 블록)


def test_moving_fires_once_after_half_second_over_67():
    det = moving()
    events, t = feed(det, 70.0, 6)
    assert events == []
    event = det.feed_db(72.5, t)
    assert event == (72.5, 67.0)          # (이어진 구간의 최대, 기준)


def test_exactly_at_threshold_counts_and_just_below_does_not():
    assert len(feed(moving(), 67.0, 7)[0]) == 1
    assert feed(moving(), 66.9, 100)[0] == []


def test_a_dip_restarts_the_half_second():
    det = moving()
    _, t = feed(det, 70.0, 6)
    _, t = feed(det, 60.0, 1, t)
    events, t = feed(det, 70.0, 6, t)
    assert events == []
    assert det.feed_db(70.0, t) is not None


@pytest.mark.parametrize("state", OTHER_STATES + ("patrol_alarm", "patrol_light", "patrol_paused"))
def test_states_without_judgement_never_fire(state):
    det = PatrolSoundDetector()
    det.set_state(state, 0.0)
    assert det.threshold(5.0) is None
    assert feed(det, 100.0, 100)[0] == []


def test_dwell_uses_67_for_the_first_second_then_53():
    det = PatrolSoundDetector()
    det.set_state("patrol_dwell", 10.0)
    assert det.threshold(10.0) == 67.0 and det.threshold(10.99) == 67.0
    assert det.threshold(11.0) == 53.0
    events, t = feed(det, 60.0, 12, 10.0)          # 10.00~10.88 — 아직 67 dB
    assert events == []
    events, t = feed(det, 60.0, 9, t)              # 10.96 은 67, 11.04 부터 53 → 7번째인 11.52 에서
    assert events == [(60.0, 53.0)]
    assert t == pytest.approx(11.68)


def test_dwell_below_53_is_quiet():
    det = PatrolSoundDetector()
    det.set_state("patrol_dwell", 0.0)
    assert feed(det, 52.9, 100, 2.0)[0] == []


def test_loud_sound_across_the_dwell_settle_keeps_counting():
    det = PatrolSoundDetector()
    det.set_state("patrol_dwell", 0.0)
    events, t = feed(det, 70.0, 6, 0.76)           # 0.76·0.84·0.92 는 67 기준, 1.00 부터 53 기준
    assert events == []
    assert det.feed_db(70.0, t) == (70.0, 53.0)


def test_repeated_state_message_does_not_restart_the_dwell_clock():
    det = PatrolSoundDetector()
    det.set_state("patrol_dwell", 0.0)
    det.set_state("patrol_dwell", 1.0)             # /vica/robot_state 1 Hz 반복
    det.set_state("patrol_dwell", 2.0)
    assert det.threshold(2.0) == 53.0
    det.set_state("patrolling", 3.0)
    det.set_state("patrol_dwell", 20.0)            # 다음 지점 — 다시 1.0 s 를 기다린다
    assert det.threshold(20.5) == 67.0 and det.threshold(21.0) == 53.0


def test_rests_three_seconds_after_an_event():
    det = moving()
    events, t = feed(det, 70.0, 7)                 # 0.48 에서 사건 → 3.48 까지 쉼
    assert len(events) == 1
    events, t = feed(det, 70.0, 37, t)             # 0.56~3.44
    assert events == []
    events, t = feed(det, 70.0, 7, t)              # 3.52~4.00
    assert len(events) == 1


def test_robot_speech_is_not_judged_and_cuts_the_run():
    det = moving()
    det.set_speaking(True)
    events, t = feed(det, 90.0, 50)
    assert events == []
    det.set_speaking(False)
    _, t = feed(det, 70.0, 4, t)
    det.set_speaking(True)
    _, t = feed(det, 70.0, 1, t)
    det.set_speaking(False)
    events, t = feed(det, 70.0, 6, t)              # 말이 끼어 이어진 시간이 0 으로 돌아갔다
    assert events == []
    assert det.feed_db(70.0, t) is not None


def test_blocks_apart_do_not_chain():
    det = moving()
    _, t = feed(det, 70.0, 6)
    events, t = feed(det, 70.0, 6, t + 2.0)        # 마이크가 2 s 끊겼다 다시 옴
    assert events == []


def test_config_from_env_reads_three_values_and_ignores_garbage():
    assert config_from_env({}) == PatrolSoundConfig()
    cfg = config_from_env({"VICA_PATROL_DB_OFFSET": "105", "VICA_PATROL_DB_MOVING": " 70 ",
                           "VICA_PATROL_DB_STOPPED": "55.5"})
    assert (cfg.db_offset, cfg.db_moving, cfg.db_stopped) == (105.0, 70.0, 55.5)
    bad = config_from_env({"VICA_PATROL_DB_OFFSET": "", "VICA_PATROL_DB_MOVING": "크게",
                           "VICA_PATROL_DB_STOPPED": "nan"})
    assert bad == PatrolSoundConfig()
    assert (PatrolSoundConfig().db_offset, PatrolSoundConfig().db_moving,
            PatrolSoundConfig().db_stopped) == (107.5, 67.0, 53.0)


# ---- 2. 연결 ------------------------------------------------------------------------------

class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


def amp(db):
    """원음 네 채널을 이 값으로 채우면 그 dB 가 나온다(더할 값 107.5)."""
    return int(round(32768 * 10 ** ((db - 107.5) / 20)))


def block(raw_db, ch0=0, ch5=0):
    out = np.zeros((FRAME, 6), dtype=np.int16)
    out[:, 0] = ch0
    out[:, 1:5] = amp(raw_db)
    out[:, 5] = ch5
    return out


def make_tap(**kwargs):
    out, logs, clock = [], [], Clock()
    tap = PatrolSoundTap(emit=out.append, cfg=PatrolSoundConfig(), log=logs.append, clock=clock,
                         wall=lambda: 1763100000.1234, **kwargs)
    return tap, out, logs, clock


def push(tap, clock, blk, n):
    for _ in range(n):
        clock.t += DT
        tap.push(blk)


def kinds(out):
    return [json.loads(s)["kind"] for s in out]


def sounds(out):
    return [json.loads(s) for s in out if json.loads(s)["kind"] == "sound"]


@pytest.mark.parametrize("state", ["patrolling", "patrol_dwell"])
def test_armed_while_judging(state):
    tap, *_ = make_tap()
    assert tap.armed is False
    tap.set_state(state)
    assert tap.armed is True


@pytest.mark.parametrize("state", OTHER_STATES + (None, "patrol_alarm", "patrol_light", "patrol_paused"))
def test_not_armed_in_any_other_state(state):
    """경보·일시정지도 듣지 않는 상태다 — armed 가 False 라 콜백·감시 루프가 아예 부르지 않는다."""
    tap, out, logs, _ = make_tap()
    tap.set_state(state)
    assert tap.armed is False and out == [] and logs == []
    tap.set_state("patrolling")
    tap.set_state(state)
    assert tap.armed is False


# ---- 감시 항목 '큰 소리' 켜고 끄기(2026-11-14 사용자 요청) ----------------------------


def test_watch_message_is_read_as_sound_on_or_off():
    assert sound_watched('{"off": []}') is True
    assert sound_watched('{"off": ["light", "intruder"]}') is True
    assert sound_watched('{"off": ["sound"]}') is False
    for bad in ("", "nope", "[]", "{}", '{"off": "sound"}', "null"):
        assert sound_watched(bad) is None


@pytest.mark.parametrize("state", ["patrolling", "patrol_dwell"])
def test_sound_switched_off_is_never_armed(state):
    tap, out, logs, clock = make_tap()
    tap.set_watch(False)
    tap.set_state(state)
    assert tap.armed is False and logs == []          # 켠 적이 없으니 '끔' 줄도 없다
    tap.set_state("idle")
    tap.set_state(state)
    assert tap.armed is False


def test_switching_sound_off_during_a_patrol_disarms_at_once_and_on_rearms():
    tap, out, logs, clock = make_tap()
    tap.set_state("patrolling")
    push(tap, clock, block(75.0), 3)
    tap.set_watch(False)
    assert tap.armed is False and logs[-1] == "[경비] 소리 판정 끔"
    tap.set_watch(True)
    assert tap.armed is True
    push(tap, clock, block(75.0), 4)                  # 끄기 전 3블록은 버렸다 — 7블록이 안 된다
    tap.poll()
    assert out == []
    push(tap, clock, block(75.0), 3)
    tap.poll()
    assert kinds(out) == ["sound"]


def test_the_same_watch_value_again_does_nothing():
    tap, out, logs, clock = make_tap()
    tap.set_state("patrolling")
    push(tap, clock, block(75.0), 3)
    lines = len(logs)
    tap.set_watch(True)
    assert tap.armed is True and len(logs) == lines
    push(tap, clock, block(75.0), 4)
    tap.poll()
    assert kinds(out) == ["sound"]                    # 쌓인 블록을 비우지 않았다


def test_sound_event_json_is_the_contract():
    tap, out, _, clock = make_tap()
    tap.set_state("patrolling")
    push(tap, clock, block(70.0), 7)
    tap.poll()
    assert len(out) == 1
    payload = json.loads(out[0])
    assert set(payload) == {"src", "kind", "stamp", "db", "threshold"}
    assert payload["src"] == "audio" and payload["kind"] == "sound"
    assert payload["threshold"] == 67.0
    assert payload["db"] == pytest.approx(70.0, abs=0.05)
    assert payload["stamp"] == 1763100000.123


def test_only_the_four_raw_channels_are_measured():
    tap, out, _, clock = make_tap()
    tap.set_state("patrolling")
    push(tap, clock, block(40.0, ch0=30000, ch5=30000), 20)   # 칩 처리음·재생 참조만 크다
    tap.poll()
    assert out == []
    push(tap, clock, block(70.0), 7)
    tap.poll()
    assert len(sounds(out)) == 1


def test_dwell_listens_at_53_after_one_second():
    tap, out, _, clock = make_tap()
    tap.set_state("patrol_dwell")
    push(tap, clock, block(60.0), 12)     # 0.96 s — 아직 67 dB
    tap.poll()
    assert out == []
    push(tap, clock, block(60.0), 7)
    tap.poll()
    assert [(s["threshold"], s["db"]) for s in sounds(out)] == [(53.0, pytest.approx(60.0, abs=0.05))]


def test_robot_speech_pauses_the_judgement():
    tap, out, _, clock = make_tap()
    tap.set_state("patrolling")
    tap.set_speaking(True)
    push(tap, clock, block(90.0), 20)
    tap.poll()
    assert "sound" not in kinds(out)
    tap.set_speaking(False)
    push(tap, clock, block(90.0), 7)
    tap.poll()
    assert kinds(out).count("sound") == 1


def test_nothing_but_sound_is_published():
    """읽는 곳이 없는 audio status 는 내지 않는다(2026-11-14 감사) — 조용한 순찰 10 s 동안 발행 0."""
    tap, out, _, clock = make_tap()
    for i in range(125):                  # 10 s
        if i % 12 == 0:
            tap.set_state("patrolling" if i < 60 else "patrol_dwell")   # 미션의 1 Hz 반복
        push(tap, clock, block(45.0), 1)
        tap.poll()
    assert tap.armed is True and out == []
    push(tap, clock, block(70.0), 7)
    tap.poll()
    assert kinds(out) == ["sound"]


def test_mission_state_coming_back_rearms():
    tap, out, logs, clock = make_tap()
    tap.set_state("patrolling")
    clock.t += 3.5                        # 미션 상태가 3 s 넘게 끊김
    tap.poll()
    assert tap.armed is False
    tap.set_state("patrolling")           # 같은 값으로 돌아와도 다시 켜진다
    assert tap.armed is True
    push(tap, clock, block(70.0), 7)
    tap.poll()
    assert kinds(out) == ["sound"]


def test_leaving_patrol_drops_the_backlog():
    tap, out, logs, clock = make_tap()
    tap.set_state("patrolling")
    push(tap, clock, block(70.0), 6)      # 판정 전에 순찰이 끝났다
    tap.set_state("idle")
    clock.t += 60.0
    tap.set_state("patrolling")
    push(tap, clock, block(70.0), 1)
    tap.poll()
    assert out == []
    assert [line[:12] for line in logs] == ["[경비] 소리 판정 켬", "[경비] 소리 판정 끔", "[경비] 소리 판정 켬"]


def test_publish_failure_does_not_escape():
    logs, clock = [], Clock()

    def broken(_data):
        raise RuntimeError("publisher gone")

    tap = PatrolSoundTap(emit=broken, cfg=PatrolSoundConfig(), log=logs.append, clock=clock)
    tap.set_state("patrolling")
    for _ in range(3):
        push(tap, clock, block(70.0), 7)
        clock.t += 4.0
        tap.poll()                        # 예외가 나오면 감시 스레드가 죽는다
    assert sum("판정 오류" in line for line in logs) == 1


def test_bad_block_in_the_audio_callback_does_not_escape():
    tap, out, _, clock = make_tap()
    tap.set_state("patrolling")
    tap.push(np.zeros(FRAME, dtype=np.int16))    # 1차원 — 채널을 자를 수 없다
    tap.push(None)
    tap.poll()
    assert out == []


def test_tap_reads_thresholds_from_env_by_default(monkeypatch):
    monkeypatch.setenv("VICA_PATROL_DB_MOVING", "70")
    monkeypatch.delenv("VICA_PATROL_DB_STOPPED", raising=False)
    monkeypatch.delenv("VICA_PATROL_DB_OFFSET", raising=False)
    tap = PatrolSoundTap(emit=lambda s: None)
    assert (tap.cfg.db_moving, tap.cfg.db_stopped, tap.cfg.db_offset) == (70.0, 53.0, 107.5)


# ---- 3. 감시 루프 배선 ---------------------------------------------------------------------

class _Stop(Exception):
    """가짜 마이크의 블록을 다 쓰면 run() 을 끝낸다."""


class _FakeStream:
    def __init__(self, blocks, callback):
        self._blocks, self._callback = blocks, callback

    def __enter__(self):
        for blk in self._blocks:
            self._callback(blk.tobytes(), len(blk), None, None)
        return self

    def __exit__(self, *exc):
        return False


class _FakeSd:
    def __init__(self, blocks):
        self._blocks = blocks
        self.opened = 0

    def query_devices(self):
        return [{"name": "ReSpeaker 4 Mic Array (UAC1.0)", "max_input_channels": 6}]

    def RawInputStream(self, callback, **_kwargs):  # noqa: N802 — sounddevice 이름 그대로
        self.opened += 1
        return _FakeStream(self._blocks, callback)


class _FakeDsp:
    available = True

    def speech_detected(self):
        return False

    def doa_angle(self):
        return None

    def voice_activity(self):
        return False


class _SpyTap:
    def __init__(self, armed):
        self.armed = armed
        self.pushed, self.polls = [], 0

    def push(self, blk):
        self.pushed.append((blk.shape, blk.dtype, int(blk[0, 0]), int(blk[0, 1]), int(blk[0, 5])))

    def poll(self):
        self.polls += 1


def run_monitor(monkeypatch, blocks, tap="none"):
    """가짜 마이크로 WakewordMonitor.run() 을 돌리고 process_frame 이 받은 프레임을 돌려준다."""
    sd = _FakeSd(blocks)
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    monkeypatch.setattr("src.dsp_state.DspState", _FakeDsp)
    frames = []

    def predict(frame):
        frames.append(frame.copy())
        if len(frames) == len(blocks):
            raise _Stop
        return {"a": 0.0, "b": 0.0}

    monitor = WakewordMonitor(on_emergency=lambda e: None, on_user_text=lambda t: None,
                              predict=predict, transcribe=lambda audio: "", log=lambda line: None)
    if tap != "none":
        monitor.patrol_tap = tap
    with pytest.raises(_Stop):
        monitor.run()
    assert sd.opened == 1                 # 마이크는 한 번만 연다
    return frames


def mic_blocks(n, raw_db=40.0):
    return [block(raw_db, ch0=7, ch5=9) for _ in range(n)]


def test_monitor_has_no_tap_by_default(monkeypatch):
    monitor = WakewordMonitor(on_emergency=lambda e: None, on_user_text=lambda t: None)
    assert monitor.patrol_tap is None
    frames = run_monitor(monkeypatch, mic_blocks(5))
    assert len(frames) == 5 and all(f.shape == (FRAME,) and (f == 7).all() for f in frames)


def test_unarmed_tap_is_never_called(monkeypatch):
    spy = _SpyTap(armed=False)
    frames = run_monitor(monkeypatch, mic_blocks(5), spy)
    assert spy.pushed == [] and spy.polls == 0       # 순찰이 아니면 추가 계산 0
    assert all((f == 7).all() for f in frames)       # 호출어·긴급어 쪽은 그대로 ch0 만 받는다


def test_armed_tap_gets_every_block_and_a_poll_per_loop(monkeypatch):
    spy = _SpyTap(armed=True)
    frames = run_monitor(monkeypatch, mic_blocks(5), spy)
    assert spy.pushed == [((FRAME, 6), np.dtype(np.int16), 7, amp(40.0), 9)] * 5
    assert spy.polls == 5
    assert all((f == 7).all() for f in frames)


@pytest.mark.parametrize("state", ["navigating", "patrol_alarm", "patrol_paused"])
def test_real_tap_outside_judged_states_does_nothing_in_the_loop(monkeypatch, state):
    out = []
    tap = PatrolSoundTap(emit=out.append, cfg=PatrolSoundConfig())
    tap.set_state(state)

    def forbidden(*_args):
        raise AssertionError("순찰이 아닌데 불렸다")

    tap.push = tap.poll = forbidden
    run_monitor(monkeypatch, mic_blocks(10, raw_db=90.0), tap)
    assert out == []


def test_real_tap_in_patrol_reports_a_loud_sound_through_the_loop(monkeypatch):
    out = []
    tap = PatrolSoundTap(emit=out.append, cfg=PatrolSoundConfig())
    tap.set_state("patrolling")
    frames = run_monitor(monkeypatch, mic_blocks(8, raw_db=75.0), tap)
    assert len(frames) == 8
    assert [(json.loads(s)["kind"], json.loads(s)["threshold"]) for s in out] == [("sound", 67.0)]
    assert json.loads(out[0])["db"] == pytest.approx(75.0, abs=0.05)


def test_node_connects_the_tap_with_one_new_publisher():
    src = (ROOT / "src" / "ros_wakeword_node.py").read_text(encoding="utf-8")
    assert 'self.create_publisher(String, "/vica/patrol/event", 10)' in src
    assert "self._patrol.set_state(msg.dialog_state)" in src        # 이미 있는 /vica/robot_state 구독
    assert "self._patrol.set_speaking(self._tts_speaking)" in src   # 이미 있는 /vica/tts_state 구독
    # 감시 스레드는 run() 에 들어갈 때 tap 을 한 번만 읽는다 — 스레드를 띄우기 전에 붙여야 한다.
    assert src.index("self._monitor.patrol_tap = self._patrol") < src.index("self._thread.start()")
    patrol_lines = [line for line in src.splitlines() if "patrol" in line.lower()]
    assert not any("create_timer" in line for line in patrol_lines)
    # 순찰로 늘어난 구독은 감시 항목 알림 하나다(마지막 값 보관이라 순찰 사이에는 오지 않는다).
    assert src.count('"/vica/patrol/watch"') == 1
    assert 'String, "/vica/patrol/watch", self._on_patrol_watch,' in src
    assert "durability=DurabilityPolicy.TRANSIENT_LOCAL))" in src.split('"/vica/patrol/watch"')[1][:200]


# ---- 문장 2개 -----------------------------------------------------------------------------

def test_patrol_sentences_are_the_spec_text():
    assert mp.PATROL_INTRUDER == "침입자 발생"
    assert mp.PATROL_LIGHT == "불 켜짐 감지"
    assert mp.PATROL_MENTS == {"mission_msg_patrol_intruder": "침입자 발생",
                               "mission_msg_patrol_light": "불 켜짐 감지"}


def test_patrol_sentences_are_in_the_recording_list_with_the_same_text():
    manifest = json.loads((ASSETS_DIR / "baked" / "manifest.json").read_text(encoding="utf-8"))
    for stem, text in mp.PATROL_MENTS.items():
        assert manifest.get(f"{stem}.wav") == text, stem


def test_patrol_sentences_are_recorded():
    """wav 가 있어야 한다 — /assets/ 가 gitignore 라 커밋 때 git add -f 를 빠뜨리면 여기서 걸린다."""
    cache = MentCache(ASSETS_DIR)
    for stem, text in mp.PATROL_MENTS.items():
        assert (ASSETS_DIR / "baked" / f"{stem}.wav").exists(), stem
        wav, rate = cache.lookup(text)
        assert rate == 16000 and len(wav) > 0
        assert f"baked/{stem}.wav" not in cache.missing


def test_patrol_sentences_stay_out_of_the_female_voice_batch():
    """bake_one_cv.py --mission --force 는 이 표를 여성(F2) 목소리로 다시 굽는다 — 남성 목소리 녹음을 덮어쓰면 안 된다."""
    assert not set(mp.PATROL_MENTS) & set(mp.baked_mission_ments())


@pytest.mark.parametrize("mission_pkg", [
    ROOT.parent / "vica_ros2_ws" / "src" / "vica_mission_manager",
    Path.home() / "wt-patrol" / "src" / "vica_mission_manager",
], ids=["vica_ros2_ws", "wt-patrol"])
def test_patrol_sentences_match_the_mission_text(mission_pkg):
    """미션 MSG_PATROL_* 와 글자까지 같아야 녹음·합성 보관함이 맞는다. 미션에 아직 없으면 건너뛴다."""
    path = mission_pkg / "vica_mission_manager" / "mission_logic.py"
    if not path.exists():
        pytest.skip(f"미션 파일 없음: {path}")
    found = dict(re.findall(r'^(MSG_PATROL_(?:INTRUDER|LIGHT))\s*=\s*"([^"]*)"',
                            path.read_text(encoding="utf-8"), flags=re.M))
    if len(found) < 2:
        pytest.skip(f"미션에 MSG_PATROL_INTRUDER·MSG_PATROL_LIGHT 가 아직 없다: {path}")
    assert found == {"MSG_PATROL_INTRUDER": mp.PATROL_INTRUDER, "MSG_PATROL_LIGHT": mp.PATROL_LIGHT}


# ---- 4. 11-11 실측 자료 회귀 ---------------------------------------------------------------
# levels_*.csv: sound_level.py 가 0.1 s 블록마다 적은 est_db(= 원음 dBFS + 107.5). track.csv: 같은 주행의 v·w.

DATA = Path.home() / "vica_data" / "patrol_tests"
RUNS = [("levels_171957.csv", "171955"), ("levels_172744.csv", "172741")]


def _tod(text):
    return int(text[:2]) * 3600 + int(text[3:5]) * 60 + float(text[6:])


def _load(levels, run):
    level_path = DATA / "2026-11-11_sound" / levels
    track_path = DATA / "2026-11-11_waypoint" / run / "track.csv"
    if not (level_path.exists() and track_path.exists()):
        pytest.skip(f"실측 자료 없음: {level_path} / {track_path}")
    with open(level_path, encoding="utf-8") as f:
        sound = [(_tod(r["time"]), float(r["est_db"])) for r in csv.DictReader(f) if r["est_db"]]
    with open(track_path, encoding="utf-8") as f:
        track = [(_tod(r["time"]), abs(float(r["v"])), abs(float(r["w"]))) for r in csv.DictReader(f)]
    return sound, track


def _replay(sound, state_at, cfg=None):
    """(시각, 사건) 목록 — 자료가 0.1 s 블록이라 5개 연속이 0.5 s 다."""
    det = PatrolSoundDetector(cfg or PatrolSoundConfig(), block_sec=0.1)
    events = []
    for t, db in sound:
        det.set_state(state_at(t), t)
        event = det.feed_db(db, t)
        if event is not None:
            events.append((t, event))
    return events


@pytest.mark.parametrize("levels, run, drive_sec", [RUNS[0] + (149,), RUNS[1] + (467,)])
def test_measured_drive_gives_no_event_at_67db_for_half_a_second(levels, run, drive_sec):
    sound, track = _load(levels, run)
    start, end = track[0][0], track[-1][0]
    in_drive = [db for t, db in sound if start <= t <= end]
    assert len(in_drive) * 0.1 >= drive_sec           # 주행 구간 전체가 소리 기록에 들어 있다
    assert max(in_drive) >= 67.0                      # 순간값은 기준을 넘는다 — 0.5 s 지속이 걸러낸다
    events = _replay(sound, lambda t: "patrolling" if start <= t <= end else "idle")
    assert events == []


@pytest.mark.parametrize("levels, run, peak", [RUNS[0] + (81.0,), RUNS[1] + (69.2,)])
def test_measured_drive_would_alarm_with_a_shorter_hold(levels, run, peak):
    """지속을 0.3 s 로 줄이면 주행마다 1건(README 의 81.0·69.2 dB) — 0.5 s 가 필요한 근거이자 회귀의 대조군."""
    sound, track = _load(levels, run)
    start, end = track[0][0], track[-1][0]
    events = _replay(sound, lambda t: "patrolling" if start <= t <= end else "idle",
                     PatrolSoundConfig(hold_sec=0.3))
    assert [event for _, event in events] == [(peak, 67.0)]


@pytest.mark.parametrize("levels, run, after", [RUNS[0] + ([(69.9, 53.0)],), RUNS[1] + ([],)])
def test_measured_runs_with_stops_as_dwell(levels, run, after):
    """완전 정지(|v|<0.03 m/s·|w|<0.1 rad/s)를 patrol_dwell, 주행 뒤 대기도 patrol_dwell 로 놓는다.

    주행 구간은 두 번 다 0건. 171957 주행 뒤 17:22:59 의 69.9 dB(사람 소리) 한 건만 53 dB 기준에 잡힌다.
    """
    sound, track = _load(levels, run)
    times = [row[0] for row in track]
    start, end = times[0], times[-1]

    def state_at(t):
        if t < start:
            return "idle"
        if t > end:
            return "patrol_dwell"
        _, v, w = track[min(bisect.bisect_left(times, t), len(track) - 1)]
        return "patrol_dwell" if v < 0.03 and w < 0.1 else "patrolling"

    events = _replay(sound, state_at)
    assert [event for t, event in events if t <= end] == []
    assert [event for t, event in events if t > end] == after


class TestStaleMissionState:
    """미션 상태(1 Hz)가 3 s 넘게 끊기면 판정을 끈다 — 미션이 순찰 상태에서 멈춰도 계산·발행을 이어 가지 않는다."""

    def _tap(self):
        from src.patrol_sound import PatrolSoundTap
        clock = {"t": 100.0}
        sent = []
        tap = PatrolSoundTap(emit=sent.append, clock=lambda: clock["t"], wall=lambda: 0.0)
        return tap, clock, sent

    def test_disarms_after_three_seconds_of_silence(self):
        tap, clock, sent = self._tap()
        tap.set_state("patrolling")
        assert tap.armed
        clock["t"] += 2.9
        tap.poll()
        assert tap.armed
        clock["t"] += 0.2
        tap.poll()
        assert not tap.armed

    def test_repeated_same_state_keeps_it_armed(self):
        tap, clock, sent = self._tap()
        tap.set_state("patrolling")
        for _ in range(6):
            clock["t"] += 1.0
            tap.set_state("patrolling")   # 미션의 1 Hz 반복
            tap.poll()
        assert tap.armed
